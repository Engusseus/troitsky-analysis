"""Rule boundaries for the Troitsky 2027 rule set (``rules/troitsky_2027.yaml``).

Band tables are tested directly (value -> judges' rounding -> first matching band) and
through ``evaluate()`` on generated Warren trusses whose geometry is derived by hand from
``WarrenParams`` and the stick-stack section sizes (nothing depends on the particular
default sections).
"""

from __future__ import annotations

import math

import pytest

from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.mass import bridge_mass
from bridgesim.measure import SAMPLE_MM, measure
from bridgesim.rules import Rule, RuleSet, RulesReport, evaluate, mass_penalty, round_half_up
from bridgesim.schema import Bridge, Material, Prop, Stick
from bridgesim.sections import stick_stack_dims

BANS_125_127 = ["12.5", "12.6", "12.7"]


# --------------------------------------------------------------------------- helpers

def get_rule(ruleset: RuleSet, key: str) -> Rule:
    return next(r for r in ruleset.rules if r.key == key)


def band_outcome(rule: Rule, value: float, step: float = 1.0) -> tuple[float, list[str]]:
    """Judges' rounding then first matching band, exactly as the rulebook table reads."""
    v = round_half_up(value, step)
    band = next(b for b in rule.bands if b.matches(v))
    return band.penalty, list(band.bans)


def result(report: RulesReport, key: str):
    return next(r for r in report.results if r.key == key)


def dims(params: WarrenParams, group: str) -> tuple[float, float]:
    """(b, d) of a generator section group: flat -> (w, n t), on_edge -> (n t, w)."""
    s, st = params.sections[group], Stick()
    return stick_stack_dims(s.sticks, s.layout, st.width_mm, st.thickness_mm)


def pier_face_offset(params: WarrenParams) -> float:
    """Half-size along X of the members standing on the table at each pier line.

    Vertical pier: local y = -X, so its depth d lies along X -> d_pier / 2.
    Pier braces lie in the x = const plane with local z = -+X -> b_pier_brace / 2.
    """
    return max(dims(params, "pier")[1], dims(params, "pier_brace")[0]) / 2.0


def bottom_chord_y(params: WarrenParams) -> float:
    """y_b = deck top - deck thickness - d_floor_beam / 2 (deck rests on the floor beams)."""
    return (params.deck_top_elevation_mm - params.deck_thickness_mm
            - dims(params, "floor_beam")[1] / 2.0)


def evaluate_params(material: Material, **kw) -> RulesReport:
    return evaluate(generate_warren(WarrenParams(**kw)), material)


# --------------------------------------------------------------------------- rounding

def test_round_half_up_examples() -> None:
    """Judges round halves up: 999.5 mm -> 1000 mm, 6.005 kg -> 6.01 kg (binary floats
    such as 6.005 = 6.00499999... must still round up), 2.675 -> 2.68.
    """
    assert round_half_up(999.5, 1) == 1000
    assert round_half_up(6.005, 0.01) == 6.01
    assert round_half_up(2.675, 0.01) == 2.68
    assert round_half_up(999.49, 1) == 999
    assert round_half_up(1200.5, 1) == 1201
    assert round_half_up(math.inf, 1) == math.inf
    assert math.isnan(round_half_up(math.nan, 1))


# --------------------------------------------------------------------------- band tables

def test_every_band_rule_covers_each_integer_exactly_once(ruleset: RuleSet) -> None:
    """After rounding to whole mm every value 0..3000 must match exactly one band
    (no gaps, which would raise in evaluate(); no overlaps, which would hide a band).
    """
    for rule in (r for r in ruleset.rules if r.type == "bands"):
        for v in range(0, 3001):
            n = sum(b.matches(float(v)) for b in rule.bands)
            assert n == 1, f"{rule.key}: {v} mm matches {n} bands"


@pytest.mark.parametrize(
    ("clear_span", "penalty", "bans"),
    [
        (1000, 0, []),
        (1200, 0, []),
        (1201, 5, []),
        (999, 5, []),
        (950, 5, []),
        (949, 10, BANS_125_127),
        (999.5, 0, []),  # rounds to 1000
        (999.49, 5, []),  # rounds to 999
        (1200.49, 0, []),  # rounds to 1200
        (1200.5, 5, []),  # rounds to 1201
        (949.5, 5, []),  # rounds to 950
        (949.49, 10, BANS_125_127),  # rounds to 949
    ],
)
def test_span_band_table(ruleset: RuleSet, clear_span: float, penalty: float,
                         bans: list[str]) -> None:
    """§8.2.1.1: 1000-1200 -> 0; >= 1201 -> -5; 950-999 -> -5; <= 949 -> -10 and bans
    §12.5-12.7, after rounding to the nearest mm (halves up).
    """
    rule = get_rule(ruleset, "span_length")
    step = ruleset.rounding["length_mm"]
    assert band_outcome(rule, clear_span, step) == (penalty, bans)


@pytest.mark.parametrize(
    ("deck_width", "penalty", "bans"),
    [(150, 0, []), (170, 0, []), (149, 5, []), (91, 5, []), (90, 10, BANS_125_127),
     (149.5, 0, []), (90.49, 10, BANS_125_127)],
)
def test_deck_width_band_table(ruleset: RuleSet, deck_width: float, penalty: float,
                               bans: list[str]) -> None:
    """§8.2.3.1: >= 150 -> 0; 91-149 -> -5; <= 90 -> -10 and bans §12.5-12.7."""
    assert band_outcome(get_rule(ruleset, "deck_width"), deck_width) == (penalty, bans)


# --------------------------------------------------------------------------- span (integration)

@pytest.mark.parametrize(
    ("clear_span", "penalty", "bans"),
    [(1000, 0, []), (999, 5, []), (950, 5, []), (949, 10, BANS_125_127)],
)
def test_span_rule_through_evaluate(material: Material, clear_span: float, penalty: float,
                                    bans: list[str]) -> None:
    """Generated Warren with span_mm chosen so the inner-face clear span hits the boundary.

        clear span = span_cc - 2 f,  f = max(d_pier, b_pier_brace) / 2
    so span_cc = clear + 2 f. For these clear spans span_cc stays at or below 1200 + 2 f
    and the clear span decides the outcome (as long as 2 f <= 200 mm).
    """
    params = WarrenParams()
    span_cc = clear_span + 2 * pier_face_offset(params)
    bridge = generate_warren(WarrenParams(span_mm=span_cc))
    m = measure(bridge, material)
    assert m["clear_span_mm"] == pytest.approx(clear_span, abs=1e-6)
    assert m["span_cc_mm"] == pytest.approx(span_cc, abs=1e-6)
    r = result(evaluate(bridge, material), "span_length")
    assert (r.penalty, r.bans) == (penalty, bans)
    assert r.passed is (penalty == 0)


def test_span_rule_keeps_worst_of_clear_and_centre_spans(material: Material) -> None:
    """span_cc = 1205: the clear span 1205 - 2 f is in band (0) but centre-to-centre
    1205 >= 1201 (-5). The rule keeps the worst outcome, so -5 with the cc value reported.
    """
    assert 1000 <= 1205 - 2 * pier_face_offset(WarrenParams()) <= 1200
    r = result(evaluate_params(material, span_mm=1205.0), "span_length")
    assert r.penalty == 5
    assert r.measured == 1205
    assert not r.passed


def test_span_clear_1200_fails_on_centre_span(material: Material) -> None:
    """Clear span exactly 1200 is in band, but its centre-to-centre span is 1200 + 2 f
    (> 1200), so the stricter combined reading gives -5. (The band table alone says
    1200 -> 0, see test_span_band_table.)
    """
    span_cc = 1200 + 2 * pier_face_offset(WarrenParams())
    bridge = generate_warren(WarrenParams(span_mm=span_cc))
    assert measure(bridge, material)["clear_span_mm"] == pytest.approx(1200.0)
    r = result(evaluate(bridge, material), "span_length")
    assert r.penalty == 5
    assert r.measured == round_half_up(span_cc, 1)


# --------------------------------------------------------------------------- deck / total width

@pytest.mark.parametrize(
    ("deck_width", "penalty", "bans"),
    [(150, 0, []), (149, 5, []), (91, 5, []), (90, 10, BANS_125_127)],
)
def test_deck_width_through_evaluate(material: Material, deck_width: float, penalty: float,
                                     bans: list[str]) -> None:
    """The generator's deck clear width is measured as deck_width_mm and banded."""
    r = result(evaluate_params(material, deck_clear_width_mm=deck_width), "deck_width")
    assert r.measured == deck_width
    assert (r.penalty, r.bans) == (penalty, bans)


def test_total_width_boundary(material: Material) -> None:
    """Total width = w + b_truss + max(b_truss, b_pier): truss planes at
    +-(w + b_truss)/2 and the widest member in a truss plane (chords, diagonals, piers)
    sticks out by half its width b. Choosing w so the total is exactly 350 passes;
    1 mm wider (351) gives -5; w = 340 is far over.
    """
    params = WarrenParams()
    b_truss = max(dims(params, g)[0] for g in ("top_chord", "bottom_chord", "diagonal"))
    extra = b_truss + max(b_truss, dims(params, "pier")[0])
    for w, penalty in ((350 - extra, 0), (351 - extra, 5), (340, 5)):
        r = result(evaluate_params(material, deck_clear_width_mm=w), "total_width")
        assert r.measured == w + extra, w
        assert r.penalty == penalty, w
        assert r.passed is (penalty == 0)


# --------------------------------------------------------------------------- mass

@pytest.mark.parametrize(
    ("mass", "penalty"),
    [(0.0, 0), (6.00, 0), (6.01, 2), (6.50, 2), (6.51, 4), (7.00, 4), (7.01, 6),
     (15.00, 36), (15.01, 50), (20.0, 50), (6.004, 0), (6.005, 2)],
)
def test_mass_penalty_steps(ruleset: RuleSet, mass: float, penalty: float) -> None:
    """§8.8: free up to 6.00 kg, then -2 per started 0.50 kg, up to 15.00 kg; -50 above.

        6.01 -> ceil(0.01 / 0.5) = 1 step -> 2;  6.51 -> 2 steps -> 4;
        15.00 -> ceil(9.00 / 0.5) = 18 steps -> 36;  15.01 -> over cap -> 50.
    The mass is rounded half-up to 0.01 kg first (6.005 -> 6.01 -> 2).
    """
    steps = get_rule(ruleset, "mass").steps
    assert steps is not None
    assert mass_penalty(mass, steps) == penalty


@pytest.mark.parametrize(("target_kg", "penalty"), [(5.90, 0), (6.25, 2), (6.75, 4)])
def test_mass_rule_through_evaluate(default_bridge: Bridge, material: Material,
                                    target_kg: float, penalty: float) -> None:
    """Mass is linear in density, so scaling rho by target / m_default gives a bridge of
    the target mass: 5.90 kg -> 0, 6.25 kg -> 2 (1 started step), 6.75 kg -> 4 (2 steps).
    """
    m0 = bridge_mass(default_bridge, material).total_kg
    rho = material.density_kg_m3.value * target_kg / m0
    scaled = material.model_copy(update={"density_kg_m3": Prop(value=rho)})
    assert bridge_mass(default_bridge, scaled).total_kg == pytest.approx(target_kg)
    r = result(evaluate(default_bridge, scaled), "mass")
    assert r.measured == pytest.approx(target_kg)
    assert r.penalty == penalty
    assert r.passed is (penalty == 0)


# --------------------------------------------------------------------------- clear opening

def opening_margin(params: WarrenParams, opening: float = 100.0) -> float:
    """Nearest members above the deck at mid-span are the top struts at x = mid +- p/2,
    with width b along X: margin = p/2 - b_strut/2 - opening/2.
    """
    panel = params.span_mm / params.n_panels
    return panel / 2 - dims(params, "top_strut")[0] / 2 - opening / 2


def test_clear_opening_default_passes(default_params: WarrenParams, default_bridge: Bridge,
                                      material: Material) -> None:
    """Default 8 panels of 143.75 mm with 3-stick flat struts (b = 10):
    margin = 71.875 - 5 - 50 = 16.875 mm >= 0 -> pass.
    """
    m = measure(default_bridge, material)
    assert m["clear_opening_margin_mm"] == pytest.approx(opening_margin(default_params),
                                                         abs=1e-6)
    assert m["clear_opening_margin_mm"] >= 0
    r = result(evaluate(default_bridge, material), "clear_opening")
    assert r.passed is True
    assert r.bans == []


@pytest.mark.parametrize("n_panels", [8, 10, 12, 16])
def test_clear_opening_margin_vs_panel_count(material: Material, n_panels: int) -> None:
    """margin = (1150 / n) / 2 - b_strut / 2 - 50: with b_strut = 10, n = 10 -> +2.5 (pass),
    n = 12 -> -7.08 and n = 16 -> -19.06 (struts intrude into the square: fail).
    """
    params = WarrenParams(n_panels=n_panels)
    margin = opening_margin(params)
    m = measure(generate_warren(params), material)
    assert m["clear_opening_margin_mm"] == pytest.approx(margin, abs=1e-5)
    assert m["clear_opening_ok"] == float(margin >= 0)


def test_clear_opening_fails_with_dense_panels(material: Material) -> None:
    """n = 12 panels: struts at mid-span +- p/2 block the opening. Consequence (§8.9):
    no points deducted, but §12.5-12.7 are banned.
    """
    assert opening_margin(WarrenParams(n_panels=12)) < 0
    rep = evaluate_params(material, n_panels=12)
    r = result(rep, "clear_opening")
    assert r.passed is False
    assert r.penalty == 0
    assert r.bans == BANS_125_127
    assert set(BANS_125_127) <= set(rep.bans)
    assert not rep.all_passed


def test_clear_opening_fails_with_midspan_strut(default_bridge: Bridge,
                                                material: Material) -> None:
    """Add a top strut exactly at mid-span, 20 mm above the top-chord level (so its end nodes
    do not sit on the chords, which the schema rejects as hidden joints) and across the two
    truss planes. It crosses the centre of the 100 x 100 square:
    margin = max(|dx| - b/2, |dz|) - 50 ~ -50 (to within the 2 mm sampling step).
    """
    nodes = default_bridge.node_map()
    x_mid = 0.5 * (nodes["P0n"].x_mm + nodes[default_bridge.supports.roller[0]].x_mm)
    data = default_bridge.model_dump()
    data["nodes"] += [
        {"id": f"M{s}", "x_mm": x_mid, "y_mm": nodes[f"T0{s}"].y_mm + 20.0,
         "z_mm": nodes[f"T0{s}"].z_mm}
        for s in "nf"
    ]
    data["members"].append({"id": "ts_mid", "i": "Mn", "j": "Mf", "section": "top_strut",
                            "group": "top_strut"})
    bridge = Bridge.model_validate(data)
    m = measure(bridge, material)
    assert m["clear_opening_margin_mm"] == pytest.approx(-50.0, abs=SAMPLE_MM)
    assert "ts_mid" in m.details["clear_opening_crossing"]
    r = result(evaluate(bridge, material), "clear_opening")
    assert r.passed is False
    assert r.bans == BANS_125_127


# --------------------------------------------------------------------------- heights

def test_clearance_above_deck_boundary(material: Material) -> None:
    """Lowest member over the cart path is the top strut / top brace (depth d vertical):

        clearance = y_b + h - d_strut / 2 - deck top,   y_b = 200 - 2 - d_floor_beam / 2
    so the truss height giving exactly the 200 mm cart height is
        h* = 200 + deck top + d_strut / 2 - y_b      (= 223 mm for the defaults)
    h* passes; h* - 1 (199 mm clear) and h* - 23 fail with -5.
    """
    params = WarrenParams()
    d_low = max(dims(params, "top_strut")[1], dims(params, "top_brace")[1])
    h_star = 200.0 + params.deck_top_elevation_mm + d_low / 2 - bottom_chord_y(params)
    for h, ok in ((h_star, True), (h_star - 1, False), (h_star - 23, False)):
        bridge = generate_warren(WarrenParams(truss_height_mm=h))
        m = measure(bridge, material)
        assert m["clearance_above_deck_mm"] == pytest.approx(200.0 - (h_star - h), abs=1e-6)
        r = result(evaluate(bridge, material), "clearance_above_deck")
        assert r.passed is ok, h
        assert r.penalty == (0 if ok else 5)


def test_total_height_boundary(material: Material) -> None:
    """Total height = y_b + h + d_top_chord / 2 (table at 0; the top chord is the highest
    member). h chosen for exactly 550 passes; 1 mm more (551) gives -5; h = 520 also fails.
    """
    params = WarrenParams()
    base = bottom_chord_y(params) + dims(params, "top_chord")[1] / 2
    for h, penalty in ((550 - base, 0), (551 - base, 5), (520.0, 5)):
        r = result(evaluate_params(material, truss_height_mm=h), "total_height")
        assert r.measured == round_half_up(base + h, 1), h
        assert r.penalty == penalty, h


# --------------------------------------------------------------------------- piers

def test_inclined_pier_is_disqualification_risk(default_bridge: Bridge,
                                                material: Material) -> None:
    """Move the base of pier P0n 10 mm outwards: inclination = atan(10 / y_b) (about
    3.2 deg for y_b = 180) > 1 deg, so §8.1 flags an A-frame (disqualification), with no
    points deducted.
    """
    y_b = default_bridge.node_map()["B0n"].y_mm
    data = default_bridge.model_dump()
    for n in data["nodes"]:
        if n["id"] == "P0n":
            n["x_mm"] -= 10.0
    bridge = Bridge.model_validate(data)
    m = measure(bridge, material)
    assert m["pier_max_inclination_deg"] == pytest.approx(math.degrees(math.atan(10 / y_b)))
    rep = evaluate(bridge, material)
    r = result(rep, "piers_vertical")
    assert r.passed is False
    assert r.penalty == 0
    assert r.disqualification is True
    assert r in rep.disqualification_risks


# --------------------------------------------------------------------------- report

def test_report_aggregates(material: Material) -> None:
    """Narrow deck (90 mm): deck width -10 with bans, the cart path (150 mm) no longer fits
    (-5), so total penalty = sum of per-rule penalties and bans are the union.
    """
    rep = evaluate_params(material, deck_clear_width_mm=90.0)
    assert rep.total_penalty == sum(r.penalty for r in rep.results)
    assert rep.total_penalty >= 15
    assert rep.bans == sorted({b for r in rep.results for b in r.bans})
    assert not rep.all_passed
    assert result(rep, "clearance_above_deck").passed is False
    assert all(r.passed is None for r in rep.info)
    assert {r.key for r in rep.checked} | {r.key for r in rep.info} == {
        r.key for r in rep.results}


def test_table_level_tie_between_piers_blocks_clear_span_box(material: Material) -> None:
    """§8.6: a tie lying on the table between the pier bases blocks the 1000 x 150 mm box,
    even though it touches the table like the supports do."""
    from bridgesim.schema import Member

    b = generate_warren()
    n = b.metadata["params"]["n_panels"]
    tie = Member(id="tie", i="P0n", j=f"P{n}n", section="bottom_brace", group="other")
    b = b.model_copy(update={"members": [*b.members, tie]})
    m = measure(b, material)
    assert m["clear_box_free_length_mm"] < 1000
    rep = evaluate(b, material)
    assert next(r for r in rep.results if r.key == "clear_span_box").passed is False
