"""Geometric measurements (rulebook §8) of generated Warren trusses, checked against the
generator geometry derived by hand from ``WarrenParams`` and the stick-stack section sizes.

Generator geometry (see ``generators/warren.py``), w = 10 mm stick width, t = 2 mm:

    section (b, d): flat n sticks -> (w, n t); on_edge n sticks -> (n t, w)
    b_truss = max(b_top_chord, b_bottom_chord, b_diagonal)     (widest truss member)
    truss planes at z = +-s/2,  s = deck clear width + b_truss
    y_b = deck top - deck thickness - d_floor_beam / 2          (bottom chord)
    y_t = y_b + truss height                                     (top chord)
    piers: vertical at x = 0 and x = span, from y = 0 (table) to y_b

Member half-sizes along global axes follow the Pynite local axes: horizontal members have
d vertical; a vertical pier has d along X and b along Z; members along Z (floor beams,
struts) have b along X; the pier braces lie in an x = const plane with b along X.

No number here depends on a particular choice of default sections: every expectation is
computed from ``WarrenParams().sections``.
"""

from __future__ import annotations

import math

import pytest

from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.measure import DEFAULT_CONSTANTS, measure
from bridgesim.schema import Bridge, Material, Stick
from bridgesim.sections import stick_stack_dims

OPENING = DEFAULT_CONSTANTS["clear_opening_mm"]


def section_bd(p: WarrenParams, group: str) -> tuple[float, float]:
    """(b, d) of a generator section group: flat -> (w, n t), on_edge -> (n t, w)."""
    s, st = p.sections[group], Stick()
    return stick_stack_dims(s.sticks, s.layout, st.width_mm, st.thickness_mm)


def expected_measurements(p: WarrenParams) -> dict[str, float]:
    """Hand-derived measurements for a generated Warren truss (table at y = 0)."""

    def bd(group: str) -> tuple[float, float]:
        return section_bd(p, group)

    b_truss = max(bd(g)[0] for g in ("top_chord", "bottom_chord", "diagonal"))
    s = p.deck_clear_width_mm + b_truss
    y_b = p.deck_top_elevation_mm - p.deck_thickness_mm - bd("floor_beam")[1] / 2
    y_t = y_b + p.truss_height_mm
    panel = p.span_mm / p.n_panels

    # inner faces of the members on the table: pier depth or pier-brace width along X
    pier_face = max(bd("pier")[1], bd("pier_brace")[0]) / 2
    # outermost X faces of members at the pier lines (piers, pier braces, end floor beams)
    end_face = max(pier_face, bd("floor_beam")[0] / 2)

    # widest members in the truss planes (width b along Z): chords, diagonals and piers
    half_width = s / 2 + max(b_truss, bd("pier")[0]) / 2

    # highest point: members at y_t with depth vertical (top chord, struts, top braces) or
    # a diagonal end at y_t (its local y is tilted: vertical share = (p/2) / L_diag)
    L_diag = math.hypot(panel / 2, p.truss_height_mm)
    top_half = max(bd("top_chord")[1], bd("top_strut")[1], bd("top_brace")[1],
                   bd("diagonal")[1] * (panel / 2) / L_diag) / 2

    # lowest members over the cart path: struts / top braces (depth vertical)
    over_deck_low = y_t - max(bd("top_strut")[1], bd("top_brace")[1]) / 2

    deck_length = p.span_mm + 2 * p.deck_overhang_mm
    return {
        "span_cc_mm": p.span_mm,
        "clear_span_mm": p.span_mm - 2 * pier_face,
        "deck_length_mm": deck_length,
        "total_length_mm": max(deck_length, p.span_mm + 2 * end_face),
        "deck_height_mm": p.deck_top_elevation_mm,
        "total_height_mm": y_t + top_half,
        "deck_width_mm": p.deck_clear_width_mm,
        "total_width_mm": 2 * half_width,
        "clearance_above_deck_mm": over_deck_low - p.deck_top_elevation_mm,
        "cart_clear_width_mm": p.deck_clear_width_mm,
        "clear_box_free_length_mm": p.span_mm - 2 * pier_face,
        # nearest members above the deck at mid-span: top struts at x = mid +- p/2 (b along X)
        "clear_opening_margin_mm": panel / 2 - bd("top_strut")[0] / 2 - OPENING / 2,
        "pier_max_inclination_deg": 0.0,
        # bookkeeping for preconditions / details
        "_pier_face_mm": pier_face,
        "_end_face_mm": end_face,
        "_lowest_span_member_mm": min(y_b - bd(g)[1] / 2
                                      for g in ("bottom_chord", "floor_beam", "bottom_brace")),
    }


def test_default_hand_values(default_params: WarrenParams, default_bridge: Bridge) -> None:
    """The hand derivation for the default design, with the section sizes read from the
    generated bridge:

        span_cc = 1150; clear span = 1150 - d_pier (pier depth lies along X)
        deck length = 1150 + 2 * 50 = 1250 = total length (overhang 50 > pier half-depth)
        deck height = 200; y_b = 200 - 2 - d_floor_beam / 2; y_t = y_b + 260
        total height = y_t + d_top_chord / 2 (= 445 for 18-stick flat floor beams and
        4-stick on-edge top chords); total width = s + max(b_truss, b_pier)
        = 170 + b_truss + max(b_truss, b_pier)
        clear-box free length = clear span (nothing lower than 150 mm between the piers)
    """
    e = expected_measurements(default_params)
    stick = Stick()
    secs = default_bridge.section_map()
    b_pier, d_pier = secs["pier"].dims(stick)
    b_pb = secs["pier_brace"].dims(stick)[0]
    b_tc, d_tc = secs["top_chord"].dims(stick)
    d_fb = secs["floor_beam"].dims(stick)[1]
    b_truss = max(secs[g].dims(stick)[0] for g in ("top_chord", "bottom_chord", "diagonal"))

    assert e["span_cc_mm"] == 1150
    assert e["clear_span_mm"] == 1150 - max(d_pier, b_pb)
    assert e["deck_length_mm"] == 1250
    assert e["total_length_mm"] == 1250
    assert e["deck_height_mm"] == 200
    assert e["total_height_mm"] == 200 - 2 - d_fb / 2 + 260 + d_tc / 2
    assert e["total_width_mm"] == 170 + b_truss + max(b_truss, b_pier)
    assert e["clear_box_free_length_mm"] == e["clear_span_mm"]
    assert e["_lowest_span_member_mm"] >= DEFAULT_CONSTANTS["clear_box_height_mm"]


PARAM_SETS = {
    "default": WarrenParams(),
    "custom": WarrenParams(span_mm=1000.0, n_panels=8, truss_height_mm=300.0,
                           deck_top_elevation_mm=250.0, deck_clear_width_mm=160.0,
                           deck_overhang_mm=100.0),
    "no overhang": WarrenParams(deck_overhang_mm=0.0),
    "ten panels": WarrenParams(n_panels=10, truss_height_mm=300.0),
}


@pytest.mark.parametrize("case", list(PARAM_SETS))
def test_measurements_match_generator_geometry(material: Material, case: str) -> None:
    """measure() agrees with the hand geometry (expected_measurements) to 1e-6 mm."""
    params = PARAM_SETS[case]
    exp = expected_measurements(params)
    assert exp["_lowest_span_member_mm"] >= DEFAULT_CONSTANTS["clear_box_height_mm"]
    got = measure(generate_warren(params), material)
    for key, value in exp.items():
        if key.startswith("_"):
            continue
        assert got[key] == pytest.approx(value, abs=1e-6), key


def test_no_overhang_total_length_set_by_members(material: Material) -> None:
    """Without overhang the deck is exactly the span long, but the pier / pier-brace /
    floor-beam faces stick out beyond the pier centre lines at each end:

        total length = span + 2 * max(d_pier / 2, b_pier_brace / 2, b_floor_beam / 2)
    """
    params = PARAM_SETS["no overhang"]
    exp = expected_measurements(params)
    m = measure(generate_warren(params), material)
    assert m["deck_length_mm"] == pytest.approx(params.span_mm)
    assert m["total_length_mm"] == pytest.approx(params.span_mm + 2 * exp["_end_face_mm"])
    assert m["total_length_mm"] > m["deck_length_mm"]


def test_default_details(default_params: WarrenParams, default_bridge: Bridge,
                         material: Material) -> None:
    """Inner faces at x = f and span - f (f = pier half-depth along X); the table-supported
    members are the four piers and four pier braces; mid-span at x = span / 2; nothing
    obstructs the clear-span box or the clear opening; the checks pass.
    """
    exp = expected_measurements(default_params)
    span, f = default_params.span_mm, exp["_pier_face_mm"]
    n = default_params.n_panels
    m = measure(default_bridge, material)
    assert m.details["clear_span_faces_mm"] == pytest.approx((f, span - f))
    assert sorted(m.details["supporting_members"]) == sorted(
        [f"pier0{s}" for s in "nf"] + [f"pier{n}{s}" for s in "nf"]
        + ["pb0a", "pb0b", f"pb{n}a", f"pb{n}b"])
    assert m.details["mid_span_x_mm"] == pytest.approx(span / 2)
    assert m.details["clear_box_obstructions"] == []
    assert m.details["clear_opening_crossing"] == []
    assert m["cart_envelope_ok"] == 1.0
    assert m["clear_opening_ok"] == 1.0
    assert m["piers_vertical_ok"] == 1.0


def test_example_measures_like_generated(example_bridge: Bridge, default_bridge: Bridge,
                                         material: Material) -> None:
    """The committed example has the default geometry (regenerate it after changing the
    generator defaults).
    """
    assert measure(example_bridge, material).values == measure(default_bridge, material).values


def test_low_deck_blocks_clear_span_box(material: Material) -> None:
    """Deck top at 160 mm puts the bottom chord at y_b = 160 - 2 - d_floor_beam / 2, below
    the 150 mm-high clear-span box along the whole span, so the free length is 0.
    """
    params = WarrenParams(deck_top_elevation_mm=160.0)
    y_b = 160.0 - params.deck_thickness_mm - section_bd(params, "floor_beam")[1] / 2
    assert y_b < DEFAULT_CONSTANTS["clear_box_height_mm"]
    m = measure(generate_warren(params), material)
    assert m["clear_box_free_length_mm"] == 0.0
    assert "bc0n" in m.details["clear_box_obstructions"]


def test_clear_opening_constant_override(default_params: WarrenParams, default_bridge: Bridge,
                                         material: Material) -> None:
    """Changing the opening size in the constants moves the margin one-for-one:
    a 120 mm opening (20 mm larger) gives margin_100 - 10 mm.
    """
    base = expected_measurements(default_params)["clear_opening_margin_mm"]
    m = measure(default_bridge, material, {"clear_opening_mm": OPENING + 20.0})
    assert m["clear_opening_margin_mm"] == pytest.approx(base - 10.0)
