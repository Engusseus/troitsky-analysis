"""Untrusted text from shared YAML files: control characters, HTML/Markdown, CSV formulas,
and model-size limits."""

import pytest
from pydantic import ValidationError

from bridgesim.analysis import analyze
from bridgesim.generators import generate_warren
from bridgesim.rules import RuleSet
from bridgesim.schema import MAX_NODES, Bridge, Material, Node
from bridgesim.stability import global_buckling
from bridgesim.textsafe import csv_cell, csv_row, has_control_chars, md, printable

OSC52 = "evil\x1b]52;c;ZWNobyBoaQ==\x07"  # terminal clipboard write
BIDI = "abc‮dcba"  # right-to-left override


def test_printable_strips_terminal_controls() -> None:
    assert printable(OSC52) == "evil]52;c;ZWNobyBoaQ=="
    assert printable(BIDI) == "abcdcba"
    assert printable("line1\nline2\ttab") == "line1\nline2\ttab"


def test_has_control_chars() -> None:
    assert has_control_chars(OSC52)
    assert has_control_chars(BIDI)
    assert not has_control_chars("Warren truss 2027 (v2)")
    assert has_control_chars("a\nb") and not has_control_chars("a\nb", allow="\n")


def test_md_escapes_html_links_and_directives() -> None:
    out = md('<img src=x onerror=alert(1)> [click](http://evil) :red[x]')
    assert "<" not in out and ">" not in out
    assert "\\[click\\]\\(http\\" in out
    assert "\\:red\\[x\\]" in out


@pytest.mark.parametrize("cell", ["=HYPERLINK(\"http://x\")", "+1+1", "-2+3", "@SUM(A1)",
                                  "\t=1"])
def test_csv_cell_neutralises_formulas(cell: str) -> None:
    assert csv_cell(cell) == "'" + cell


def test_csv_cell_keeps_numbers_and_plain_text() -> None:
    assert csv_cell(-5) == -5
    assert csv_cell(3.2) == 3.2
    assert csv_cell("floor_beam") == "floor_beam"
    assert csv_row({"a": "=1", "b": 2}) == {"a": "'=1", "b": 2}


def test_schema_rejects_control_chars_in_names_and_ids() -> None:
    with pytest.raises(ValidationError):
        Node(id=OSC52, x_mm=0, y_mm=0, z_mm=0)
    b = generate_warren()
    data = b.model_dump(mode="json")
    data["name"] = OSC52
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)
    data = b.model_dump(mode="json")
    data["members"][0]["id"] = BIDI
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_material_and_rules_reject_control_chars() -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump()
    data["name"] = OSC52
    with pytest.raises(ValidationError):
        Material.model_validate(data)
    rs = RuleSet.load().model_dump()
    rs["rules"][0]["title"] = OSC52
    with pytest.raises(ValidationError):
        RuleSet.model_validate(rs)


def test_bridge_size_is_bounded() -> None:
    """A file with more than MAX_NODES nodes is rejected before any analysis runs."""
    data = generate_warren().model_dump(mode="json")
    data["nodes"] += [{"id": f"X{i}", "x_mm": i, "y_mm": 1.0, "z_mm": 0.0}
                      for i in range(MAX_NODES)]
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_sparse_and_dense_buckling_solvers_agree() -> None:
    """Large models use ARPACK (sparse); it must match the dense solver."""
    r = analyze(generate_warren())
    dense = global_buckling(r.fe_model, r.bridge)
    sparse = global_buckling(r.fe_model, r.bridge, dense_max_dof=0)
    assert sparse.lambdas == pytest.approx(dense.lambdas, rel=1e-8)
    # Symmetric members tie for the largest share of the mode energy, so compare energies
    # rather than which of the mirror-image members wins the tie.
    assert dense.member_energy[sparse.key_member] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- physical inputs


@pytest.mark.parametrize("key, value", [
    ("density_kg_m3", -650.0), ("E_MPa", 0.0), ("f_c_MPa", float("nan")),
    ("E_MPa", float("inf")), ("nu", 0.5),
])
def test_material_rejects_nonphysical_values(key: str, value: float) -> None:
    """A negative density would give a negative mass that passes the mass rule."""
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump()
    data[key]["value"] = value
    with pytest.raises(ValidationError, match="not physical|finite number"):
        Material.model_validate(data)


@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_glue_fraction_must_be_in_unit_interval(value: float) -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump()
    data["glue"]["mass_fraction"]["value"] = value
    with pytest.raises(ValidationError, match="not physical"):
        Material.model_validate(data)


def test_rules_crusher_plate_overrides_bridge_plate() -> None:
    """The rules file describes the competition's crusher, so its plate size is analysed."""
    from bridgesim.loads import plate_nodal_loads
    from bridgesim.rules import apply_crushing

    bridge = generate_warren()
    rs = RuleSet.load()
    assert apply_crushing(bridge, rs).load.plate_length_mm == 200.0
    rs.crushing.plate_length_mm = 400.0
    rs.crushing.plate_width_mm = 120.0
    wide = apply_crushing(bridge, rs)
    assert (wide.load.plate_length_mm, wide.load.plate_width_mm) == (400.0, 120.0)
    assert bridge.load.plate_length_mm == 200.0  # original untouched
    narrow_loads, wide_loads = plate_nodal_loads(bridge), plate_nodal_loads(wide)
    assert sum(wide_loads.values()) == pytest.approx(sum(narrow_loads.values()))
    assert len(wide_loads) > len(narrow_loads)  # a longer plate reaches more floor beams


@pytest.mark.parametrize("field, value", [
    ("plate_length_mm", -200.0), ("plate_width_mm", 0.0),
    ("deflection_limit_mm", float("nan")), ("deflection_limit_mm", float("inf")),
])
def test_rules_reject_invalid_crusher(field: str, value: float) -> None:
    """A negative deflection limit would give a negative predicted load."""
    data = RuleSet.load().model_dump()
    data["crushing"][field] = value
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


def test_rules_reject_invalid_constants() -> None:
    data = RuleSet.load().model_dump()
    data["constants"]["cart_height_mm"] = -1.0
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


def test_report_states_the_material_actually_used() -> None:
    from bridgesim.materials import load_material
    from bridgesim.report import model_assumptions, to_markdown

    data = load_material("popsicle_birch").model_dump()
    data["E_MPa"] = {"value": 12345.0, "source": "measured", "note": ""}
    mat = Material.model_validate(data)
    result = analyze(generate_warren(), mat)
    lines = model_assumptions(result)
    assert "E = 12345" in lines[0] and "11 of 12 values are ASSUMED" in lines[0]
    assert "E = 10 GPa" not in "\n".join(lines)
    assert "E = 12345" in to_markdown(result)


def test_report_states_the_crusher_actually_used() -> None:
    """Assumptions quote the analysed plate size and deflection limit, not 2027 defaults."""
    from bridgesim.materials import load_material
    from bridgesim.report import model_assumptions
    from bridgesim.rules import apply_crushing

    rs = RuleSet.load()
    rs.crushing.plate_length_mm, rs.crushing.plate_width_mm = 150.0, 80.0
    bridge = apply_crushing(generate_warren(), rs)
    result = analyze(bridge, load_material("popsicle_birch"), deflection_limit_mm=40.0)
    text = "\n".join(model_assumptions(result))
    assert "150 mm x 80 mm" in text and "40 mm mid-span deflection" in text
    assert "200 mm x 90 mm" not in text and "at 50 mm" not in text


def test_ban_labels_reject_control_characters() -> None:
    data = RuleSet.load().model_dump()
    span = next(r for r in data["rules"] if r["key"] == "span_length")
    span["bands"][-1]["bans"] = [OSC52]
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


def test_unsupported_schema_version_is_rejected() -> None:
    data = generate_warren().model_dump(mode="json")
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_node_cannot_be_both_pinned_and_roller() -> None:
    data = generate_warren().model_dump(mode="json")
    data["supports"]["roller"].append(data["supports"]["pinned"][0])
    with pytest.raises(ValidationError, match="both pinned and roller"):
        Bridge.model_validate(data)


def test_failed_buckling_solve_is_reported_not_infinite(monkeypatch) -> None:
    """If the eigen-solver fails, F_cr is NaN ('not evaluated') with a warning, never inf."""
    import math

    import bridgesim.analysis as analysis_mod
    from bridgesim.stability import BucklingResult

    monkeypatch.setattr(analysis_mod, "global_buckling",
                        lambda model, bridge: BucklingResult(math.nan, note="solver failed"))
    r = analysis_mod.analyze(generate_warren())
    assert math.isnan(r.Fu_buckling_N)
    assert math.isfinite(r.Fu_pred_N) and r.Fu_pred_N > 0
    assert any("solver failed" in w for w in r.warnings)
    from bridgesim.report import to_markdown

    assert "NOT EVALUATED" in to_markdown(r)


def test_buckling_view_hover_shows_utilisation_not_energy() -> None:
    from bridgesim import viz

    r = analyze(generate_warren())
    fig = viz.bridge_figure(r.bridge, r, "buckling")
    hover = next(t for t in fig.data if getattr(t, "hovertext", None) is not None
                 and any("Governing check" in h for h in t.hovertext))
    m = r.critical_members(1)[0]
    text = next(h for h in hover.hovertext if f"<b>{m.id}</b>" in h)
    assert f"at F_u,p = {m.U * r.load_factor:.2f}" in text
    assert "Share of buckling-mode energy" in text


# --------------------------------------------------------------------------- fifth review round


def test_report_lists_custom_effective_length_factors() -> None:
    from bridgesim.report import model_assumptions

    b = generate_warren()
    members = [m.model_copy(update={"K": 0.7}) if m.group == "top_chord" else m
               for m in b.members]
    r = analyze(b.model_copy(update={"members": members}))
    text = "\n".join(model_assumptions(r))
    assert "K = 0.7" in text and "K = 1 for every member" not in text
    assert "K = 1 for every member" in "\n".join(model_assumptions(analyze(generate_warren())))


def test_report_formula_uses_the_analysed_deflection_limit() -> None:
    from bridgesim.report import to_markdown

    md_text = to_markdown(analyze(generate_warren(), deflection_limit_mm=40.0))
    assert r"\frac{40\ \text{mm}}" in md_text and r"F\_u,δ \(40 mm\)" in md_text
    assert r"\frac{50" not in md_text


def test_colocated_deck_stations_share_the_load() -> None:
    """Two deck-support nodes at each X station (one per truss plane) split its load."""
    from bridgesim.loads import plate_nodal_loads

    b = generate_warren()
    n = b.metadata["params"]["n_panels"]
    nodes_n = [f"B{i}n" for i in range(n + 1)]
    nodes_f = [f"B{i}f" for i in range(n + 1)]
    load = b.load.model_copy(update={"deck_support_nodes": nodes_n + nodes_f})
    loads = plate_nodal_loads(b.model_copy(update={"load": load}))
    assert sum(loads.values()) == pytest.approx(b.load.P_ref_N)
    mid = n // 2
    assert loads[f"B{mid}n"] == pytest.approx(loads[f"B{mid}f"])
    assert loads[f"B{mid}n"] == pytest.approx(652.1739 / 2, rel=1e-4)


def test_cart_path_is_bounded_by_the_deck_edges(material=None) -> None:
    """With nothing beside the path, a deck narrower than the cart still fails."""
    from bridgesim.materials import load_material
    from bridgesim.measure import measure

    b = generate_warren()
    narrow = b.model_copy(update={"deck": b.deck.model_copy(update={"clear_width_mm": 120.0}),
                                  "members": [m for m in b.members
                                              if m.group not in ("diagonal", "top_chord",
                                                                 "top_strut", "top_brace")]})
    m = measure(narrow, load_material("popsicle_birch"))
    assert m["cart_clear_width_mm"] == pytest.approx(120.0)
    assert m["cart_envelope_ok"] == 0.0


def test_extra_restraints_appear_in_reactions() -> None:
    """A support given only through extra_restraints must still report its reaction."""
    from bridgesim.schema import Supports

    b = generate_warren()
    n = b.metadata["params"]["n_panels"]
    sup = Supports(pinned=["P0n", "P0f"], roller=[f"P{n}n"],
                   extra_restraints={f"P{n}f": ["DY", "DZ"]})
    r = analyze(b.model_copy(update={"supports": sup}))
    nodes = {x.node for x in r.reactions}
    assert f"P{n}f" in nodes
    assert sum(x.FY_N for x in r.reactions) == pytest.approx(b.load.P_ref_N, rel=1e-6)


@pytest.mark.parametrize("field, value", [
    ("step", 0.0), ("step", -0.5), ("step", float("nan")), ("cap", 5.0),
    ("points_per_step", -2.0),
])
def test_mass_step_rules_are_validated(field: str, value: float) -> None:
    data = RuleSet.load().model_dump()
    mass = next(r for r in data["rules"] if r["key"] == "mass")
    mass["steps"][field] = value
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


def test_no_load_path_is_an_error_not_infinite_capacity() -> None:
    from bridgesim.analysis import AnalysisError
    from bridgesim.schema import Supports

    b = generate_warren()
    deck_nodes = b.load.deck_support_nodes
    sup = b.supports.model_copy(update={
        "extra_restraints": {nid: ["DX", "DY", "DZ"] for nid in deck_nodes}})
    with pytest.raises(AnalysisError, match="no structural load path"):
        analyze(b.model_copy(update={"supports": Supports.model_validate(sup.model_dump())}))


def test_dollar_signs_in_names_do_not_break_png_export() -> None:
    from bridgesim import viz

    b = generate_warren()
    b = b.model_copy(update={"name": r"Team $\notacommand$ bridge"})
    r = analyze(b)
    assert viz.figure_png(viz.global_sfd_bmd_figure(r))[:4] == b"\x89PNG"
    assert viz.figure_png(viz.member_diagrams_figure(r, ["bc0n"], r"$bad$"))[:4] == b"\x89PNG"


# --------------------------------------------------------------------------- sixth review round


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_member_and_geometry_values_are_rejected(value: float) -> None:
    data = generate_warren().model_dump(mode="json")
    data["members"][0]["K"] = value
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)
    data = generate_warren().model_dump(mode="json")
    data["deck"]["thickness_mm"] = value
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_absurd_coordinates_are_rejected() -> None:
    """A 1e9 mm member would make geometric sampling allocate ~500 million points."""
    data = generate_warren().model_dump(mode="json")
    data["nodes"][0]["x_mm"] = 1e9
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_long_members_keep_the_2_mm_sample_spacing() -> None:
    """A 90 m member is still sampled every 2 mm (coarser sampling could hide a clash)."""
    import numpy as np

    from bridgesim.materials import load_material
    from bridgesim.measure import SAMPLE_MM, _samples

    data = generate_warren().model_dump(mode="json")
    data["nodes"].append({"id": "far", "x_mm": 90_000.0, "y_mm": 300.0, "z_mm": 0.0})
    data["members"].append({"id": "long", "i": "C0", "j": "far", "section": "pier"})
    b = Bridge.model_validate(data)
    s = next(x for x in _samples(b, load_material("popsicle_birch"), 0.0) if x.member == "long")
    assert np.linalg.norm(np.diff(s.pts, axis=0), axis=1).max() <= SAMPLE_MM + 1e-9


@pytest.mark.parametrize("penalty", [-5.0, float("nan"), float("inf")])
def test_band_penalties_must_be_finite_and_nonnegative(penalty: float) -> None:
    data = RuleSet.load().model_dump()
    span = next(r for r in data["rules"] if r["key"] == "span_length")
    span["bands"][1]["penalty"] = penalty
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


def test_vertical_extra_restraints_count_as_supports_for_geometry() -> None:
    """Supports given only via extra_restraints (DY) define mid-span and the table level."""
    from bridgesim.loads import mid_span_x
    from bridgesim.materials import load_material
    from bridgesim.measure import measure
    from bridgesim.schema import Supports

    b = generate_warren()
    n = b.metadata["params"]["n_panels"]
    sup = Supports(pinned=["P0n"], roller=[],
                   extra_restraints={"P0f": ["DX", "DY", "DZ"], f"P{n}n": ["DY", "DZ"],
                                     f"P{n}f": ["DY", "DZ"]})
    b2 = b.model_copy(update={"supports": sup})
    assert mid_span_x(b2) == pytest.approx(mid_span_x(b))
    assert measure(b2, load_material("popsicle_birch"))["span_cc_mm"] == pytest.approx(1150.0)


# --------------------------------------------------------------------------- seventh review round


def test_markdown_report_neutralises_links_and_images() -> None:
    from bridgesim.report import to_markdown

    b = generate_warren().model_copy(
        update={"name": "Pixel ![x](https://example.invalid/p.png) [click](http://evil)"})
    text = to_markdown(analyze(b))
    assert "![x](" not in text and "[click](" not in text
    assert r"\!\[x\]\(" in text
    assert "**F_u,p" in text or "**" in text  # intentional bold survives


def test_blank_identifiers_are_rejected() -> None:
    data = generate_warren().model_dump(mode="json")
    data["nodes"][0]["id"] = "   "
    with pytest.raises(ValidationError, match="blank"):
        Bridge.model_validate(data)


def test_deck_support_nodes_must_not_be_empty() -> None:
    data = generate_warren().model_dump(mode="json")
    data["load"]["deck_support_nodes"] = []
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


@pytest.mark.parametrize("rule_type, drop", [("bands", "measures"), ("bands", "bands"),
                                             ("check", "measure"), ("steps", "steps")])
def test_rules_need_the_fields_of_their_type(rule_type: str, drop: str) -> None:
    data = RuleSet.load().model_dump()
    rule = next(r for r in data["rules"] if r["type"] == rule_type)
    rule[drop] = [] if drop in ("measures", "bands") else None
    with pytest.raises(ValidationError, match="needs"):
        RuleSet.model_validate(data)


def test_node_inside_a_member_is_rejected_as_a_hidden_joint() -> None:
    """Pynite would silently split the member there; the file must say so explicitly."""
    b = generate_warren()
    data = b.model_dump(mode="json")
    t0, t1 = b.node_map()["T0n"], b.node_map()["T1n"]
    data["nodes"].append({"id": "mid", "x_mm": (t0.x_mm + t1.x_mm) / 2, "y_mm": t0.y_mm,
                          "z_mm": t0.z_mm})
    data["members"].append({"id": "post", "i": "mid", "j": "C0", "section": "pier"})
    with pytest.raises(ValidationError, match="lies on member 'tc0n'"):
        Bridge.model_validate(data)


def test_extra_restraint_roller_gets_the_same_buckling_load() -> None:
    """A roller written as extra_restraints [DY, DZ] is held by friction during buckling,
    exactly like an entry in supports.roller."""
    from bridgesim.schema import Supports

    b = generate_warren()
    n = b.metadata["params"]["n_panels"]
    alt = Supports(pinned=b.supports.pinned, roller=[],
                   extra_restraints={f"P{n}n": ["DY", "DZ"], f"P{n}f": ["DY", "DZ"]})
    r1 = analyze(b)
    r2 = analyze(b.model_copy(update={"supports": alt}))
    assert r2.Fu_buckling_N == pytest.approx(r1.Fu_buckling_N, rel=1e-9)


def test_only_connected_chains_are_offered_as_group_diagrams() -> None:
    from bridgesim import viz

    b = generate_warren()
    assert viz.is_connected_chain(b, viz.chain_for_group(b, "bottom_chord", "n"))
    assert viz.is_connected_chain(b, viz.chain_for_group(b, "top_chord", "f"))
    assert not viz.is_connected_chain(b, viz.chain_for_group(b, "floor_beam", "n"))
    assert not viz.is_connected_chain(b, viz.chain_for_group(b, "pier", "n"))
    assert not viz.is_connected_chain(b, [])


# --------------------------------------------------------------------------- eighth review round


def test_total_member_length_is_bounded() -> None:
    """Thousands of long members are rejected up front instead of being sampled coarsely."""
    from bridgesim.measure import MAX_TOTAL_SAMPLES, _sample_counts

    data = generate_warren().model_dump(mode="json")
    for k in range(60):
        data["nodes"].append({"id": f"far{k}", "x_mm": 90_000.0, "y_mm": 300.0 + k,
                              "z_mm": 0.0})
        data["members"].append({"id": f"long{k}", "i": "C0", "j": f"far{k}",
                                "section": "pier"})
    with pytest.raises(ValidationError, match="the limit is 500 m"):
        Bridge.model_validate(data)
    with pytest.raises(ValueError):
        _sample_counts([10_000.0] * 6000)
    assert sum(_sample_counts([500_000.0 / 6000] * 6000)) <= MAX_TOTAL_SAMPLES


def test_deflection_governing_label_uses_the_active_limit() -> None:
    import dataclasses

    r = analyze(generate_warren(), deflection_limit_mm=2.0)  # 4.5 mm at the strength limit
    assert r.governing_mode == "deflection"
    assert r.governing_label == "deflection > 2 mm"
    assert dataclasses.replace(r, deflection_limit_mm=50.0).governing_label == \
        "deflection > 50 mm"


def test_glue_exclusions_must_be_member_groups() -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump(mode="json")
    data["glue"]["exclude_groups"] = ["top_chrod"]
    with pytest.raises(ValidationError, match="top_chrod"):
        Material.model_validate(data)


@pytest.mark.parametrize("restraints", [{"P0n": []}, {"P0n": ["RX", "RZ"]}])
def test_empty_or_rotation_only_restraints_are_not_supports(restraints: dict) -> None:
    data = generate_warren().model_dump(mode="json")
    data["supports"]["pinned"] = []
    data["supports"]["extra_restraints"] = restraints
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


@pytest.mark.parametrize("field, key", [("constants", "cart_heigth_mm"),
                                        ("rounding", "lenght_mm")])
def test_unknown_rule_constants_are_rejected(field: str, key: str) -> None:
    data = RuleSet.load().model_dump()
    data[field][key] = 210.0
    with pytest.raises(ValidationError, match=key):
        RuleSet.model_validate(data)


def test_supports_given_only_as_extra_restraints_are_drawn() -> None:
    from bridgesim import viz
    from bridgesim.schema import Supports

    b = generate_warren()
    n = b.metadata["params"]["n_panels"]
    sup = Supports(pinned=[], roller=[],
                   extra_restraints={"P0n": ["DX", "DY", "DZ"], "P0f": ["DX", "DY", "DZ"],
                                     f"P{n}n": ["DY", "DZ"], f"P{n}f": ["DY", "DZ"]})
    fig = viz.bridge_figure(b.model_copy(update={"supports": sup}), view="undeformed")
    markers = next(t for t in fig.data if t.name == "supports")
    assert len(markers.x) == 4
    assert "P0n: restrained (DX, DY, DZ)" in markers.hovertext


def test_failed_buckling_solve_is_not_shown_as_no_compression(monkeypatch) -> None:
    import math

    from bridgesim import viz
    from bridgesim.stability import BucklingResult

    r = analyze(generate_warren())
    r.buckling = BucklingResult(math.nan, note="solver failed")
    title = viz.bridge_figure(r.bridge, r, view="buckling").layout.title.text
    assert "NOT EVALUATED" in title and "nothing in compression" not in title


def test_chain_diagrams_do_not_depend_on_member_end_order() -> None:
    """A chord written as A->B, C->B must plot exactly like A->B, B->C."""
    import numpy as np

    from bridgesim import viz

    b = generate_warren()
    for group in ("bottom_chord", "top_chord", "diagonal"):
        ids = viz.chain_for_group(b, group, "n")
        flip = set(ids[1::2])
        mems = [m.model_copy(update={"i": m.j, "j": m.i}) if m.id in flip else m
                for m in b.members]
        ref = viz.member_chain_diagrams(analyze(b, include_buckling=False), ids)
        alt = viz.member_chain_diagrams(
            analyze(b.model_copy(update={"members": mems}), include_buckling=False), ids)
        for a, c in zip(ref[:4], alt[:4], strict=True):
            np.testing.assert_allclose(c, a, atol=1e-6)


def _render_rules_card(app_path: str, report_json: str) -> None:
    """Streamlit script: run the app's _rules_card on a given report."""
    from pathlib import Path

    from bridgesim.rules import RulesReport

    src = Path(app_path).read_text(encoding="utf-8").rsplit("\nmain()", 1)[0]
    ns = {"__name__": "bridgesim_app", "__file__": app_path}
    exec(compile(src, app_path, "exec"), ns)
    ns["_rules_card"](RulesReport.model_validate_json(report_json))


def test_rule_banner_does_not_claim_success_over_a_failed_check() -> None:
    """A failing check with no penalty, ban or DQ is still not 'All checked rules pass'."""
    from pathlib import Path

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    rep = evaluate(generate_warren(), load_material("popsicle_birch"))
    k = next(i for i, r in enumerate(rep.results) if r.key == "clearance_above_deck")
    rep.results[k] = rep.results[k].model_copy(
        update={"passed": False, "penalty": 0, "bans": [], "disqualification": False})
    app_path = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
    at = AppTest.from_function(_render_rules_card, args=(app_path, rep.model_dump_json()),
                               default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("All checked rules pass" in s.value for s in at.success)
    assert any("fail" in w.value for w in at.warning)


# --------------------------------------------------------------------------- ninth review round


def _hanging_post() -> Bridge:
    """Default bridge plus a post hanging 30 mm below the table (y = 0)."""
    b = generate_warren()
    data = b.model_dump(mode="json")
    b3 = b.node_map()["B3n"]
    data["nodes"].append({"id": "low", "x_mm": b3.x_mm, "y_mm": -30.0, "z_mm": b3.z_mm})
    data["members"].append({"id": "hang", "i": "B3n", "j": "low", "section": "pier"})
    return Bridge.model_validate(data)


def test_geometry_below_the_table_fails_a_rule_instead_of_being_clipped() -> None:
    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    rep = evaluate(_hanging_post(), load_material("popsicle_birch"))
    r = next(r for r in rep.results if r.key == "above_platform")
    assert r.passed is False and "30" in r.measured_text
    assert not rep.all_passed


def _raised_supports() -> Bridge:
    """Default bridge lifted 300 mm, deck top left at 200 mm: the deck is below the table."""
    data = generate_warren().model_dump(mode="json")
    for n in data["nodes"]:
        n["y_mm"] += 300.0
    return Bridge.model_validate(data)


def test_deck_below_raised_supports_fails_the_platform_rule() -> None:
    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    rep = evaluate(_raised_supports(), load_material("popsicle_birch"))
    assert next(r for r in rep.results if r.key == "above_platform").passed is False


def test_cli_check_fails_on_a_failed_rule_without_penalty(tmp_path) -> None:
    from typer.testing import CliRunner

    from bridgesim.cli import app
    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    b = generate_warren()  # plus a 5 mm stub poking through the table under a pier base
    data = b.model_dump(mode="json")
    p0 = b.node_map()["P0n"]
    data["nodes"].append({"id": "foot", "x_mm": p0.x_mm, "y_mm": -5.0, "z_mm": p0.z_mm})
    data["members"].append({"id": "stub", "i": "P0n", "j": "foot", "section": "pier"})
    stub = Bridge.model_validate(data)
    rep = evaluate(stub, load_material("popsicle_birch"))
    assert [r.key for r in rep.checked if not r.passed] == ["above_platform"]
    assert rep.total_penalty == 0 and not rep.bans and not rep.disqualification_risks
    path = tmp_path / "stub.yaml"
    stub.save(path)
    res = CliRunner().invoke(app, ["check", str(path)])
    assert res.exit_code == 1, res.output


def test_non_mapping_generator_metadata_does_not_break_reports() -> None:
    from bridgesim.report import to_html, to_markdown

    b = generate_warren().model_copy(update={"metadata": {"params": 1, "generator": [1]}})
    r = analyze(b)
    assert "Generator" in to_markdown(r) and "Generator" in to_html(r)


@pytest.mark.parametrize("path, value", [
    (("deck", "clear_width_mm"), 1e308),
    (("deck", "x_end_mm"), 1e308),
    (("load", "P_ref_N"), 1e308),
    (("load", "plate_length_mm"), 1e308),
    (("sections", 0, "b_mm"), 1e308),
])
def test_finite_but_absurd_bridge_values_are_rejected(path: tuple, value: float) -> None:
    data = generate_warren().model_dump(mode="json")
    if path[0] == "sections":
        data["sections"][0] = {"id": data["sections"][0]["id"], "b_mm": value, "d_mm": 10.0}
    else:
        data[path[0]][path[1]] = value
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


@pytest.mark.parametrize("key", ["density_kg_m3", "E_MPa", "f_t_MPa"])
def test_finite_but_absurd_material_values_are_rejected(key: str) -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump(mode="json")
    data[key]["value"] = 1e308
    with pytest.raises(ValidationError, match="not physical"):
        Material.model_validate(data)


@pytest.mark.parametrize("field", ["cap", "free_up_to", "step"])
def test_absurd_mass_steps_are_rejected(field: str) -> None:
    data = RuleSet.load().model_dump()
    mass = next(r for r in data["rules"] if r["type"] == "steps")
    mass["steps"][field] = 1e308
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


# --------------------------------------------------------------------------- hands-on testing


def test_friendly_error_names_the_field_and_value() -> None:
    from bridgesim.textsafe import friendly_error

    data = generate_warren().model_dump(mode="json")
    data["nodes"][2]["x_mm"] = "abc"
    with pytest.raises(ValidationError) as info:
        Bridge.model_validate(data)
    msg = friendly_error(info.value)
    assert msg.startswith("nodes #3 › x_mm: Input should be a valid number")
    assert "(got 'abc')" in msg and "pydantic.dev" not in msg


@pytest.mark.parametrize("args", [["run", "{bad}"], ["check", "{bad}"],
                                  ["run", "{ok}", "--rules", "no_such_rules"],
                                  ["run", "{ok}", "--material", "no_such_material"]])
def test_cli_reports_unreadable_inputs_without_a_traceback(tmp_path, args: list) -> None:
    from pathlib import Path

    from typer.testing import CliRunner

    from bridgesim.cli import app

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: broken\nnodes: [1, 2\n", encoding="utf-8")
    ok = Path(__file__).resolve().parents[1] / "examples" / "warren_2027.yaml"
    res = CliRunner().invoke(app, [a.format(bad=bad, ok=ok) for a in args])
    assert res.exit_code == 2, res.output
    assert "ERROR: could not read" in res.output and "Traceback" not in res.output


def test_member_diagrams_plot_sagging_positive_with_v_equal_dm_dx() -> None:
    import numpy as np

    from bridgesim import viz

    r = analyze(generate_warren(), include_buckling=False)
    x, _, V, M, _ = viz.member_chain_diagrams(r, ["fb4n"], r.P_ref_N)
    assert M.max() > 0 and M[-1] == pytest.approx(M.max())  # sagging at the deck centre
    np.testing.assert_allclose(V[1:-1], np.gradient(M, x)[1:-1], rtol=1e-6)


# --------------------------------------------------------------------------- tenth review round


def test_member_surface_through_the_platform_fails_the_rule() -> None:
    """A member not resting on the table whose section dips below it is reported."""
    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    b = generate_warren()
    data = b.model_dump(mode="json")
    p0, pn = b.node_map()["P0n"], b.node_map()[f"P{b.metadata['params']['n_panels']}n"]
    data["nodes"] += [{"id": "lo1", "x_mm": p0.x_mm + 100, "y_mm": 1.0, "z_mm": p0.z_mm},
                      {"id": "lo2", "x_mm": pn.x_mm - 100, "y_mm": 1.0, "z_mm": pn.z_mm}]
    data["members"].append({"id": "low_tie", "i": "lo1", "j": "lo2", "section": "pier"})
    rep = evaluate(Bridge.model_validate(data), load_material("popsicle_birch"))
    r = next(r for r in rep.results if r.key == "above_platform")
    assert r.passed is False


def test_branching_group_is_not_a_chain() -> None:
    from bridgesim import viz

    b = generate_warren()
    nodes = b.node_map()
    data = b.model_dump(mode="json")
    t = nodes["T1n"]
    data["nodes"].append({"id": "spur", "x_mm": t.x_mm, "y_mm": t.y_mm + 50, "z_mm": t.z_mm})
    data["members"].append({"id": "branch", "i": "T1n", "j": "spur", "section": "top_chord",
                            "group": "top_chord"})
    b2 = Bridge.model_validate(data)
    chord = viz.chain_for_group(b, "top_chord", "n")
    assert viz.is_connected_chain(b2, chord)
    mm = {m.id: m for m in b.members}
    k = next(i for i, mid in enumerate(chord) if mm[mid].j == "T1n")
    # T0-T1, T1-spur, T1-T2: every neighbour pair shares a node, but it is a branch
    assert not viz.is_connected_chain(b2, chord[:k + 1] + ["branch"] + chord[k + 1:])


@pytest.mark.parametrize("step", [1e-30, 1e-7])
def test_tiny_rounding_steps_are_rejected(step: float) -> None:
    data = RuleSet.load().model_dump()
    data["rounding"]["length_mm"] = step
    with pytest.raises(ValidationError, match="greater than"):
        RuleSet.model_validate(data)


def test_stick_dimensions_that_underflow_are_rejected() -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump(mode="json")
    data["stick"] = {"length_mm": 1e-300, "width_mm": 1e-300, "thickness_mm": 1e-300}
    with pytest.raises(ValidationError):
        Material.model_validate(data)


# --------------------------------------------------------------------------- eleventh review round


def _mechanism() -> Bridge:
    """One member pinned at one end only: free to rotate, so the stiffness is singular."""
    from bridgesim.schema import Deck, Member, Node, PlateLoad, Section, Supports

    return Bridge(
        name="mechanism", nodes=[Node(id="A", x_mm=0, y_mm=0, z_mm=0),
                                 Node(id="B", x_mm=100, y_mm=0, z_mm=0)],
        members=[Member(id="M", i="A", j="B", section="s")],
        sections=[Section(id="s", b_mm=10, d_mm=10)],
        deck=Deck(x_start_mm=0, x_end_mm=100, top_elevation_mm=5, clear_width_mm=10),
        supports=Supports(pinned=["A"]),
        load=PlateLoad(P_ref_N=1, plate_length_mm=1, x_center_mm=100,
                       deck_support_nodes=["B"]))


def test_unstable_structure_is_an_analysis_error(tmp_path) -> None:
    from typer.testing import CliRunner

    from bridgesim.analysis import AnalysisError
    from bridgesim.cli import app

    with pytest.raises(AnalysisError, match="unstable"):
        analyze(_mechanism())
    path = tmp_path / "mech.yaml"
    _mechanism().save(path)
    res = CliRunner().invoke(app, ["run", str(path)])
    assert res.exit_code == 2 and "unstable" in res.output and "Traceback" not in res.output


def test_user_node_named_like_an_interior_node_is_fine() -> None:
    from bridgesim.schema import Deck, Member, Node, PlateLoad, Section, Supports

    col = Bridge(
        name="column", nodes=[Node(id="~mid M", x_mm=0, y_mm=0, z_mm=0),
                              Node(id="T", x_mm=0, y_mm=115, z_mm=0)],
        members=[Member(id="M", i="~mid M", j="T", section="s")],
        sections=[Section(id="s", b_mm=10, d_mm=2)],
        deck=Deck(x_start_mm=-1, x_end_mm=1, top_elevation_mm=115, clear_width_mm=10),
        supports=Supports(pinned=["~mid M"], extra_restraints={"~mid M": ["RY"],
                                                               "T": ["DX", "DZ"]}),
        load=PlateLoad(P_ref_N=1, plate_length_mm=1e-3, x_center_mm=0,
                       deck_support_nodes=["T"]))
    r = analyze(col)
    assert r.buckling.lambda_cr == pytest.approx(49.75, rel=0.01)


def test_deck_underside_below_the_platform_fails_the_rule() -> None:
    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    b = generate_warren()
    deck = b.deck.model_copy(update={"top_elevation_mm": 1.0, "thickness_mm": 2.0})
    rep = evaluate(b.model_copy(update={"deck": deck}), load_material("popsicle_birch"))
    assert next(r for r in rep.results if r.key == "above_platform").passed is False


@pytest.mark.parametrize("bands, msg", [
    ([{"max": 1350}], "above 1350"),
    ([{"min": 0, "max": 1350}, {"min": 1351}], "below 0"),
    ([{"max": 1350.2}, {"min": 1351.1, "penalty": 5}], "after 1350.2"),
])
def test_band_rules_must_leave_no_gaps(bands: list, msg: str) -> None:
    data = RuleSet.load().model_dump()
    rule = next(r for r in data["rules"] if r["key"] == "total_length")
    rule["bands"] = bands
    with pytest.raises(ValidationError, match=msg):
        RuleSet.model_validate(data)


@pytest.mark.parametrize("field, value", [("step", 0.006), ("step", 0.015),
                                          ("free_up_to", 6.005), ("cap", 15.001)])
def test_mass_steps_must_be_whole_centigrams(field: str, value: float) -> None:
    data = RuleSet.load().model_dump()
    mass = next(r for r in data["rules"] if r["type"] == "steps")
    mass["steps"][field] = value
    with pytest.raises(ValidationError, match="whole number of 0.01 kg"):
        RuleSet.model_validate(data)


def test_dense_buckling_is_chosen_from_the_full_matrix_size(monkeypatch) -> None:
    from bridgesim.materials import load_material
    from bridgesim.model import build_model
    from bridgesim.stability import _free_dofs, global_buckling

    b = generate_warren()
    m = build_model(b, load_material("popsicle_birch"))
    m.analyze_linear()
    n_free, n_all = len(_free_dofs(m, b)), 6 * len(m.nodes)
    seen = {}
    original = m.Ke

    def spy(*args, **kwargs):
        seen["sparse"] = kwargs.get("sparse")
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "Ke", spy)
    global_buckling(m, b, dense_max_dof=(n_free + n_all) // 2)  # free DOFs alone would fit
    assert seen["sparse"] is True


# --------------------------------------------------------------------------- twelfth review round


def test_report_states_the_buckling_mesh_used(monkeypatch) -> None:
    import bridgesim.analysis as analysis_mod
    import bridgesim.model as model_mod
    from bridgesim.report import model_assumptions

    r = analyze(generate_warren())
    assert any("two elements per member" in a for a in model_assumptions(r))
    monkeypatch.setattr(model_mod, "REFINE_MAX_MEMBERS", 10)
    monkeypatch.setattr(analysis_mod, "REFINE_MAX_MEMBERS", 10)
    r1 = analyze(generate_warren())
    assert any("one element per member" in a for a in model_assumptions(r1))
    assert any("one element for every member" in w for w in r1.warnings)
    assert not r1.fe_model.members["bc0n"].sub_members or len(
        r1.fe_model.members["bc0n"].sub_members) == 1


def test_html_report_drops_bidi_controls_from_shared_text() -> None:
    from bridgesim.report import to_html

    b = generate_warren().model_copy(update={"metadata": {"generator": "warren‮spoof"}})
    r = analyze(b)
    assert "‮" not in to_html(r)


def test_material_notes_reject_control_characters() -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump(mode="json")
    data["E_MPa"]["note"] = "spoof‮txet"
    with pytest.raises(ValidationError, match="control characters"):
        Material.model_validate(data)


def test_tiny_reference_load_is_rejected() -> None:
    data = generate_warren().model_dump(mode="json")
    data["load"]["P_ref_N"] = 5e-324
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_one_free_dof_buckling_does_not_crash(monkeypatch) -> None:
    """An inclined strut whose top can only move vertically: the reduced eigenproblem is
    1x1 (with axial force and a non-zero K_g), which ARPACK cannot solve (k < N)."""
    import math

    import bridgesim.model as model_mod
    from bridgesim.materials import load_material
    from bridgesim.model import build_model
    from bridgesim.schema import Deck, Member, Node, PlateLoad, Section, Supports
    from bridgesim.stability import _free_dofs, global_buckling

    monkeypatch.setattr(model_mod, "REFINE_MAX_MEMBERS", 0)  # no interior split node
    b = Bridge(
        name="strut", nodes=[Node(id="A", x_mm=0, y_mm=0, z_mm=0),
                             Node(id="B", x_mm=100, y_mm=100, z_mm=0)],
        members=[Member(id="M", i="A", j="B", section="s")],
        sections=[Section(id="s", b_mm=10, d_mm=10)],
        deck=Deck(x_start_mm=99, x_end_mm=101, top_elevation_mm=100, clear_width_mm=10),
        supports=Supports(pinned=["A"], extra_restraints={
            "A": ["RX", "RY", "RZ"], "B": ["DX", "DZ", "RX", "RY", "RZ"]}),
        load=PlateLoad(P_ref_N=1, plate_length_mm=1e-3, x_center_mm=100,
                       deck_support_nodes=["B"]))
    m = build_model(b, load_material("popsicle_birch"))
    m.analyze_linear()
    assert len(_free_dofs(m, b)) == 1
    res = global_buckling(m, b, dense_max_dof=1)  # full-size rule picks sparse assembly
    assert math.isinf(res.lambda_cr) or res.lambda_cr > 0


def test_glued_faces_must_be_whole() -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump(mode="json")
    data["glue"]["faces"]["value"] = 1.5
    with pytest.raises(ValidationError, match="whole number"):
        Material.model_validate(data)


# ------------------------------------------------------------------------ thirteenth review round


def test_members_that_cannot_be_split_are_reported() -> None:
    """Two coincident members lie on each other at every split point: both stay single
    elements, and the report and warnings say so."""
    from bridgesim.report import model_assumptions

    data = generate_warren().model_dump(mode="json")
    bc = next(m for m in data["members"] if m["id"] == "bc0n")
    data["members"].append({**bc, "id": "bc0n_twin"})
    r = analyze(Bridge.model_validate(data))
    assert sorted(r.buckling_unsplit_members) == ["bc0n", "bc0n_twin"]
    assert any("except 2 member(s)" in a for a in model_assumptions(r))


def test_dense_fallback_stays_within_a_modest_memory_budget() -> None:
    from bridgesim.stability import DENSE_FALLBACK_MAX_DOF

    assert 2 * DENSE_FALLBACK_MAX_DOF**2 * 8 <= 100e6  # two dense float64 matrices


@pytest.mark.parametrize("key, value", [("density_kg_m3", 5e-324), ("E_MPa", 1e-300),
                                        ("f_b_MPa", 1e-9)])
def test_material_values_that_would_underflow_are_rejected(key: str, value: float) -> None:
    from bridgesim.materials import load_material

    data = load_material("popsicle_birch").model_dump(mode="json")
    data[key]["value"] = value
    with pytest.raises(ValidationError, match="not physical"):
        Material.model_validate(data)


def test_band_endpoints_are_bounded() -> None:
    data = RuleSet.load().model_dump()
    data["rounding"]["length_mm"] = 2e-6
    for rule in data["rules"]:  # gap-free at any step, but 1e308 / 2e-6 overflows
        if rule["type"] == "bands":
            rule["bands"] = [{"max": 1e308}, {"min": 1e308}]
    with pytest.raises(ValidationError):
        RuleSet.model_validate(data)


def test_a_support_above_the_platform_fails_the_rule() -> None:
    from bridgesim.materials import load_material
    from bridgesim.rules import evaluate

    data = generate_warren().model_dump(mode="json")
    next(n for n in data["nodes"] if n["id"] == "P0n")["y_mm"] = 10.0
    rep = evaluate(Bridge.model_validate(data), load_material("popsicle_birch"))
    assert next(r for r in rep.results if r.key == "above_platform").passed is False


def test_horizontal_reaction_without_vertical_reaction_warns() -> None:
    """A node restrained only in DX carries thrust with no normal force, so no friction."""
    b = generate_warren()
    sup = b.supports.model_copy(update={"extra_restraints": {"B1n": ["DX"]}})
    r = analyze(b.model_copy(update={"supports": sup}))
    assert any("Support B1n needs a horizontal reaction" in w for w in r.warnings)


def test_duplicate_rule_keys_are_rejected() -> None:
    data = RuleSet.load().model_dump()
    data["rules"].append(dict(next(r for r in data["rules"] if r["key"] == "mass")))
    with pytest.raises(ValidationError, match="duplicate rule keys"):
        RuleSet.model_validate(data)


@pytest.mark.parametrize("keep", ["none", "info"])
def test_rules_files_must_check_something(keep: str) -> None:
    data = RuleSet.load().model_dump()
    data["rules"] = [] if keep == "none" else [r for r in data["rules"] if r["type"] == "info"]
    with pytest.raises(ValidationError, match="no rule the tool can check"):
        RuleSet.model_validate(data)


def test_rule_labels_must_be_single_line_but_notes_may_wrap() -> None:
    data = RuleSet.load().model_dump()
    data["rules"][0]["title"] = "Span\n  [PASS] forged result"
    with pytest.raises(ValidationError, match="line breaks"):
        RuleSet.model_validate(data)
    data = RuleSet.load().model_dump()
    data["rules"][0]["note"] = "first line\nsecond line"
    RuleSet.model_validate(data)
