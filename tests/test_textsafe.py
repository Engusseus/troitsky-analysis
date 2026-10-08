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
    with pytest.raises(ValidationError, match="not physical"):
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
    assert r"\frac{40\ \text{mm}}" in md_text and "F_u,δ (40 mm)" in md_text
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
