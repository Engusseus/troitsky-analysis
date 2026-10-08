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
