"""The deck as an in-plane shear diaphragm (equivalent diagonals)."""

import pytest
from pydantic import ValidationError

from bridgesim.analysis import analyze
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.model import COMBO, LOAD_CASE, build_model, diaphragm_bars
from bridgesim.report import model_assumptions
from bridgesim.schema import Bridge

A, B, T, G = 200.0, 150.0, 2.0, 100.0  # panel length, width, thickness (mm); G (MPa)
Y = 100.0
STIFF = {"b_mm": 100.0, "d_mm": 100.0}  # chords and strut: axially rigid next to the deck


def _panel() -> Bridge:
    """One deck panel n0-n1-f1-f0 (a = 200 along X, b = 150 across), held at x = 0. The
    chords n0-n1 and f0-f1 are pin-ended, so only the deck resists sideways movement of the
    strut n1-f1. Rotations and vertical movement of n1, f1 are held to keep it in plane."""
    held = ["DY", "RX", "RY", "RZ"]
    pins = ["Ryi", "Rzi", "Ryj", "Rzj"]
    return Bridge.model_validate({
        "nodes": [{"id": "n0", "x_mm": 0, "y_mm": Y, "z_mm": -B / 2},
                  {"id": "f0", "x_mm": 0, "y_mm": Y, "z_mm": B / 2},
                  {"id": "n1", "x_mm": A, "y_mm": Y, "z_mm": -B / 2},
                  {"id": "f1", "x_mm": A, "y_mm": Y, "z_mm": B / 2}],
        "sections": [{"id": "stiff", **STIFF}],
        "members": [{"id": "cn", "i": "n0", "j": "n1", "section": "stiff", "releases": pins},
                    {"id": "cf", "i": "f0", "j": "f1", "section": "stiff", "releases": pins},
                    {"id": "strut", "i": "n1", "j": "f1", "section": "stiff"}],
        "deck": {"x_start_mm": 0, "x_end_mm": A, "top_elevation_mm": Y, "clear_width_mm": B,
                 "thickness_mm": T,
                 "diaphragm": {"G_MPa": G, "connected_to": ["n0", "n1", "f0", "f1"]}},
        "supports": {"pinned": [], "extra_restraints": {
            "n0": ["DX", "DY", "DZ", "RX", "RY", "RZ"], "f0": ["DX", "DY", "DZ", "RX", "RY", "RZ"],
            "n1": held, "f1": held}},
        "load": {"plate_length_mm": 1e-3, "x_center_mm": A, "deck_support_nodes": ["n1"]},
    })


def test_panel_racks_with_the_plate_shear_stiffness(material) -> None:
    """Shear strain Delta / a over the b x t section: H = G t b Delta / a, so
    k = 100 x 2 x 150 / 200 = 150 N/mm and H = 150 N moves the free end by 1 mm.
    Diagonal area A_d = G t L^3 / (2 E a b) = 100 x 2 x 250^3 / (2 x 10000 x 200 x 150)
    = 5.208 mm^2 (L = 250 mm)."""
    bridge = _panel()
    bars = diaphragm_bars(bridge, material.E_MPa.value)
    assert [(i, j) for i, j, _ in bars] == [("n0", "f1"), ("f0", "n1")]
    assert [a for *_, a in bars] == pytest.approx([G * T * 250**3 / (2e4 * A * B)] * 2)
    model = build_model(bridge, material, P_N=1e-9)
    model.add_node_load("n1", "FZ", 150.0, case=LOAD_CASE)
    model.analyze_linear()
    assert model.nodes["n1"].DZ[COMBO] == pytest.approx(1.0, rel=2e-3)
    assert model.nodes["f1"].DZ[COMBO] == pytest.approx(1.0, rel=2e-3)


def test_no_diaphragm_no_lateral_stiffness(material) -> None:
    """Without the deck the same panel is a mechanism sideways."""
    data = _panel().model_dump(mode="json")
    data["deck"]["diaphragm"] = None
    model = build_model(Bridge.model_validate(data), material, P_N=1e-9)
    model.add_node_load("n1", "FZ", 150.0, case=LOAD_CASE)
    with pytest.raises(Exception):  # noqa: B017  (Pynite raises a bare Exception)
        model.analyze_linear(check_stability=True)


def _warren_deck(G: float | None, bottom: str = "none") -> Bridge:
    p = WarrenParams(top_bracing="none", bottom_bracing=bottom)
    data = generate_warren(p).model_dump(mode="json")
    if G is not None:
        data["deck"]["diaphragm"] = {"G_MPa": G, "connected_to": [
            f"B{i}{s}" for i in range(p.n_panels + 1) for s in "nf"]}
    return Bridge.model_validate(data)


def test_deck_diaphragm_stiffens_the_bridge_sideways() -> None:
    """With no bottom bracing, a stiffer deck raises the sideways buckling load, towards the
    value with X-bracing under the deck. Bars coinciding with bottom X-braces do not stop
    those braces being split for the buckling mesh."""
    loads = [analyze(_warren_deck(G)).Fu_buckling_N for G in (None, 50.0, 625.0)]
    assert loads[0] < loads[1] < loads[2]
    braced = analyze(_warren_deck(150.0, bottom="x"))
    assert braced.buckling_unsplit_members == []
    assert [m.id for m in braced.members] == [m.id for m in braced.bridge.members]
    assert any(a.startswith("The deck is a shear diaphragm in its own plane (G = 150 MPa")
               for a in model_assumptions(braced))


@pytest.mark.parametrize("nodes, message", [
    (["B0n", "B1n", "B0f"], "List should have at least 4 items"),
    (["B0n", "B1n", "B2n", "B0f"], "3 node\\(s\\) on the -Z side and 1"),
    (["B0n", "B1n", "B0f", "B2f"], "stations do not line up"),
    (["B0n", "B1n", "C0", "C1"], "lie on the deck centreline"),
    (["B0n", "B1n", "B0f", "nope"], "Unknown node 'nope'"),
    (["B0n", "B0n", "B0f", "B1f"], "Nodes listed twice"),
])
def test_bad_diaphragm_nodes_are_explained(nodes: list, message: str) -> None:
    data = generate_warren().model_dump(mode="json")
    data["deck"]["diaphragm"] = {"G_MPa": 100.0, "connected_to": nodes}
    with pytest.raises(ValidationError, match=message):
        Bridge.model_validate(data)


def test_node_on_a_diaphragm_diagonal_is_rejected() -> None:
    """Pynite would join a node on a diagonal's line to it, an unintended joint. (No bottom
    X-braces here: they lie on the same lines and would be reported first.)"""
    data = generate_warren(WarrenParams(bottom_bracing="none")).model_dump(mode="json")
    data["deck"]["diaphragm"] = {"G_MPa": 100.0, "connected_to": ["B0n", "B1n", "B0f", "B1f"]}
    b0n, b1f = (next(n for n in data["nodes"] if n["id"] == k) for k in ("B0n", "B1f"))
    data["nodes"].append({"id": "hub", **{k: (b0n[k] + b1f[k]) / 2 + (k == "x_mm") * 10
                                         for k in ("x_mm", "y_mm", "z_mm")}})
    Bridge.model_validate(data)  # off the line: fine
    data["nodes"][-1] = {"id": "hub", **{k: b0n[k] + 0.25 * (b1f[k] - b0n[k])
                                         for k in ("x_mm", "y_mm", "z_mm")}}
    with pytest.raises(ValidationError, match="lies on the deck diaphragm's diagonal"):
        Bridge.model_validate(data)


def test_diaphragm_round_trips() -> None:
    b = _warren_deck(150.0)
    assert Bridge.from_yaml_str(b.to_yaml()) == b
    assert "diaphragm" not in generate_warren().to_yaml()
