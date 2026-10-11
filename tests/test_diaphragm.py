"""The deck as an in-plane shear diaphragm (constant-shear panels, buckling analysis only)."""

import numpy as np
import pytest
from pydantic import ValidationError

from bridgesim.analysis import analyze
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.model import build_model
from bridgesim.report import model_assumptions
from bridgesim.schema import Bridge
from bridgesim.stability import diaphragm_terms, global_buckling

A, B, T, G = 200.0, 150.0, 2.0, 100.0  # panel length, width, thickness (mm); G (MPa)
H = 100.0  # deck height (mm)
ALL = ["DX", "DY", "DZ", "RX", "RY", "RZ"]


def _post_under_deck(**diaphragm) -> Bridge:
    """One deck panel n0-n1-f1-f0 (a = 200 along X, b = w = 150 across) at y = 100, held at
    n0, f0 and f1. A stocky post rises from a pinned base to corner n1 and carries the load;
    n1 is held along X, so only the deck stops it swaying sideways (Z)."""
    return Bridge.model_validate({
        "nodes": [{"id": "c0", "x_mm": A, "y_mm": 0, "z_mm": -B / 2},
                  {"id": "n1", "x_mm": A, "y_mm": H, "z_mm": -B / 2},
                  {"id": "n0", "x_mm": 0, "y_mm": H, "z_mm": -B / 2},
                  {"id": "f0", "x_mm": 0, "y_mm": H, "z_mm": B / 2},
                  {"id": "f1", "x_mm": A, "y_mm": H, "z_mm": B / 2}],
        "sections": [{"id": "post", "b_mm": 50.0, "d_mm": 50.0}],
        "members": [{"id": "post", "i": "c0", "j": "n1", "section": "post"}],
        "deck": {"x_start_mm": 0, "x_end_mm": A, "top_elevation_mm": H, "clear_width_mm": B,
                 "thickness_mm": T, "diaphragm": {"G_MPa": G, "connected_to": [
                     "n0", "n1", "f0", "f1"], **diaphragm}},
        "supports": {"pinned": ["c0"], "extra_restraints": {
            "c0": ["RY"], "n1": ["DX"], "n0": ALL, "f0": ALL, "f1": ALL}},
        "load": {"P_ref_N": 1.0, "plate_length_mm": 1e-3, "x_center_mm": A,
                 "deck_support_nodes": ["n1"]},
    })


def _force(k: np.ndarray, moves: dict[str, tuple]) -> np.ndarray:
    """Corner forces (n0, n1, f0, f1 rows; X, Y, Z columns) for the given movements."""
    u = np.concatenate([moves.get(n, (0.0, 0.0, 0.0)) for n in ("n0", "n1", "f0", "f1")])
    return (k @ u).reshape(4, 3)


@pytest.fixture(scope="module")
def panel_k() -> np.ndarray:
    [(nodes, k)] = diaphragm_terms(_post_under_deck())
    assert nodes == ["n0", "n1", "f0", "f1"]
    return k


def test_panel_racks_with_the_plate_shear_stiffness(panel_k) -> None:
    """Shear strain Delta / a over the w x t section: V = G t w Delta / a, so
    k = 100 x 2 x 150 / 200 = 150 N/mm; moving n1 and f1 1 mm across takes 150 N, resisted
    at n0 and f0."""
    f = _force(panel_k, {"n1": (0, 0, 1.0), "f1": (0, 0, 1.0)})
    assert f[1, 2] + f[3, 2] == pytest.approx(G * T * B / A)
    assert f[:, 2].sum() == pytest.approx(0.0, abs=1e-9)


def test_panel_takes_no_chord_force_and_no_rigid_body_force(panel_k) -> None:
    """Stretching the chords (n1, f1 moving 0.1 mm along X), moving the panel as a whole or
    turning it in its plane (u = theta x r about Y) strains no shear: no force at all."""
    stretch = {"n1": (0.1, 0, 0), "f1": (0.1, 0, 0)}
    shift = {n: (0.3, -0.2, 0.5) for n in ("n0", "n1", "f0", "f1")}
    pos = {"n0": (0, -B / 2), "n1": (A, -B / 2), "f0": (0, B / 2), "f1": (A, B / 2)}
    turn = {n: (1e-3 * z, 0, -1e-3 * x) for n, (x, z) in pos.items()}  # about +Y
    for moves in (stretch, shift, turn):
        assert np.abs(_force(panel_k, moves)).max() == pytest.approx(0.0, abs=1e-9)


def test_plate_width_and_thickness_set_the_stiffness() -> None:
    """k = G t w / a: half the width (or the thickness) halves it."""
    full = diaphragm_terms(_post_under_deck())[0][1]
    for opt in ({"width_mm": B / 2}, {"thickness_mm": T / 2}):
        assert diaphragm_terms(_post_under_deck(**opt))[0][1] == pytest.approx(full / 2)


def test_post_braced_by_the_deck_sways_at_k_h(material) -> None:
    """A straight post pinned at the base and held sideways at the top by a spring k
    buckles by tilting, at P_cr = k H (the post stays straight, so its EI does not enter;
    its own Euler load is far higher). Here only corner n1 moves, so gamma = Delta / (2 a)
    and k = G t w / (4 a) = 100 x 2 x 150 / 800 = 37.5 N/mm: P_cr = 3750 N. A very soft
    spring (1e-6 N/mm) holds n1 in the static solve, where the deck is not modelled."""
    bridge = _post_under_deck()
    model = build_model(bridge, material)
    model.add_node("anchor", A, H, -B / 2 - 100.0)
    model.def_support("anchor", True, True, True, True, True, True)
    model.add_spring("soft", "n1", "anchor", 1e-6)
    model.analyze_linear()
    buck = global_buckling(model, bridge, extra=diaphragm_terms(bridge))
    assert buck.lambda_cr == pytest.approx((G * T * B / (4 * A) + 1e-6) * H, rel=1e-6)
    assert buck.description.startswith("sideways (Z) movement, largest at n1")
    assert global_buckling(model, bridge).lambda_cr == pytest.approx(1e-6 * H, rel=1e-3)


def _warren_deck(G: float | None, bottom: str = "none") -> Bridge:
    p = WarrenParams(top_bracing="none", bottom_bracing=bottom)
    data = generate_warren(p).model_dump(mode="json")
    if G is not None:
        data["deck"]["diaphragm"] = {"G_MPa": G, "connected_to": [
            f"B{i}{s}" for i in range(p.n_panels + 1) for s in "nf"]}
    return Bridge.model_validate(data)


def test_deck_diaphragm_stiffens_the_bridge_sideways() -> None:
    """With no bottom bracing, a stiffer deck raises the sideways buckling load, towards the
    value with X-bracing under the deck."""
    loads = [analyze(_warren_deck(G)).Fu_buckling_N for G in (None, 50.0, 625.0)]
    assert loads[0] < loads[1] < loads[2]
    braced = analyze(_warren_deck(None, bottom="x")).Fu_buckling_N
    assert loads[2] == pytest.approx(braced, rel=0.01)


def test_deck_diaphragm_changes_no_member_force() -> None:
    """The deck resists shear only: however stiff, it takes none of the bottom chords'
    tension, so every member force, utilisation and F_u,p (strength-governed here) stays
    the same."""
    plain, stiff = analyze(_warren_deck(None)), analyze(_warren_deck(5000.0))
    assert stiff.Fu_buckling_N > plain.Fu_buckling_N
    for a, b in zip(plain.members, stiff.members, strict=True):
        assert (a.id, a.N_N, a.U) == (b.id, b.N_N, b.U)
    assert stiff.Fu_pred_N == plain.Fu_pred_N == plain.Fu_strength_N
    text = model_assumptions(stiff)
    assert any(a.startswith("The deck is a shear diaphragm in its own plane (G = 5000 MPa")
               and "takes no chord force" in a for a in text)


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


def test_diaphragm_round_trips() -> None:
    b = _warren_deck(150.0)
    assert Bridge.from_yaml_str(b.to_yaml()) == b
    assert "diaphragm" not in generate_warren().to_yaml()
