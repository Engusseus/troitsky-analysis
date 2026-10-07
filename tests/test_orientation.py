"""Lock Pynite 3.2.0's member local-axis convention as bridgesim relies on it.

bridgesim puts the section depth d on Pynite's local y axis (member rotation 0) and passes
Iz = b d^3 / 12, Iy = d b^3 / 12 (see ``sections.py``). For horizontal members local y is
global +Y, so vertical bending must use Iz; for vertical members local y is -X (going up),
so bending along X must use Iz. These tests use b = 10, d = 20 so Iz = 4 Iy and a mix-up
would change the deflection by a factor of 4.
"""

from __future__ import annotations

import numpy as np
import pytest
from Pynite import FEModel3D

from bridgesim.analysis import analyze
from bridgesim.geometry import half_extents, inclination_from_vertical_deg, local_axes
from bridgesim.model import COMBO, LOAD_CASE, build_model
from bridgesim.schema import Bridge, Deck, Material, Member, Node, PlateLoad, Section, Supports

B, D = 10.0, 20.0
L = 200.0
P = 10.0
IZ = B * D**3 / 12.0  # 6666.67 mm^4
IY = D * B**3 / 12.0  # 1666.67 mm^4
ALL_DOFS = ["DX", "DY", "DZ", "RX", "RY", "RZ"]

ROOT = (0.0, 100.0, 0.0)
TIPS = {
    "+X": (L, 100.0, 0.0),
    "+Y": (0.0, 100.0 + L, 0.0),
    "-Y": (0.0, 100.0 - L, 0.0),
    "+Z": (0.0, 100.0, L),
}


def two_node_bridge(pi: tuple[float, float, float], pj: tuple[float, float, float],
                    restrained: dict[str, list[str]], pinned: list[str], load_node: str,
                    b: float = B, d: float = D) -> Bridge:
    xs = (pi[0], pj[0])
    return Bridge(
        name="two-node test",
        nodes=[Node(id="R", x_mm=pi[0], y_mm=pi[1], z_mm=pi[2]),
               Node(id="T", x_mm=pj[0], y_mm=pj[1], z_mm=pj[2])],
        sections=[Section(id="rect", b_mm=b, d_mm=d)],
        members=[Member(id="m", i="R", j="T", section="rect", group="other")],
        deck=Deck(x_start_mm=min(xs) - 50.0, x_end_mm=max(xs) + 50.0, top_elevation_mm=50.0,
                  clear_width_mm=150.0),
        supports=Supports(pinned=pinned, extra_restraints=restrained),
        load=PlateLoad(P_ref_N=P, plate_length_mm=1e-3, x_center_mm=pj[0] if load_node == "T"
                       else pi[0], deck_support_nodes=[load_node]),
    )


def cantilever_tip_deflection(material: Material, tip: tuple[float, float, float],
                              direction: str) -> float:
    """Tip displacement (mm) of a cantilever fixed at ROOT under a tip force +P along
    ``direction`` (FX, FY or FZ). The crusher plate is parked on the fixed root node so
    it adds nothing; the test force is added to the same load case.
    """
    bridge = two_node_bridge(ROOT, tip, {"R": ALL_DOFS}, [], load_node="R")
    model = build_model(bridge, material)
    model.add_node_load("T", direction, P, case=LOAD_CASE)
    model.analyze_linear(check_stability=True)
    node = model.nodes["T"]
    return float(getattr(node, "D" + direction[1])[COMBO])


@pytest.mark.parametrize(
    ("axis", "direction", "inertia"),
    [
        ("+X", "FY", "Iz"),  # horizontal member, vertical load: primary plane
        ("+X", "FZ", "Iy"),  # horizontal member, transverse load: out of plane
        ("+Y", "FX", "Iz"),  # vertical member (going up), load along X: local y = -X
        ("+Y", "FZ", "Iy"),
        ("-Y", "FX", "Iz"),  # vertical member (going down): local y = +X
        ("+Z", "FY", "Iz"),  # horizontal member along Z (floor beam), vertical load
        ("+Z", "FX", "Iy"),
    ],
)
def test_cantilever_bending_axis(material: Material, axis: str, direction: str,
                                 inertia: str) -> None:
    """Cantilever tip deflection delta = P L^3 / (3 E I) with the expected I.

        b = 10, d = 20:  Iz = b d^3 / 12 = 6666.67 mm^4,  Iy = d b^3 / 12 = 1666.67 mm^4
        L = 200, P = 10 N, E = 10000 MPa:
        delta(Iz) = 10 * 200^3 / (3 * 10000 * 6666.67) = 0.4 mm
        delta(Iy) = 10 * 200^3 / (3 * 10000 * 1666.67) = 1.6 mm
    The other inertia would give a 4x different answer, so the 0.5 % tolerance pins the axis.
    """
    E = material.E_MPa.value
    I_exp = IZ if inertia == "Iz" else IY
    delta = cantilever_tip_deflection(material, TIPS[axis], direction)
    assert delta == pytest.approx(P * L**3 / (3 * E * I_exp), rel=5e-3)
    I_other = IY if inertia == "Iz" else IZ
    assert delta != pytest.approx(P * L**3 / (3 * E * I_other), rel=0.5)


def test_plate_loaded_cantilever_uses_iz(material: Material) -> None:
    """Same X cantilever, loaded downward through the crusher-plate path and analyze():
    delta_ref = P L^3 / (3 E Iz) = 0.4 mm and the root moment |M_z| = P L = 2000 N mm.
    """
    bridge = two_node_bridge(ROOT, TIPS["+X"], {"R": ALL_DOFS}, [], load_node="T")
    r = analyze(bridge, material)
    E = material.E_MPa.value
    assert r.delta_node == "T"
    assert r.delta_ref_mm == pytest.approx(P * L**3 / (3 * E * IZ), rel=5e-3)
    m = r.member("m")
    assert m.Mz_Nmm == pytest.approx(P * L, rel=1e-6)
    assert m.My_Nmm == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- local axes

AXIS_CASES = {
    "horizontal +X": ((0, 0, 0), (100, 0, 0)),
    "horizontal -X": ((100, 0, 0), (0, 0, 0)),
    "horizontal +Z": ((0, 50, 0), (0, 50, 100)),
    "horizontal -Z": ((0, 50, 100), (0, 50, 0)),
    "horizontal skew XZ": ((0, 180, -91), (143.75, 180, 91)),
    "vertical up": ((0, 0, 0), (0, 180, 0)),
    "vertical down": ((0, 180, 0), (0, 0, 0)),
    "inclined up XY": ((0, 180, -91), (71.875, 440, -91)),
    "inclined down XY": ((71.875, 440, -91), (143.75, 180, -91)),
    "inclined up YZ (pier brace)": ((0, 0, -91), (0, 180, 91)),
    "inclined up 3D": ((10, 20, 30), (110, 220, -70)),
    "inclined down 3D": ((110, 220, -70), (10, 20, 30)),
}


def pynite_direction_cosines(pi, pj) -> np.ndarray:
    m = FEModel3D()
    m.add_node("i", *map(float, pi))
    m.add_node("j", *map(float, pj))
    m.add_material("w", E=1e4, G=625.0, nu=0.3, rho=0.0)
    m.add_section("s", A=1.0, Iy=1.0, Iz=1.0, J=1.0)
    m.add_member("m", "i", "j", "w", "s")
    return m.members["m"].T()[0:3, 0:3]


@pytest.mark.parametrize("case", list(AXIS_CASES))
def test_local_axes_match_pynite(case: str) -> None:
    """bridgesim.geometry.local_axes reproduces Pynite's Member3D.T() rotation block."""
    pi, pj = AXIS_CASES[case]
    ours = local_axes(pi, pj)
    np.testing.assert_allclose(ours, pynite_direction_cosines(pi, pj), atol=1e-12)


@pytest.mark.parametrize("case", list(AXIS_CASES))
def test_local_axes_are_right_handed_orthonormal(case: str) -> None:
    """Rows are unit, mutually orthogonal, and z = x cross y (det = +1)."""
    ax = local_axes(*AXIS_CASES[case])
    np.testing.assert_allclose(ax @ ax.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(ax) == pytest.approx(1.0)


def test_local_axes_documented_conventions() -> None:
    """Horizontal: y = +Y. Vertical up: y = -X, z = +Z. Vertical down: y = +X, z = +Z.
    Inclined: z horizontal, y has an upward component.
    """
    np.testing.assert_allclose(local_axes((0, 0, 0), (1, 0, 0))[1], [0, 1, 0])
    np.testing.assert_allclose(local_axes((0, 0, 0), (0, 1, 0))[1:], [[-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(local_axes((0, 1, 0), (0, 0, 0))[1:], [[1, 0, 0], [0, 0, 1]])
    for pi, pj in (AXIS_CASES["inclined up 3D"], AXIS_CASES["inclined down 3D"]):
        ax = local_axes(pi, pj)
        assert ax[2][1] == pytest.approx(0.0, abs=1e-12)
        assert ax[1][1] > 0


def test_local_axes_match_pynite_on_default_bridge(default_bridge: Bridge,
                                                   material: Material) -> None:
    """Every member of the default Warren truss: local_axes == Pynite T()[0:3, 0:3]."""
    model = build_model(default_bridge, material)
    nodes = default_bridge.node_map()
    for mem in default_bridge.members:
        ours = local_axes(nodes[mem.i].xyz, nodes[mem.j].xyz)
        np.testing.assert_allclose(ours, model.members[mem.id].T()[0:3, 0:3], atol=1e-12,
                                   err_msg=mem.id)


def test_float_noise_near_zero_does_not_rotate_section(material: Material) -> None:
    """A vertical post at z = 0 whose top node carries float noise
    z = 0.1 + 0.2 - 0.3 = 5.55e-17 mm (typical of computed coordinates).

    bridgesim.geometry.local_axes: vertical -> y = -X, z = +Z (depth d along X).
    Pynite T(): isclose(0, 5.55e-17) is False -> inclined branch -> z = -X, y = -Z, so the
    analysed section is turned 90 deg (d along Z, bending along X uses d b^3 / 12).
    The measurement model and the analysis model must agree on member orientation; the
    schema snaps node coordinates to 1e-6 mm so both see z = 0 exactly.
    """
    noise = 0.1 + 0.2 - 0.3
    assert 0 < noise < 1e-15
    bridge = two_node_bridge((0.0, 0.0, 0.0), (0.0, 180.0, noise), {"R": ALL_DOFS}, [],
                             load_node="R")
    nodes = bridge.node_map()
    model = build_model(bridge, material)
    np.testing.assert_allclose(local_axes(nodes["R"].xyz, nodes["T"].xyz),
                               model.members["m"].T()[0:3, 0:3], atol=1e-9)


def test_local_axes_rejects_zero_length() -> None:
    """A zero-length member has no axes."""
    with pytest.raises(ValueError):
        local_axes((1, 2, 3), (1, 2, 3))


def test_half_extents() -> None:
    """Cross-section half-size along X, Y, Z for b = 10, d = 20:

        horizontal along X: (0, d/2, b/2) = (0, 10, 5)
        vertical (up):      (d/2, 0, b/2) = (10, 0, 5)   (local y = -X)
        horizontal along Z: (b/2, d/2, 0) = (5, 10, 0)   (local z = x cross y = -X)
    """
    np.testing.assert_allclose(half_extents((0, 0, 0), (100, 0, 0), B, D), [0, 10, 5])
    np.testing.assert_allclose(half_extents((0, 0, 0), (0, 100, 0), B, D), [10, 0, 5])
    np.testing.assert_allclose(half_extents((0, 0, 0), (0, 0, 100), B, D), [5, 10, 0],
                               atol=1e-12)


def test_inclination_from_vertical() -> None:
    """Angle from vertical = acos(|dy| / L): 0 deg vertical, 90 deg horizontal, 45 deg."""
    assert inclination_from_vertical_deg((0, 0, 0), (0, 10, 0)) == pytest.approx(0.0)
    assert inclination_from_vertical_deg((0, 0, 0), (10, 0, 0)) == pytest.approx(90.0)
    assert inclination_from_vertical_deg((0, 0, 0), (10, 10, 0)) == pytest.approx(45.0)


# --------------------------------------------------------------------------- axial sign

def test_hanging_bar_is_in_tension(material: Material) -> None:
    """A bar hanging from a fixed node with P = 10 N pulling down at its free end.

        N = +P (tension positive in bridgesim); elongation delta = P L / (E A)
          = 10 * 200 / (10000 * 200) = 0.001 mm downward.
    Pynite's own axial force is compression-positive, so its raw value is -P.
    """
    top, bottom = (0.0, 300.0, 0.0), (0.0, 100.0, 0.0)
    bridge = two_node_bridge(top, bottom, {"R": ["RX", "RY", "RZ"]}, ["R"], load_node="T")
    r = analyze(bridge, material)
    E = material.E_MPa.value
    axial = r.member("m").N_N
    assert axial == pytest.approx(+P, rel=1e-9)
    assert r.delta_ref_mm == pytest.approx(P * L / (E * B * D), rel=1e-6)
    pm = r.fe_model.members["m"]
    assert pm.axial(L / 2, COMBO) == pytest.approx(-P, rel=1e-9)


def test_loaded_column_is_in_compression(material: Material) -> None:
    """A fixed-base column with P = 10 N pushing down on its top: N = -P (compression)."""
    base, top = (0.0, 0.0, 0.0), (0.0, 200.0, 0.0)
    bridge = two_node_bridge(base, top, {"R": ["RX", "RY", "RZ"]}, ["R"], load_node="T")
    r = analyze(bridge, material)
    axial = r.member("m").N_N
    assert axial == pytest.approx(-P, rel=1e-9)
    assert r.fe_model.members["m"].axial(L / 2, COMBO) == pytest.approx(+P, rel=1e-9)
