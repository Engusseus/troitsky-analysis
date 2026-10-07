"""Validation: planar pin-jointed Warren truss against the method of sections.

Geometry (XY plane, z = 0): span L = 1200 mm, n = 6 panels of p = 200 mm, height
h = 200 mm (chord centre lines). Bottom nodes B_i at x = i p (i = 0..6), top nodes T_i at
x = (i + 1/2) p (i = 0..5). Members: bottom chords B_i-B_{i+1}, top chords T_i-T_{i+1},
"up" diagonals B_i-T_i and "down" diagonals T_i-B_{i+1}. Every member is released for
in-plane bending at both ends (Rzi, Rzj). Every node is restrained out of plane (DZ, RX,
RY) and in RZ: with all members moment-released the node rotations RZ carry no stiffness
and no moment, so restraining them changes nothing. B0 is pinned, B6 is a roller; the
whole P = 1000 N acts down at the mid-span bottom node B3 (x = 600).
"""

from __future__ import annotations

import math

import pytest

from bridgesim.analysis import AnalysisResult, analyze
from bridgesim.schema import Bridge, Deck, Material, Member, Node, PlateLoad, Section, Supports

SPAN = 1200.0
N_PANELS = 6
H = 200.0
P = 1000.0
Y_B = 100.0
PANEL = SPAN / N_PANELS
SIDE = 10.0  # 10 x 10 mm members: A = 100 mm^2
SIN_T = H / math.hypot(PANEL / 2, H)  # diagonal angle from horizontal


def warren_truss() -> Bridge:
    nodes = [Node(id=f"B{i}", x_mm=i * PANEL, y_mm=Y_B, z_mm=0.0) for i in range(N_PANELS + 1)]
    nodes += [Node(id=f"T{i}", x_mm=(i + 0.5) * PANEL, y_mm=Y_B + H, z_mm=0.0)
              for i in range(N_PANELS)]
    rel = ["Rzi", "Rzj"]
    members = []
    for i in range(N_PANELS):
        members.append(Member(id=f"bc{i}", i=f"B{i}", j=f"B{i + 1}", section="s",
                              group="bottom_chord", releases=rel))
        members.append(Member(id=f"du{i}", i=f"B{i}", j=f"T{i}", section="s",
                              group="diagonal", releases=rel))
        members.append(Member(id=f"dd{i}", i=f"T{i}", j=f"B{i + 1}", section="s",
                              group="diagonal", releases=rel))
    for i in range(N_PANELS - 1):
        members.append(Member(id=f"tc{i}", i=f"T{i}", j=f"T{i + 1}", section="s",
                              group="top_chord", releases=rel))
    restraints = {n.id: ["DZ", "RX", "RY", "RZ"] for n in nodes}
    return Bridge(
        name="Planar pin-jointed Warren truss",
        nodes=nodes,
        sections=[Section(id="s", b_mm=SIDE, d_mm=SIDE)],
        members=members,
        deck=Deck(x_start_mm=0.0, x_end_mm=SPAN, top_elevation_mm=Y_B + 10.0,
                  clear_width_mm=150.0),
        supports=Supports(pinned=["B0"], roller=[f"B{N_PANELS}"], extra_restraints=restraints),
        load=PlateLoad(P_ref_N=P, plate_length_mm=1e-3, x_center_mm=SPAN / 2,
                       deck_support_nodes=[f"B{N_PANELS // 2}"]),
    )


def moment(x: float) -> float:
    """Simple-beam bending moment under the central point load: (P/2) min(x, L - x)."""
    return P / 2 * min(x, SPAN - x)


def shear(x: float) -> float:
    """Simple-beam shear (upward force on the left free body): +P/2 left of mid, -P/2 right."""
    return P / 2 if x < SPAN / 2 else -P / 2


def expected_forces() -> dict[str, float]:
    """Hand member forces, tension positive, by the method of sections.

    Bottom chord i (B_i-B_{i+1}): cut between B_i and T_i; the other two cut members
    (up-diagonal B_i-T_i and top chord T_{i-1}-T_i) meet at T_i, so moments about T_i give
        F_bc,i = + M(x_Ti) / h,   x_Ti = (i + 1/2) p
        -> 250, 750, 1250, 1250, 750, 250 N
    Top chord i (T_i-T_{i+1}): cut just left of B_{i+1}; the bottom chord B_i-B_{i+1} and
    down-diagonal T_i-B_{i+1} meet at B_{i+1}, so moments about B_{i+1} give
        F_tc,i = - M(x_B(i+1)) / h,   x_B(i+1) = (i + 1) p
        -> -500, -1000, -1500, -1000, -500 N
    Diagonals: vertical equilibrium of the left free body with shear V = +-P/2 and
    sin(theta) = h / sqrt((p/2)^2 + h^2) = 200 / 223.61 = 0.8944:
        up-diagonal   (rising to the right):  F = -V / sin(theta)
        down-diagonal (falling to the right): F = +V / sin(theta)
        |F| = 500 / 0.8944 = 559.02 N (compression in left up-diagonals, tension in left
        down-diagonals, reversed right of mid-span).
    """
    out: dict[str, float] = {}
    for i in range(N_PANELS):
        out[f"bc{i}"] = moment((i + 0.5) * PANEL) / H
        x_mid_diag_up = (i + 0.25) * PANEL
        x_mid_diag_dn = (i + 0.75) * PANEL
        out[f"du{i}"] = -shear(x_mid_diag_up) / SIN_T
        out[f"dd{i}"] = +shear(x_mid_diag_dn) / SIN_T
    for i in range(N_PANELS - 1):
        out[f"tc{i}"] = -moment((i + 1) * PANEL) / H
    return out


@pytest.fixture(scope="module")
def truss_result(material: Material) -> AnalysisResult:
    return analyze(warren_truss(), material)


def test_hand_values_are_as_derived() -> None:
    """Spot-check the hand calculation itself (see ``expected_forces``).

        F_bc,2 = M(500)/h = 500*500/200 = 1250 N;  F_tc,2 = -M(600)/h = -300000/200 = -1500 N
        |F_diag| = 500 / (200 / sqrt(100^2 + 200^2)) = 500 * 223.607 / 200 = 559.017 N
    """
    f = expected_forces()
    assert f["bc0"] == pytest.approx(250.0)
    assert f["bc2"] == pytest.approx(1250.0)
    assert f["tc2"] == pytest.approx(-1500.0)
    assert f["du0"] == pytest.approx(-559.017, rel=1e-6)
    assert f["dd0"] == pytest.approx(559.017, rel=1e-6)
    assert f["du3"] == pytest.approx(559.017, rel=1e-6)
    assert f["dd5"] == pytest.approx(-559.017, rel=1e-6)


@pytest.mark.parametrize("prefix", ["bc", "tc"])
def test_chord_forces_match_method_of_sections(truss_result: AnalysisResult,
                                               prefix: str) -> None:
    """Chord forces: bottom F = +M(x_T)/h at the opposite top node, top F = -M(x_B)/h at
    the opposite bottom node, M(x) = (P/2) x for x <= L/2 (see ``expected_forces``).
    Tolerance 1 %.
    """
    exp = expected_forces()
    keys = [k for k in exp if k.startswith(prefix)]
    assert keys
    for k in keys:
        axial = truss_result.member(k).N_N
        assert axial == pytest.approx(exp[k], rel=1e-2), k


def test_diagonal_forces_match_shear(truss_result: AnalysisResult) -> None:
    """Diagonals: F = -+V / sin(theta) = -+559.02 N (see ``expected_forces``), 1 % tolerance."""
    exp = expected_forces()
    for k in [k for k in exp if k.startswith("d")]:
        axial = truss_result.member(k).N_N
        assert axial == pytest.approx(exp[k], rel=1e-2), k


def test_members_carry_no_bending(truss_result: AnalysisResult) -> None:
    """Pin-jointed truss with nodal loads only: every member is a two-force member,
    so M_y = M_z = 0 (within round-off, here < 1e-6 * P * p = 0.2 N mm).
    """
    tol = 1e-6 * P * PANEL
    for m in truss_result.members:
        assert abs(m.Mz_Nmm) < tol, m.id
        assert abs(m.My_Nmm) < tol, m.id


def test_reactions(truss_result: AnalysisResult) -> None:
    """Symmetric truss, central load: R_B0 = R_B6 = P/2 = 500 N, no horizontal reaction."""
    for r in truss_result.reactions:
        reaction = (r.FX_N, r.FY_N)
        assert reaction == pytest.approx((0.0, P / 2), abs=1e-6)


def test_midspan_deflection_by_unit_load_method(truss_result: AnalysisResult,
                                                material: Material) -> None:
    """Virtual work (unit-load method) with the hand member forces N_i:

        delta = sum_i N_i n_i L_i / (E A),   n_i = N_i / P
        bottom chords: 200 * 2 * (250^2 + 750^2 + 1250^2)          = 8.750e8
        top chords:    200 * (500^2 + 1000^2 + 1500^2 + 1000^2 + 500^2) = 9.500e8
        diagonals:     12 * 559.017^2 * 223.607                     = 8.385e8
        delta = 2.6635e9 / (P E A = 1000 * 10000 * 100) = 2.664 mm
    Tolerance 1 %.
    """
    E = material.E_MPa.value
    A = SIDE * SIDE
    exp = expected_forces()
    L_diag = math.hypot(PANEL / 2, H)
    lengths = {k: (L_diag if k.startswith("d") else PANEL) for k in exp}
    delta = sum(N * (N / P) * lengths[k] / (E * A) for k, N in exp.items())
    assert delta == pytest.approx(2.6635, rel=1e-3)
    assert truss_result.delta_node == f"B{N_PANELS // 2}"
    assert truss_result.delta_ref_mm == pytest.approx(delta, rel=1e-2)
