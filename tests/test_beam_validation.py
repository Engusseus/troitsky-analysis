"""Validation: simply supported beam with a mid-span point load against closed-form results.

Model: a b x d = 10 x 20 mm rectangle (d vertical), span L = 1000 mm along X at y = 100,
split into two frame elements at the mid-span node N1. N0 is pinned (DX, DY, DZ) plus a
torsion restraint (RX) to remove the rigid-body spin about the beam axis; N2 is a roller
(DY, DZ). The crusher plate is shrunk to 1e-3 mm and centred on N1 so the whole P_ref acts
there as a point load.
"""

from __future__ import annotations

import math

import pytest

from bridgesim.analysis import DEFLECTION_LIMIT_MM, AnalysisResult, analyze
from bridgesim.loads import plate_nodal_loads
from bridgesim.schema import Bridge, Deck, Material, Member, Node, PlateLoad, Section, Supports

L = 1000.0  # mm
B, D = 10.0, 20.0  # mm, d vertical (local y)
P = 1000.0  # N
Y0 = 100.0  # beam axis elevation, mm


def simply_supported_beam(P_ref: float = P) -> Bridge:
    return Bridge(
        name="Simply supported beam",
        nodes=[
            Node(id="N0", x_mm=0.0, y_mm=Y0, z_mm=0.0),
            Node(id="N1", x_mm=L / 2, y_mm=Y0, z_mm=0.0),
            Node(id="N2", x_mm=L, y_mm=Y0, z_mm=0.0),
        ],
        sections=[Section(id="rect", b_mm=B, d_mm=D)],
        members=[
            Member(id="m1", i="N0", j="N1", section="rect", group="other"),
            Member(id="m2", i="N1", j="N2", section="rect", group="other"),
        ],
        deck=Deck(x_start_mm=0.0, x_end_mm=L, top_elevation_mm=Y0 + D / 2,
                  clear_width_mm=150.0),
        supports=Supports(pinned=["N0"], roller=["N2"], extra_restraints={"N0": ["RX"]}),
        load=PlateLoad(P_ref_N=P_ref, plate_length_mm=1e-3, x_center_mm=L / 2,
                       deck_support_nodes=["N1"]),
    )


@pytest.fixture(scope="module")
def beam() -> Bridge:
    return simply_supported_beam()


@pytest.fixture(scope="module")
def beam_result(beam: Bridge, material: Material) -> AnalysisResult:
    return analyze(beam, material)


def _Iz() -> float:
    return B * D**3 / 12.0


def test_whole_load_goes_to_mid_node(beam: Bridge) -> None:
    """A 1e-3 mm plate centred on the only deck-support node puts all of P_ref on it.

    Single station at x = 500: the plate halves left and right of it are both "beyond the
    outermost station" and go to that station, so F(N1) = w * 1e-3 = P.
    """
    loads = plate_nodal_loads(beam)
    assert list(loads) == ["N1"]
    assert loads["N1"] == pytest.approx(P, rel=1e-9)


def test_midspan_deflection_matches_closed_form(beam_result: AnalysisResult,
                                                material: Material) -> None:
    """Mid-span deflection of a simply supported beam under a central point load.

        I = b d^3 / 12 = 10 * 20^3 / 12 = 6666.67 mm^4
        delta = P L^3 / (48 E I) = 1000 * 1000^3 / (48 * 10000 * 6666.67) = 312.5 mm

    (Euler-Bernoulli, no shear deformation, which is what Pynite's frame element models.)
    Tolerance 0.5 %.
    """
    E = material.E_MPa.value
    delta = P * L**3 / (48.0 * E * _Iz())
    assert delta == pytest.approx(312.5, rel=1e-9)  # hand value with E = 10000 MPa
    assert beam_result.delta_node == "N1"
    assert beam_result.delta_ref_mm == pytest.approx(delta, rel=5e-3)


def test_fu_deflection_is_load_scaled_to_50mm(beam_result: AnalysisResult) -> None:
    """Deflection-limited load from linearity: F_u,delta = P_ref * 50 / delta_ref.

    With delta_ref = 312.5 mm: F_u,delta = 1000 * 50 / 312.5 = 160 N.
    """
    r = beam_result
    assert r.Fu_deflection_N == pytest.approx(r.P_ref_N * DEFLECTION_LIMIT_MM / r.delta_ref_mm,
                                              rel=1e-12)
    assert r.Fu_deflection_N == pytest.approx(160.0, rel=5e-3)


def test_support_reactions(beam_result: AnalysisResult) -> None:
    """Symmetric beam: R_N0 = R_N2 = P / 2 = 500 N upwards, no horizontal reactions."""
    rx = {r.node: r for r in beam_result.reactions}
    assert set(rx) == {"N0", "N2"}
    for r in rx.values():
        reaction = (r.FX_N, r.FY_N, r.FZ_N)
        assert reaction == pytest.approx((0.0, P / 2, 0.0), abs=1e-6)


def test_member_forces(beam_result: AnalysisResult) -> None:
    """Internal forces of each half: |M_z|max = P L / 4, V = P / 2, N = 0, M_y = 0.

        M_max = 1000 * 1000 / 4 = 250000 N mm at mid-span (in the vertical plane, so about
        local z for a horizontal member); V = 500 N; no axial force (roller end).
    """
    for mid in ("m1", "m2"):
        m = beam_result.member(mid)
        assert m.Mz_Nmm == pytest.approx(P * L / 4, rel=1e-6)
        assert m.Vy_N == pytest.approx(P / 2, rel=1e-6)
        assert abs(m.N_N) < 1e-6
        assert m.My_Nmm == pytest.approx(0.0, abs=1e-6)
        assert m.Vz_N == pytest.approx(0.0, abs=1e-6)


def test_bending_governs_strength(beam_result: AnalysisResult, material: Material) -> None:
    """Load-factor method on the beam: bending about local z governs.

        S_z = b d^2 / 6 = 10 * 400 / 6 = 666.67 mm^3,  M_R = f_b S_z = 50 * 666.67 = 33333 N mm
        U = M_max / M_R = 250000 / 33333 = 7.5  ->  F_u,strength = P_ref / U = 133.3 N
        (= 4 f_b S_z / L). The deflection limit gives 160 N, so F_u,p = 133.3 N.
    """
    f_b = material.f_b_MPa.value
    Sz = B * D**2 / 6.0
    Fu_strength = 4.0 * f_b * Sz / L
    assert Fu_strength == pytest.approx(133.333, rel=1e-4)
    r = beam_result
    assert r.Fu_strength_N == pytest.approx(Fu_strength, rel=1e-6)
    # no axial force anywhere, so no other limit state can be lower than these two
    limits = [v for k, v in vars(r).items()
              if k.startswith("Fu_") and k.endswith("_N") and k != "Fu_pred_N"
              and isinstance(v, int | float)]
    assert r.Fu_pred_N == min(limits)
    assert r.Fu_pred_N == pytest.approx(min(Fu_strength, r.Fu_deflection_N), rel=1e-12)
    assert r.governing_mode == "bending_z"
    assert r.governing_member in {"m1", "m2"}
    assert math.isfinite(r.Fu_pred_N)


def test_response_is_linear_in_load(material: Material) -> None:
    """Linear analysis: halving P_ref halves delta_ref but leaves F_u unchanged.

        delta(P/2) = (P/2) L^3 / (48 E I) = 156.25 mm; F_u = P_ref / U is load-independent.
    """
    full = analyze(simply_supported_beam(P), material)
    half = analyze(simply_supported_beam(P / 2), material)
    assert half.delta_ref_mm == pytest.approx(full.delta_ref_mm / 2, rel=1e-9)
    assert half.Fu_pred_N == pytest.approx(full.Fu_pred_N, rel=1e-9)
    assert half.Fu_deflection_N == pytest.approx(full.Fu_deflection_N, rel=1e-9)
