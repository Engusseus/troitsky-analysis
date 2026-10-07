"""Euler buckling of a single popsicle stick and of small stick stacks.

Single stick: b = 10 mm, t = 2 mm, L = 115 mm, E = 10000 MPa, pin-ended (K = 1):

    I_min = b t^3 / 12 = 10 * 2^3 / 12 = 6.6667 mm^4
    P_cr = pi^2 E I_min / (K L)^2 = 9.8696 * 10000 * 6.6667 / 115^2 = 49.75 N
         = 49.75 / 9.80665 = 5.07 kgf
"""

from __future__ import annotations

import math

import pytest

from bridgesim.checks import capacities, utilisation
from bridgesim.schema import Material, Prop, Section
from bridgesim.sections import euler_critical_load, rectangle
from bridgesim.units import n_to_kgf

E = 10000.0
W, T, L = 10.0, 2.0, 115.0
I_STICK = W * T**3 / 12.0
P_CR = math.pi**2 * E * I_STICK / L**2


@pytest.fixture(scope="module")
def mat_e10k(material: Material) -> Material:
    """The bundled material with E pinned to 10000 MPa (independent of the YAML value)."""
    return material.model_copy(update={"E_MPa": Prop(value=E)})


def test_single_stick_euler_load() -> None:
    """P_cr = pi^2 * 10000 * (10 * 2^3 / 12) / 115^2 = 49.75 N = 5.07 kgf."""
    p_cr = euler_critical_load(E, I_STICK, L)
    assert p_cr == pytest.approx(P_CR, rel=1e-12)
    assert p_cr == pytest.approx(49.75, abs=0.01)
    assert n_to_kgf(p_cr) == pytest.approx(5.07, abs=0.005)


def test_effective_length_factor() -> None:
    """P_cr scales with 1/K^2: K = 0.5 -> 4 P_cr = 199.0 N; K = 2 -> P_cr / 4 = 12.44 N."""
    assert euler_critical_load(E, I_STICK, L, K=0.5) == pytest.approx(4 * P_CR, rel=1e-12)
    assert euler_critical_load(E, I_STICK, L, K=2.0) == pytest.approx(P_CR / 4, rel=1e-12)


def test_euler_rejects_non_positive_length() -> None:
    """A column needs a positive length."""
    with pytest.raises(ValueError):
        euler_critical_load(E, I_STICK, 0.0)
    with pytest.raises(ValueError):
        euler_critical_load(E, I_STICK, -1.0)


@pytest.mark.parametrize("layout", ["flat", "on_edge"])
def test_capacities_single_stick(mat_e10k: Material, layout: str) -> None:
    """checks.capacities uses the weak axis: for one stick, flat or on edge,
    I_min = 10 * 2^3 / 12 = 6.667 mm^4 and P_cr = 49.75 N, far below crushing
    f_c A = 30 * 20 = 600 N, so N_c,R = P_cr.
    """
    sec = Section(id="stick", sticks=1, layout=layout)  # type: ignore[arg-type]
    assert rectangle(*sec.dims(mat_e10k.stick)).I_min_mm4 == pytest.approx(I_STICK)
    cap = capacities(sec, mat_e10k, L, 1.0)
    assert cap.P_cr == pytest.approx(P_CR, rel=1e-12)
    assert cap.N_c_crush == pytest.approx(mat_e10k.f_c_MPa.value * W * T)
    assert cap.N_c_R == pytest.approx(P_CR, rel=1e-12)
    assert n_to_kgf(cap.P_cr) == pytest.approx(5.07, abs=0.005)


def test_two_stick_stack_is_eight_times_stiffer(mat_e10k: Material) -> None:
    """Two sticks glued flat (b = 10, d = 4): I_min = 10 * 4^3 / 12 = 53.33 mm^4 = 8 I_stick
    (I grows with d^3), so P_cr = 8 * 49.75 = 398.0 N. Full composite action assumed.
    """
    sec = Section(id="two", sticks=2, layout="flat")
    cap = capacities(sec, mat_e10k, L, 1.0)
    assert cap.P_cr == pytest.approx(8 * P_CR, rel=1e-12)
    assert cap.P_cr == pytest.approx(398.0, abs=0.1)


def test_utilisation_at_euler_load_is_one(mat_e10k: Material) -> None:
    """A single stick loaded in compression by exactly P_cr has U = |N| / P_cr = 1,
    labelled as buckling (P_cr < f_c A). Tension of the same magnitude gives
    U = 49.75 / (f_t A = 40 * 20 = 800 N) = 0.0622.
    """
    sec = Section(id="stick", sticks=1, layout="flat")
    cap = capacities(sec, mat_e10k, L, 1.0)
    u = utilisation(cap, -P_CR, 0.0, 0.0, 0.0, None)
    assert math.isclose(u.U, 1.0, rel_tol=1e-12)
    assert u.mode == "buckling"
    ut = utilisation(cap, P_CR, 0.0, 0.0, 0.0, None)
    assert math.isclose(ut.U, P_CR / (mat_e10k.f_t_MPa.value * W * T), rel_tol=1e-12)
    assert ut.mode == "tension"


def test_short_stocky_member_crushes(mat_e10k: Material) -> None:
    """A 6-stick on-edge stack (b = 12, d = 10) only 20 mm long:

        I_min = min(b d^3, d b^3) / 12 = 12 * 10^3 / 12 = 1000 mm^4
        P_cr = pi^2 * 10000 * 1000 / 20^2 = 246740 N  >>  f_c A = 30 * 120 = 3600 N
    so crushing governs: N_c,R = 3600 N and the compression mode is "crushing".
    """
    sec = Section(id="pier", sticks=6, layout="on_edge")
    cap = capacities(sec, mat_e10k, 20.0, 1.0)
    assert cap.P_cr == pytest.approx(math.pi**2 * E * 1000.0 / 20.0**2, rel=1e-12)
    assert cap.N_c_R == pytest.approx(3600.0)
    assert utilisation(cap, -1800.0, 0, 0, 0, None).mode == "crushing"
