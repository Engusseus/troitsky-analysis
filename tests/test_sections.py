"""Rectangular section properties, torsion constant and stick-stack dimensions."""

from __future__ import annotations

import pytest

from bridgesim.schema import Section, Stick
from bridgesim.sections import rectangle, stick_stack_dims, torsion_constant_rect


def test_rectangle_properties() -> None:
    """b = 10 (local z), d = 20 (local y):

        A  = b d        = 200 mm^2
        Iz = b d^3 / 12 = 10 * 8000 / 12 = 6666.67 mm^4   (primary plane, about local z)
        Iy = d b^3 / 12 = 20 * 1000 / 12 = 1666.67 mm^4
        Sz = b d^2 / 6  = 10 * 400 / 6   = 666.67 mm^3
        Sy = d b^2 / 6  = 20 * 100 / 6   = 333.33 mm^3
    """
    pr = rectangle(10.0, 20.0)
    assert (pr.b_mm, pr.d_mm) == (10.0, 20.0)
    assert pr.A_mm2 == pytest.approx(200.0)
    assert pr.Iz_mm4 == pytest.approx(20000.0 / 3.0)
    assert pr.Iy_mm4 == pytest.approx(5000.0 / 3.0)
    assert pr.Sz_mm3 == pytest.approx(2000.0 / 3.0)
    assert pr.Sy_mm3 == pytest.approx(1000.0 / 3.0)
    assert pr.I_min_mm4 == pytest.approx(pr.Iy_mm4)


def test_rectangle_axes_swap() -> None:
    """Swapping b and d swaps the y and z properties; A and J are unchanged."""
    a, b = rectangle(10.0, 20.0), rectangle(20.0, 10.0)
    assert a.Iy_mm4 == pytest.approx(b.Iz_mm4)
    assert a.Iz_mm4 == pytest.approx(b.Iy_mm4)
    assert a.Sy_mm3 == pytest.approx(b.Sz_mm3)
    assert a.Sz_mm3 == pytest.approx(b.Sy_mm3)
    assert a.A_mm2 == pytest.approx(b.A_mm2)
    assert a.J_mm4 == pytest.approx(b.J_mm4)


def test_torsion_constant_square() -> None:
    """Square a = c: beta = 1/3 - 0.21 (1)(1 - 1/12) = 0.140833, J = beta a^4.

    Exact Saint-Venant value for a square: beta = 0.1406 (0.140577), i.e. within 0.2 %.
    """
    a = 10.0
    beta = torsion_constant_rect(a, a) / a**4
    assert beta == pytest.approx(1 / 3 - 0.21 * (1 - 1 / 12), rel=1e-12)
    assert beta == pytest.approx(0.140833, abs=1e-6)
    assert beta == pytest.approx(0.140577, rel=2e-3)


def test_torsion_constant_two_to_one() -> None:
    """a = 20, c = 10: beta = 1/3 - 0.21 (0.5)(1 - 0.5^4/12) = 0.228880,
    J = beta a c^3 = 0.228880 * 20 * 1000 = 4577.6 mm^4 (exact 2:1 value beta = 0.2287).
    """
    j_rect = torsion_constant_rect(20.0, 10.0)
    assert j_rect == pytest.approx(4577.6, abs=0.1)
    assert j_rect / (20.0 * 10.0**3) == pytest.approx(0.2287, rel=2e-3)
    assert rectangle(10.0, 20.0).J_mm4 == pytest.approx(j_rect)


def test_torsion_constant_thin_strip_tends_to_one_third() -> None:
    """Thin strip c/a -> 0: beta -> 1/3. At a/c = 1000, beta = 1/3 - 0.21e-3 = 0.33312."""
    a, c = 1000.0, 1.0
    beta = torsion_constant_rect(a, c) / (a * c**3)
    assert beta == pytest.approx(1 / 3, rel=1e-3)
    assert beta < 1 / 3


def test_torsion_constant_argument_order_irrelevant() -> None:
    """J(a, c) = J(c, a): the long side is picked internally."""
    assert torsion_constant_rect(2.0, 10.0) == pytest.approx(torsion_constant_rect(10.0, 2.0))


@pytest.mark.parametrize(
    ("n", "layout", "expected"),
    [
        (1, "flat", (10.0, 2.0)),
        (3, "flat", (10.0, 6.0)),
        (18, "flat", (10.0, 36.0)),
        (1, "on_edge", (2.0, 10.0)),
        (4, "on_edge", (8.0, 10.0)),
        (6, "on_edge", (12.0, 10.0)),
    ],
)
def test_stick_stack_dims(n: int, layout: str, expected: tuple[float, float]) -> None:
    """flat: b = w, d = n t; on_edge: b = n t, d = w (w = 10, t = 2)."""
    assert stick_stack_dims(n, layout, 10.0, 2.0) == pytest.approx(expected)
    sec = Section(id="s", sticks=n, layout=layout)  # type: ignore[arg-type]
    assert sec.dims(Stick()) == pytest.approx(expected)


def test_section_explicit_rectangle_and_label() -> None:
    """An explicit {b_mm, d_mm} section returns its own dims; labels describe the section."""
    sec = Section(id="r", b_mm=7.5, d_mm=12.0)
    assert sec.dims(Stick()) == (7.5, 12.0)
    assert sec.props(Stick()).A_mm2 == pytest.approx(90.0)
    assert sec.label() == "7.5 x 12 mm"
    assert Section(id="s", sticks=3, layout="flat").label() == "3 sticks flat"


@pytest.mark.parametrize(("b", "d"), [(0.0, 10.0), (10.0, 0.0), (-1.0, 10.0), (10.0, -2.0)])
def test_rectangle_rejects_non_positive(b: float, d: float) -> None:
    """Non-positive dimensions are invalid."""
    with pytest.raises(ValueError):
        rectangle(b, d)


def test_torsion_rejects_non_positive() -> None:
    """J needs both sides positive."""
    with pytest.raises(ValueError):
        torsion_constant_rect(10.0, 0.0)


def test_stick_stack_rejects_bad_input() -> None:
    """At least one stick, and only the 'flat' / 'on_edge' layouts exist."""
    with pytest.raises(ValueError):
        stick_stack_dims(0, "flat", 10.0, 2.0)
    with pytest.raises(ValueError):
        stick_stack_dims(2, "diagonal", 10.0, 2.0)
