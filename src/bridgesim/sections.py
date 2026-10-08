"""Rectangular cross-section properties for laminated popsicle-stick members.

Axis convention (matches Pynite 3.2.0 with member rotation = 0):

* ``d`` (depth) lies along the member's local y axis. This is the member's *primary
  bending plane*: vertical for horizontal members, in the truss plane for members of a
  vertical truss.
* ``b`` (width) lies along the local z axis.
* ``Iz`` (bending about local z, i.e. in the primary plane) = b d^3 / 12.
* ``Iy`` (bending about local y, out of the primary plane) = d b^3 / 12.

A stick-stack section is ``n`` identical sticks glued face to face:

* ``flat``: the 10 mm stick width is perpendicular to the primary plane, so the stack
  grows in depth: b = w, d = n t.
* ``on_edge``: the 10 mm stick width lies in the primary plane, so the stack grows in
  width: b = n t, d = w.

where w = stick width (10 mm) and t = stick thickness (2 mm). Gluing is assumed perfect
(full composite action), which is optimistic; see ASSUMPTIONS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RectProps:
    """Properties of a b x d rectangle. Lengths in mm."""

    b_mm: float
    d_mm: float
    A_mm2: float
    Iy_mm4: float
    Iz_mm4: float
    J_mm4: float
    Sy_mm3: float
    Sz_mm3: float

    @property
    def I_min_mm4(self) -> float:
        """Smaller principal second moment of area, used for Euler buckling."""
        return min(self.Iy_mm4, self.Iz_mm4)


def torsion_constant_rect(a_mm: float, c_mm: float) -> float:
    """Saint-Venant torsion constant of a solid rectangle.

    J = beta * a * c^3 with a >= c (long side a, short side c) and

        beta = 1/3 - 0.21 (c/a) (1 - c^4 / (12 a^4))

    Source: Roark's Formulas for Stress and Strain, 7th ed., Table 10.7 case 4
    (Young & Budynas). Accurate to about 4% for all aspect ratios.
    """
    a, c = max(a_mm, c_mm), min(a_mm, c_mm)
    if c <= 0:
        raise ValueError("Rectangle dimensions must be positive")
    beta = 1.0 / 3.0 - 0.21 * (c / a) * (1.0 - c**4 / (12.0 * a**4))
    return beta * a * c**3


def rectangle(b_mm: float, d_mm: float) -> RectProps:
    """Properties of a solid b x d rectangle (d along local y, b along local z).

    A = b d,  Iz = b d^3 / 12,  Iy = d b^3 / 12,  Sz = b d^2 / 6,  Sy = d b^2 / 6,
    J from :func:`torsion_constant_rect`. Source: any mechanics-of-materials text,
    e.g. Hibbeler, *Mechanics of Materials*, Appendix A.
    """
    if b_mm <= 0 or d_mm <= 0:
        raise ValueError(f"Section dimensions must be positive (b={b_mm}, d={d_mm})")
    return RectProps(
        b_mm=b_mm,
        d_mm=d_mm,
        A_mm2=b_mm * d_mm,
        Iy_mm4=d_mm * b_mm**3 / 12.0,
        Iz_mm4=b_mm * d_mm**3 / 12.0,
        J_mm4=torsion_constant_rect(b_mm, d_mm),
        Sy_mm3=d_mm * b_mm**2 / 6.0,
        Sz_mm3=b_mm * d_mm**2 / 6.0,
    )


def stick_stack_dims(
    n_sticks: int, layout: str, stick_width_mm: float, stick_thickness_mm: float
) -> tuple[float, float]:
    """Return (b, d) in mm for ``n_sticks`` sticks glued face to face.

    ``flat``: b = w, d = n t.  ``on_edge``: b = n t, d = w.
    """
    if n_sticks < 1:
        raise ValueError("A stick stack needs at least one stick")
    if layout == "flat":
        return stick_width_mm, n_sticks * stick_thickness_mm
    if layout == "on_edge":
        return n_sticks * stick_thickness_mm, stick_width_mm
    raise ValueError(f"Unknown stick layout {layout!r}; use 'flat' or 'on_edge'")


def euler_critical_load(E_MPa: float, I_mm4: float, L_mm: float, K: float = 1.0) -> float:
    """Euler buckling load of a pin-ended column, in N.

    P_cr = pi^2 E I / (K L)^2. Source: Euler (1744); any stability text, e.g.
    Timoshenko & Gere, *Theory of Elastic Stability*, Ch. 2.
    """
    if L_mm <= 0:
        raise ValueError("Column length must be positive")
    return math.pi**2 * E_MPa * I_mm4 / (K * L_mm) ** 2
