"""Unit conventions and conversions.

Internal units are mm, N and MPa (= N/mm^2). Mass is in kg and density in kg/m^3.
Loads are displayed in kgf as well as N.
"""

from __future__ import annotations

#: Standard gravity, N per kgf (exact by definition).
N_PER_KGF = 9.80665

#: mm^3 per m^3.
MM3_PER_M3 = 1.0e9


def n_to_kgf(force_N: float) -> float:
    """Convert newtons to kilogram-force: F[kgf] = F[N] / 9.80665."""
    return force_N / N_PER_KGF


def kgf_to_n(force_kgf: float) -> float:
    """Convert kilogram-force to newtons: F[N] = F[kgf] * 9.80665."""
    return force_kgf * N_PER_KGF


def volume_mm3_to_mass_kg(volume_mm3: float, density_kg_m3: float) -> float:
    """Mass from volume and density: m = rho * V, with V converted from mm^3 to m^3."""
    return density_kg_m3 * volume_mm3 / MM3_PER_M3
