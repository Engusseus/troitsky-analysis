"""Member local axes and cross-section extents, matching Pynite 3.2.0 (rotation = 0).

Pynite's convention (Member3D.T): local x runs i -> j.
* vertical members: local y = -X (going up) or +X (going down), local z = +Z
* horizontal members: local y = +Y, local z = x cross y
* inclined members: local z is horizontal (perpendicular to x and its plan projection),
  local y = z cross x (always has an upward component)

The section depth d lies along local y and the width b along local z.
"""

from __future__ import annotations

import math

import numpy as np

_TOL = 1e-9


def local_axes(pi: tuple[float, float, float], pj: tuple[float, float, float]) -> np.ndarray:
    """3x3 array whose rows are the unit local x, y, z axes in global coordinates."""
    a, b = np.asarray(pi, float), np.asarray(pj, float)
    v = b - a
    L = float(np.linalg.norm(v))
    if L == 0:
        raise ValueError("Zero-length member")
    x = v / L
    if math.isclose(a[0], b[0], abs_tol=_TOL) and math.isclose(a[2], b[2], abs_tol=_TOL):
        y = np.array([-1.0, 0, 0]) if b[1] > a[1] else np.array([1.0, 0, 0])
        z = np.array([0, 0, 1.0])
    elif math.isclose(a[1], b[1], abs_tol=_TOL):
        y = np.array([0, 1.0, 0])
        z = np.cross(x, y)
        z /= np.linalg.norm(z)
    else:
        proj = np.array([v[0], 0.0, v[2]])
        z = np.cross(proj, x) if b[1] > a[1] else np.cross(x, proj)
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
    return np.vstack([x, y, z])


def half_extents(
    pi: tuple[float, float, float], pj: tuple[float, float, float], b_mm: float, d_mm: float
) -> np.ndarray:
    """Half-size of the member cross-section along global X, Y, Z (mm).

    h_e = |y_local . e| d/2 + |z_local . e| b/2 for each global axis e.
    """
    ax = local_axes(pi, pj)
    return np.abs(ax[1]) * d_mm / 2.0 + np.abs(ax[2]) * b_mm / 2.0


def inclination_from_vertical_deg(
    pi: tuple[float, float, float], pj: tuple[float, float, float]
) -> float:
    v = np.asarray(pj, float) - np.asarray(pi, float)
    L = float(np.linalg.norm(v))
    return math.degrees(math.acos(min(1.0, abs(v[1]) / L)))
