"""Flat triangular shell element: constant-strain membrane + DKT plate bending.

Local stiffness (18 x 18) of a triangle in its own plane, with 6 degrees of freedom per
node in the element axes (u, v, w, rx, ry, rz):

* membrane: constant-strain triangle (CST) with the laminate's in-plane stiffness A
  (N/mm, from classical lamination theory, so orthotropic sticks are handled);
* bending: Discrete Kirchhoff Triangle (Batoz, Bathe & Ho, *Int. J. Numer. Meth. Eng.*
  15, 1980), 3-point integration, with the laminate's bending stiffness D (N mm). Its
  rotations are theta_x = w,y = rx and theta_y = -w,x = ry;
* drilling (rz): a small stiffness so a node shared with a beam is not a mechanism.

The explicit solver (:mod:`bridgesim.sim.solver`) uses this stiffness in a corotational
frame that follows the element, so large rotations are exact while strains stay small.
``tests/test_sim_validation.py`` checks the element against plate theory.
"""

from __future__ import annotations

import numpy as np

DRILL = 0.05  # drilling stiffness, as a fraction of E t A (keeps a node's in-plane rotation
#              tied to its element once the glue around it has gone)


def cst_B(xy: np.ndarray) -> tuple[np.ndarray, float]:
    (x1, y1), (x2, y2), (x3, y3) = xy
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    B = np.zeros((3, 6))
    B[0, 0::2] = b
    B[1, 1::2] = c
    B[2, 0::2] = c
    B[2, 1::2] = b
    return B / (2 * area), area


def dkt_B(xy: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """Curvature-displacement matrix (3 x 9) of the DKT element at area coordinates
    (xi, eta); dof order per node (w, theta_x, theta_y)."""
    x = xy[:, 0]
    y = xy[:, 1]
    # sides k = 4, 5, 6 are 23, 31, 12
    xij = np.array([x[1] - x[2], x[2] - x[0], x[0] - x[1]])
    yij = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    l2 = xij**2 + yij**2
    P = -6 * xij / l2
    t = -6 * yij / l2
    q = 3 * xij * yij / l2
    r = 3 * yij**2 / l2
    P4, P5, P6 = P
    t4, t5, t6 = t
    q4, q5, q6 = q
    r4, r5, r6 = r
    s, e = xi, eta
    Hx_xi = np.array([
        P6 * (1 - 2 * s) + (P5 - P6) * e, q6 * (1 - 2 * s) - (q5 + q6) * e,
        -4 + 6 * (s + e) + r6 * (1 - 2 * s) - e * (r5 + r6),
        -P6 * (1 - 2 * s) + e * (P4 + P6), q6 * (1 - 2 * s) - e * (q6 - q4),
        -2 + 6 * s + r6 * (1 - 2 * s) + e * (r4 - r6),
        -e * (P5 + P4), e * (q4 - q5), -e * (r5 - r4)])
    Hy_xi = np.array([
        t6 * (1 - 2 * s) + e * (t5 - t6), 1 + r6 * (1 - 2 * s) - e * (r5 + r6),
        -q6 * (1 - 2 * s) + e * (q5 + q6),
        -t6 * (1 - 2 * s) + e * (t4 + t6), -1 + r6 * (1 - 2 * s) + e * (r4 - r6),
        -q6 * (1 - 2 * s) - e * (q4 - q6),
        -e * (t4 + t5), e * (r4 - r5), -e * (q4 - q5)])
    Hx_eta = np.array([
        -P5 * (1 - 2 * e) - s * (P6 - P5), q5 * (1 - 2 * e) - s * (q5 + q6),
        -4 + 6 * (s + e) + r5 * (1 - 2 * e) - s * (r5 + r6),
        s * (P4 + P6), s * (q4 - q6), -s * (r6 - r4),
        P5 * (1 - 2 * e) - s * (P4 + P5), q5 * (1 - 2 * e) + s * (q4 - q5),
        -2 + 6 * e + r5 * (1 - 2 * e) + s * (r4 - r5)])
    Hy_eta = np.array([
        -t5 * (1 - 2 * e) - s * (t6 - t5), 1 + r5 * (1 - 2 * e) - s * (r5 + r6),
        -q5 * (1 - 2 * e) + s * (q5 + q6),
        s * (t4 + t6), s * (r4 - r6), -s * (q4 - q6),
        t5 * (1 - 2 * e) - s * (t4 + t5), -1 + r5 * (1 - 2 * e) + s * (r4 - r5),
        -q5 * (1 - 2 * e) - s * (q4 - q5)])
    x31, x12 = x[2] - x[0], x[0] - x[1]
    y31, y12 = y[2] - y[0], y[0] - y[1]
    area2 = x31 * (-y12) - (-x12) * y31  # 2 * area = x21*y31 - x31*y21
    area2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    B = np.vstack([
        y31 * Hx_xi + y12 * Hx_eta,
        -x31 * Hy_xi - x12 * Hy_eta,
        -x31 * Hx_xi - x12 * Hx_eta + y31 * Hy_xi + y12 * Hy_eta,
    ]) / area2
    return B, area2 / 2


_GP = [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]  # mid-side points, weight 1/3 each


def local_stiffness(xy: np.ndarray, A: np.ndarray, D: np.ndarray, E_t: float
                    ) -> np.ndarray:
    """18 x 18 stiffness in element axes, dofs per node (u, v, w, rx, ry, rz)."""
    K = np.zeros((18, 18))
    Bm, area = cst_B(xy)
    Km = area * Bm.T @ A @ Bm
    im = np.array([[6 * a + 0, 6 * a + 1] for a in range(3)]).ravel()
    K[np.ix_(im, im)] += Km
    Kb = np.zeros((9, 9))
    for s, e in _GP:
        Bb, ar = dkt_B(xy, s, e)
        Kb += ar / 3 * Bb.T @ D @ Bb
    ib = np.array([[6 * a + 2, 6 * a + 3, 6 * a + 4] for a in range(3)]).ravel()
    K[np.ix_(ib, ib)] += Kb
    kd = DRILL * E_t * area
    idr = np.array([5, 11, 17])
    K[np.ix_(idr, idr)] += kd * (np.eye(3) - np.full((3, 3), 1 / 3))
    return K


def recovery(xy: np.ndarray) -> np.ndarray:
    """6 x 18 matrix giving (membrane strains ex, ey, gxy; curvatures kx, ky, kxy) at the
    centroid from the 18 local dofs."""
    Bm, _ = cst_B(xy)
    Bb, _ = dkt_B(xy, 1 / 3, 1 / 3)
    R = np.zeros((6, 18))
    im = np.array([[6 * a + 0, 6 * a + 1] for a in range(3)]).ravel()
    ib = np.array([[6 * a + 2, 6 * a + 3, 6 * a + 4] for a in range(3)]).ravel()
    R[:3, im] = Bm
    R[3:, ib] = Bb
    return R
