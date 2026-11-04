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
``tests/test_sim.py`` (test_dkt_patch_test) checks the element against plate theory.
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


IM = np.array([[6 * a + 0, 6 * a + 1] for a in range(3)]).ravel()  # membrane dofs
IB = np.array([[6 * a + 2, 6 * a + 3, 6 * a + 4] for a in range(3)]).ravel()  # bending


def operators(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Membrane strain operator Bm (3 x 6, dofs u, v per node), DKT curvature operators
    at the 3 integration points (3, 3, 9; dofs w, rx, ry per node) and the area."""
    Bm, area = cst_B(xy)
    Bb = np.stack([dkt_B(xy, s, e)[0] for s, e in _GP])
    return Bm, Bb, area


def drill_stiffness(E_t: float, area: float) -> float:
    return DRILL * E_t * area


def local_stiffness(xy: np.ndarray, A: np.ndarray, D: np.ndarray, E_t: float,
                    B: np.ndarray | None = None) -> np.ndarray:
    """18 x 18 stiffness in element axes, dofs per node (u, v, w, rx, ry, rz). ``B`` is
    the laminate's membrane-bending coupling (zero for symmetric layups)."""
    K = np.zeros((18, 18))
    Bm, Bbs, area = operators(xy)
    K[np.ix_(IM, IM)] += area * Bm.T @ A @ Bm
    Kb = np.zeros((9, 9))
    Kmb = np.zeros((6, 9))
    for Bb in Bbs:
        Kb += area / 3 * Bb.T @ D @ Bb
        if B is not None:
            Kmb += area / 3 * Bm.T @ B @ Bb
    K[np.ix_(IB, IB)] += Kb
    K[np.ix_(IM, IB)] += Kmb
    K[np.ix_(IB, IM)] += Kmb.T
    kd = drill_stiffness(E_t, area)
    idr = np.array([5, 11, 17])
    # drilling: each node's in-plane rotation is tied to the element's in-plane rigid
    # rotation omega = (v,x - u,y) / 2 (Hughes & Brezzi 1989), energy
    # kd/2 sum_a (theta_a - omega)^2, so no rotation about the normal is free
    g = drill_omega(Bm)
    K[np.ix_(idr, idr)] += kd * np.eye(3)
    K[np.ix_(idr, IM)] += -kd * np.outer(np.ones(3), g)
    K[np.ix_(IM, idr)] += -kd * np.outer(g, np.ones(3))
    K[np.ix_(IM, IM)] += 3 * kd * np.outer(g, g)
    return K


def drill_omega(Bm: np.ndarray) -> np.ndarray:
    """Row g with omega = g . (u1, v1, u2, v2, u3, v3), the in-plane rigid rotation of
    a constant-strain triangle."""
    g = np.zeros(6)
    for a in range(3):
        g[2 * a] = -0.5 * Bm[1, 2 * a + 1]  # -u,y / 2  (Bm[1, 2a+1] = c_a / 2A = N_a,y)
        g[2 * a + 1] = 0.5 * Bm[0, 2 * a]  # v,x / 2  (Bm[0, 2a] = b_a / 2A = N_a,x)
    return g


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


# --------------------------------------------------------------- layered, damaging shell
#
# The forces of a shell element are integrated ply by ply: at each of the 3 in-plane
# points and at 3 points through each ply group (Simpson, exact for the elastic bending
# stiffness), the strain is rotated into the grain axes and each failure mode keeps its
# own damage (Matzenmiller-type, as in LS-DYNA's MAT_054 or Abaqus' Hashin model):
#
#   fibre tension      linear softening, energy G_f per unit crack area
#   fibre compression  crushing plateau at (1 - crush_softening) f_c
#   across-grain tension (sticks parting edge to edge)  linear softening, G_f90
#   across-grain compression  plateau at f_c90
#   in-plane shear along the grain  linear softening, G_v
#
# Cracks close: a ply broken in tension still carries compression. The softening strain
# is set by the element's characteristic length (crack band, Bazant & Oh 1983), so the
# energy a crack dissipates does not depend on the mesh.

NMODE = 5  # history per point: fibre t, fibre c, across t, across c, shear


def element_data(xy: np.ndarray, angles: list[float], t: float, ply, Ef: float, Sf: float,
                 G: tuple[float, float, float], E_t_drill: float) -> dict:
    """Everything the layered kernel needs for one triangle with local in-plane corner
    coordinates ``xy`` and ply ``angles`` (deg, in element axes)."""
    from bridgesim.sim.materials import laminate, ply_layers

    lam = laminate(ply, angles, t)
    K = local_stiffness(xy, lam.A * Ef, lam.D * Ef, E_t_drill, lam.B * Ef)
    Bm, Bb, area = operators(xy)
    lay = [(zb, zt, np.cos(np.radians(th)), np.sin(np.radians(th)))
           for zb, zt, th in ply_layers(angles, t)]
    Q = ply.Q() * Ef
    lim = np.array([ply.f_t / ply.E_L, ply.f_c / ply.E_L, ply.f_t90 / ply.E_T,
                    ply.f_c90 / ply.E_T, ply.f_v / ply.G_LT]) * Sf
    lc = float(np.sqrt(2.0 * area))
    Em = np.array([ply.E_L, ply.E_T, ply.G_LT]) * Ef
    e0 = lim[[0, 2, 4]]
    rf_raw = 2.0 * np.asarray(G, float) / (Em * e0**2 * lc)
    rf = np.maximum(rf_raw, 1.05)  # element too large for the fracture energy: snap-back
    return {"K": K, "Bm": Bm, "Bb": Bb, "area": area,
            "kd": drill_stiffness(E_t_drill, area), "lay": lay,
            "Q": np.array([Q[0, 0], Q[1, 1], Q[0, 1], Q[2, 2]]), "lim": lim, "rf": rf,
            "lc": lc, "snapback": bool((rf_raw < 1.05).any()),
            "excess": float(np.max(1.05 / np.maximum(rf_raw, 1e-9))),
            "snap_modes": [k for k, r in zip(("fibre tension", "across the grain", "shear"),
                                             rf_raw, strict=True) if r < 1.05]}


def pack(elems: list[dict]) -> dict[str, np.ndarray]:
    """Stack element_data dicts into the Model's s_* arrays."""
    ns = len(elems)
    nl = max((len(e["lay"]) for e in elems), default=1)
    lay = np.zeros((ns, nl, 4))
    for k, e in enumerate(elems):
        lay[k, : len(e["lay"])] = e["lay"]

    def st(key, shape):
        return np.array([e[key] for e in elems], float).reshape(shape)

    return {"s_K": st("K", (ns, 18, 18)), "s_Bm": st("Bm", (ns, 3, 6)),
            "s_Bb": st("Bb", (ns, 3, 3, 9)), "s_area": st("area", (ns,)),
            "s_kd": st("kd", (ns,)), "s_lay": lay,
            "s_nl": np.array([len(e["lay"]) for e in elems], np.int64),
            "s_Q": st("Q", (ns, 4)), "s_lim": st("lim", (ns, 5)), "s_rf": st("rf", (ns, 3)),
            "s_lc": st("lc", (ns,))}


EMPTY = {"s_K": (0, 18, 18), "s_Bm": (0, 3, 6), "s_Bb": (0, 3, 3, 9), "s_area": (0,),
         "s_kd": (0,), "s_lay": (0, 1, 4), "s_Q": (0, 4), "s_lim": (0, 5), "s_rf": (0, 3),
         "s_lc": (0,)}


def empty() -> dict[str, np.ndarray]:
    out = {k: np.zeros(v) for k, v in EMPTY.items()}
    out["s_nl"] = np.zeros(0, np.int64)
    return out
