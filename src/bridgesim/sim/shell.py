"""Flat triangular shell element: constant-strain membrane + DKMT (or DKT) plate bending.

Local stiffness (18 x 18) of a triangle in its own plane, with 6 degrees of freedom per
node in the element axes (u, v, w, rx, ry, rz):

* membrane: constant-strain triangle (CST) with the laminate's in-plane stiffness A
  (N/mm, from classical lamination theory, so orthotropic sticks are handled);
* bending: Discrete Kirchhoff Triangle (Batoz, Bathe & Ho, *Int. J. Numer. Meth. Eng.*
  15, 1980), 3-point integration, with the laminate's bending stiffness D (N mm). Its
  rotations are theta_x = w,y = rx and theta_y = -w,x = ry; or, shear-deformable
  (since round 28), the Discrete Kirchhoff-Mindlin Triangle (Katili 1993, *Int. J.
  Numer. Meth. Eng.* 36:1859), which adds the laminate's transverse shear stiffness H
  (N/mm) and tends to DKT as the plate gets thin (below);
* drilling (rz): a small stiffness so a node shared with a beam is not a mechanism.

The explicit solver (:mod:`bridgesim.sim.solver`) uses this stiffness in a corotational
frame that follows the element, so large rotations are exact while strains stay small.
``tests/test_sim.py`` (test_dkt_patch_test, test_dkmt_*) checks the element against
plate theory.

DKMT. The rotations of the normal, beta = (beta_x, beta_y) = (theta_y, -theta_x) at the
nodes, are linear plus a quadratic bubble of the tangential rotation on each side k
(nodes i -> j, length L_k, tangent t_k), 4 lambda_i lambda_j t_k dbeta_k. Along each side
the transverse shear strain gamma_s is constant and the Mindlin constraint is held in the
mean, int (w,s + beta_s - gamma_s) ds = 0:

    gamma_s,k = r_k + (2/3) dbeta_k,   r_k = (w_j - w_i) / L_k + (beta_s,i + beta_s,j) / 2

and each side is a Timoshenko beam (Katili's DKMT/DKMQ: the shear along the side from the
moment gradient along it, M_s = D_ss beta_s,s): gamma_s,k = -(2/3) phi_k dbeta_k with

    phi_k = 12 D_ss,k / (H_ss,k L_k^2),

D_ss,k the bending stiffness along the side (D turned into the side's axes: inside the
element the membrane strain is constant, so the moment gradient is D times the curvature
gradient, an unsymmetric layup's B included) and 1 / H_ss,k = t_k . H^-1 t_k its shear
compliance. So

    dbeta_k = -(3/2) r_k / (1 + phi_k),   gamma_s,k = phi_k / (1 + phi_k) r_k:

phi -> 0 (thin) is DKT (dbeta_k = -(3/2) r_k, gamma = 0), phi -> oo (thick, small
elements) a linear element whose sides shear by their whole Kirchhoff residual; there is
no shear locking. The element's transverse shear strain is the mean of the lowest-order
edge (Raviart-Thomas, "rotated") field through the three side values, constant:
gamma = sum_k gamma_s,k L_k (grad lambda_j - grad lambda_i) / 3, exact for a constant
field. Bending energy at the 3 mid-side points as DKT, shear energy area x gamma . H gamma.
Written as a correction to DKT, the curvature at point g and the shear strain are

    kappa_g = B_dkt,g u + (3/2) KH_g om (R u),   gamma = Gs om (R u),   om = phi / (1 + phi)

with R (3 x 9) the side residuals r_k, KH_g the curvature of a unit side bubble at g and Gs
the side-to-element shear map. A plate split between its plies bends with D x s^2 (s: 1,
1/2, ... 1/n_plies, the lever arms of its sub-laminates), which is phi x s^2.
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


# sides k = 4, 5, 6 of Batoz's DKT: nodes 2 -> 3, 3 -> 1, 1 -> 2
SIDES = ((1, 2), (2, 0), (0, 1))


def dkmt_parts(xy: np.ndarray) -> dict[str, np.ndarray]:
    """The geometric operators of the DKMT correction (module docstring), dofs per node
    (w, theta_x, theta_y): ``R`` (3 x 9) the side residuals r_k = (w_j - w_i) / L_k +
    (beta_s,i + beta_s,j) / 2 (beta_s = c beta_x + s beta_y = c theta_y - s theta_x);
    ``KH`` (3 points, 3 curvatures, 3 sides) the curvature of a unit tangential bubble
    4 lambda_i lambda_j t_k at the mid-side points; ``Gs`` (2 x 3) the element's constant
    shear strain from the three sides' (the mean of the Raviart-Thomas field); ``t``
    (3 x 2) the side tangents and ``L`` (3,) their lengths."""
    x, y = xy[:, 0], xy[:, 1]
    a2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    gx = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / a2  # d lambda_a / dx
    gy = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / a2  # d lambda_a / dy
    R = np.zeros((3, 9))
    KH = np.zeros((3, 3, 3))
    Gs = np.zeros((2, 3))
    tk = np.zeros((3, 2))
    Lk = np.zeros(3)
    for k, (i, j) in enumerate(SIDES):
        d = xy[j] - xy[i]
        Lk[k] = float(np.hypot(d[0], d[1]))
        c, s = d / Lk[k]
        tk[k] = c, s
        R[k, 3 * j] += 1.0 / Lk[k]
        R[k, 3 * i] -= 1.0 / Lk[k]
        for a in (i, j):
            R[k, 3 * a + 1] -= 0.5 * s
            R[k, 3 * a + 2] += 0.5 * c
        for g, (xi, eta) in enumerate(_GP):
            lam = (1.0 - xi - eta, xi, eta)
            px = 4.0 * (lam[j] * gx[i] + lam[i] * gx[j])  # gradient of the bubble
            py = 4.0 * (lam[j] * gy[i] + lam[i] * gy[j])
            KH[g, :, k] = (c * px, s * py, c * py + s * px)
        Gs[:, k] = Lk[k] / 3.0 * np.array([gx[j] - gx[i], gy[j] - gy[i]])
    return {"R": R, "KH": KH, "Gs": Gs, "t": tk, "L": Lk}


def side_phi(D: np.ndarray, H: np.ndarray, tk: np.ndarray, Lk: np.ndarray) -> np.ndarray:
    """phi_k = 12 D_ss,k / (H_ss,k L_k^2) of each side (module docstring): D_ss the
    bending stiffness for curvature along the side (D turned into the side's axes:
    D11 c^4 + 2 (D12 + 2 D66) c^2 s^2 + D22 s^4 + 4 D16 c^3 s + 4 D26 c s^3), 1 / H_ss
    the shear compliance along it, t . H^-1 t."""
    Hi = np.linalg.inv(H)
    out = np.zeros(3)
    for k in range(3):
        c, s = tk[k]
        dss = (D[0, 0] * c**4 + 2.0 * (D[0, 1] + 2.0 * D[2, 2]) * c * c * s * s
               + D[1, 1] * s**4 + 4.0 * D[0, 2] * c**3 * s + 4.0 * D[1, 2] * c * s**3)
        out[k] = 12.0 * dss * float(tk[k] @ Hi @ tk[k]) / Lk[k] ** 2
    return out


def dkmt_operators(xy: np.ndarray, phi: np.ndarray, parts: dict | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """DKMT curvature operators at the 3 mid-side points (3, 3, 9) and the constant
    shear strain operator (2, 9), dofs per node (w, theta_x, theta_y), for the sides'
    ``phi`` (all zero: DKT, no shear strain)."""
    p = parts or dkmt_parts(xy)
    om = phi / (1.0 + phi)
    Bb = np.stack([dkt_B(xy, s, e)[0] for s, e in _GP])
    Bb = Bb + 1.5 * np.einsum("gik,k,kj->gij", p["KH"], om, p["R"])
    Bs = p["Gs"] @ (om[:, None] * p["R"])
    return Bb, Bs


def local_stiffness(xy: np.ndarray, A: np.ndarray, D: np.ndarray, E_t: float,
                    B: np.ndarray | None = None, H: np.ndarray | None = None
                    ) -> np.ndarray:
    """18 x 18 stiffness in element axes, dofs per node (u, v, w, rx, ry, rz). ``B`` is
    the laminate's membrane-bending coupling (zero for symmetric layups); ``H`` (2 x 2)
    its transverse shear stiffness: given, the bending is DKMT (shear-deformable), else
    DKT."""
    K = np.zeros((18, 18))
    Bm, Bbs, area = operators(xy)
    Bs = None
    if H is not None:
        p = dkmt_parts(xy)
        Bbs, Bs = dkmt_operators(xy, side_phi(D, H, p["t"], p["L"]), p)
    K[np.ix_(IM, IM)] += area * Bm.T @ A @ Bm
    Kb = np.zeros((9, 9))
    Kmb = np.zeros((6, 9))
    for Bb in Bbs:
        Kb += area / 3 * Bb.T @ D @ Bb
        if B is not None:
            Kmb += area / 3 * Bm.T @ B @ Bb
    if Bs is not None:
        Kb += area * Bs.T @ H @ Bs
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


def recovery(xy: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
    """6 x 18 matrix giving (membrane strains ex, ey, gxy; curvatures kx, ky, kxy) at the
    centroid from the 18 local dofs; ``phi``: the sides' DKMT phi of a shear-deformable
    element (its curvature is linear: the centroid's is the mean of the mid-side
    points')."""
    Bm, _ = cst_B(xy)
    Bb, _ = dkt_B(xy, 1 / 3, 1 / 3)
    if phi is not None and np.any(phi):
        Bb = dkmt_operators(xy, np.asarray(phi, float))[0].mean(axis=0)
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
                 G: tuple[float, float, float], E_t_drill: float, kt: float = 1.0,
                 kc: float = 1.0, H: np.ndarray | None = None,
                 interlaminar: tuple[float, float, float] | None = None) -> dict:
    """Everything the layered kernel needs for one triangle with local in-plane corner
    coordinates ``xy`` and ply ``angles`` (deg, in element axes); ``kt`` scales the fibre
    tension strength alone (the butt joints of a plate laid from sticks). ``H``: the
    laminate's transverse shear stiffness in element axes (N/mm, scatter included;
    materials.laminate_shear): the element is then shear-deformable (DKMT), else DKT.
    ``interlaminar``: the strengths of its glue planes (MPa: f_v along a ply's grain,
    rolling shear across it, the bond line) for the split check of a laminate of 2 or
    more plies (None: it cannot split)."""
    from bridgesim.sim.materials import laminate, ply_layers

    lam = laminate(ply, angles, t)
    K = local_stiffness(xy, lam.A * Ef, lam.D * Ef, E_t_drill, lam.B * Ef, H)
    shear = {}
    if H is not None:
        p = dkmt_parts(xy)
        phi = side_phi(lam.D * Ef, H, p["t"], p["L"])
        fil = np.zeros(3)
        if interlaminar is not None and len(angles) >= 2:
            # 1.5 / (t f): the interlaminar stress 1.5 |Q| / t over each strength
            fil = np.array([1.5 / (t * max(f, 1e-9)) for f in interlaminar])
        shear = {"R": p["R"], "KH": p["KH"], "Gs": p["Gs"], "phi": phi,
                 "H": np.array([H[0, 0], H[1, 1], 0.5 * (H[0, 1] + H[1, 0])]), "fil": fil,
                 "nply": float(max(1, len(angles)))}
    Bm, Bb, area = operators(xy)
    lay = [(zb, zt, np.cos(np.radians(th)), np.sin(np.radians(th)))
           for zb, zt, th in ply_layers(angles, t)]
    Q = ply.Q() * Ef
    lim = np.array([ply.f_t / ply.E_L, ply.f_c / ply.E_L, ply.f_t90 / ply.E_T,
                    ply.f_c90 / ply.E_T, ply.f_v / ply.G_LT]) * Sf
    lim[0] *= kt
    lim[1] *= kc  # butt joints that do not bear (gapped) in compression
    lc = float(np.sqrt(2.0 * area))
    Em = np.array([ply.E_L, ply.E_T, ply.G_LT]) * Ef
    e0 = lim[[0, 2, 4]]
    rf_raw = 2.0 * np.asarray(G, float) / (Em * e0**2 * lc)
    # an element larger than the crack band its fracture energy allows (snap-back): its
    # strength in that mode is lowered until its steepest softening dissipates exactly
    # G_f (Bazant & Oh 1983; for cohesive elements Turon et al. 2007): f' = f
    # sqrt(rf / 1.05), the stress a crack as long as the element needs to release G_f.
    # Keeping the strength (until round 16) dissipated up to several times G_f, a
    # coarse plate tougher than a fine one
    red = np.where(rf_raw < 1.05, np.sqrt(np.maximum(rf_raw, 1e-12) / 1.05), 1.0)
    lim[[0, 2, 4]] *= red
    rf = np.maximum(rf_raw / red**2, 1.05)
    return {"K": K, "Bm": Bm, "Bb": Bb, "area": area,
            "kd": drill_stiffness(E_t_drill, area), "lay": lay,
            "Q": np.array([Q[0, 0], Q[1, 1], Q[0, 1], Q[2, 2]]), "lim": lim, "rf": rf,
            "lc": lc, "snapback": bool((rf_raw < 1.05).any()),
            "snap3": (rf_raw < 1.05),
            "excess": float(np.min(red)),
            "snap_modes": [k for k, r in zip(("fibre tension", "across the grain", "shear"),
                                             rf_raw, strict=True) if r < 1.05],
            **shear}


# the DKMT arrays of an element (zeros for a DKT one: s_H[:, 0] == 0 marks it)
SHEAR_SHAPES = {"R": (3, 9), "KH": (3, 3, 3), "Gs": (2, 3), "phi": (3,), "H": (3,),
                "fil": (3,), "nply": ()}


def pack(elems: list[dict]) -> dict[str, np.ndarray]:
    """Stack element_data dicts into the Model's s_* arrays."""
    ns = len(elems)
    nl = max((len(e["lay"]) for e in elems), default=1)
    lay = np.zeros((ns, nl, 4))
    for k, e in enumerate(elems):
        lay[k, : len(e["lay"])] = e["lay"]

    def st(key, shape):
        return np.array([e[key] for e in elems], float).reshape(shape)

    def sh(key):
        shape = SHEAR_SHAPES[key]
        return np.array([e.get(key, np.zeros(shape)) for e in elems],
                        float).reshape((ns, *shape))

    return {"s_K": st("K", (ns, 18, 18)), "s_Bm": st("Bm", (ns, 3, 6)),
            "s_Bb": st("Bb", (ns, 3, 3, 9)), "s_area": st("area", (ns,)),
            "s_kd": st("kd", (ns,)), "s_lay": lay,
            "s_nl": np.array([len(e["lay"]) for e in elems], np.int64),
            "s_Q": st("Q", (ns, 4)), "s_lim": st("lim", (ns, 5)), "s_rf": st("rf", (ns, 3)),
            "s_lc": st("lc", (ns,)),
            **{f"s_{k}": sh(k) for k in SHEAR_SHAPES}}


EMPTY = {"s_K": (0, 18, 18), "s_Bm": (0, 3, 6), "s_Bb": (0, 3, 3, 9), "s_area": (0,),
         "s_kd": (0,), "s_lay": (0, 1, 4), "s_Q": (0, 4), "s_lim": (0, 5), "s_rf": (0, 3),
         "s_lc": (0,), **{f"s_{k}": (0, *v) for k, v in SHEAR_SHAPES.items()}}


def empty() -> dict[str, np.ndarray]:
    out = {k: np.zeros(v) for k, v in EMPTY.items()}
    out["s_nl"] = np.zeros(0, np.int64)
    return out
