"""Explicit nonlinear dynamics of the bridge (central-difference time integration).

Every time step computes the internal forces of the deformed structure directly; there is
no stiffness matrix to factor, which is what lets the bridge crack, lose joints, buckle,
slide on the platform and collapse without convergence failures (the approach of
LS-DYNA and Abaqus/Explicit). Ingredients:

* **beams**: corotational 3D elements (the element frame follows the deformed chord, so
  rotations of any size are exact) with *fibre sections*: each section is a grid of
  wood fibres with their own stress-strain law, integrated at two Gauss points. A fibre is
  linear elastic, softens to zero after reaching f_t in tension (fracture energy G_f,
  regularised by element length) and yields at f_c in compression with mild softening
  (crushing). Torsion is elastic; a section whose shear stress 1.5 V/A + T/W_t passes
  f_v splits and the element is removed.
* **shells**: corotational CST + DKT triangles with laminate stiffness; a ply strain
  limit reached anywhere in an element cracks it (its stiffness falls to zero over a
  small strain range); a node held only by broken elements becomes loose debris.
* **glue lines**: zero-thickness cohesive springs with a bilinear mixed-mode law
  (quadratic initiation, Benzeggagh-Kenane propagation). In compression they keep
  bearing (contact); once broken they slide with Coulomb friction.
* **floss**: tension-only cables that snap at their breaking load.
* **contact**: penalty contact with friction against the base platform, the steel
  loading plate (a rigid body free to tilt on the jack's hemisphere) and the jack rod.
* **gravity** on everything, and mass-proportional damping (bridges are lightly damped).

Initial curvature of members (bow imperfections) and material scatter are set by the
model builder (:mod:`bridgesim.sim.idealize`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bridgesim.sim import kernels as K
from bridgesim.sim.model import G_MM_S2, Model

GP = np.array([0.5 - 0.5 / math.sqrt(3), 0.5 + 0.5 / math.sqrt(3)])


def skew_vee(M: np.ndarray) -> np.ndarray:
    return 0.5 * np.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0],
                           M[..., 1, 0] - M[..., 0, 1]], axis=-1)


def logvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector of rotation matrices (..., 3, 3)."""
    tr = np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2, -1.0, 1.0)
    ang = np.arccos(tr)
    v = skew_vee(R)
    s = np.sin(ang)
    fac = np.where(ang < 1e-4, 1.0 + ang * ang / 6, ang / np.where(s > 1e-12, s, 1.0))
    return v * fac[..., None]


def expmap(w: np.ndarray) -> np.ndarray:
    """Rotation matrices of rotation vectors (..., 3) (Rodrigues)."""
    th = np.linalg.norm(w, axis=-1)
    small = th < 1e-8
    k = w / np.where(small, 1.0, th)[..., None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s = np.sin(th)[..., None, None]
    c = (1 - np.cos(th))[..., None, None]
    R = np.eye(3) + s * K + c * (K @ K)
    R[small] = np.eye(3)
    return R


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-300, n, 1.0)


@dataclass
class State:
    x: np.ndarray
    R: np.ndarray
    v: np.ndarray
    w: np.ndarray  # spatial angular velocity
    # beam fibre history (nb, 2, nf)
    eps_p: np.ndarray
    dmg_t: np.ndarray
    b_alive: np.ndarray
    b_fractured: np.ndarray
    b_shear: np.ndarray
    s_dmg: np.ndarray
    c_dmax: np.ndarray  # max mixed-mode separation reached
    c_d: np.ndarray  # damage
    c_slip: np.ndarray  # (nc, 3) friction anchor offset
    k_alive: np.ndarray
    p_anchor: np.ndarray  # (np, 3) platform friction anchors
    p_touch: np.ndarray
    q_anchor: np.ndarray  # (nq, 3) plate-frame friction anchors
    q_touch: np.ndarray
    b_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    s_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 18)))
    c_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    work: tuple | None = None  # scratch arrays of the compiled shell kernel
    scratch: tuple | None = None  # contact force scratch arrays
    c_alive: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    loose: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # debris nodes
    t: float = 0.0
    jack_y: float = 0.0
    jack_v: float = 0.0  # jack speed (mm/s, negative = down)
    jack_force: float = 0.0
    jack_lat: np.ndarray = field(default_factory=lambda: np.zeros(2))
    work_ext: float = 0.0


def initial_state(m: Model, jack_y: float) -> State:
    n = m.n_nodes
    nb, nf = m.b_fA.shape if m.b_fA.size else (0, 0)
    nc = len(m.c_a)
    return State(
        x=m.X.copy(), R=np.repeat(np.eye(3)[None], n, axis=0), v=np.zeros((n, 3)),
        w=np.zeros((n, 3)), eps_p=np.zeros((nb, 2, nf)), dmg_t=np.zeros((nb, 2, nf)),
        b_alive=np.ones(nb, bool), b_fractured=np.zeros(nb, bool),
        b_shear=np.zeros(nb, bool), s_dmg=np.zeros(len(m.s_nodes)),
        c_dmax=np.zeros(nc), c_d=np.zeros(nc), c_slip=np.zeros((nc, 3)),
        k_alive=np.ones(len(m.k_a), bool), p_anchor=np.zeros((len(m.p_att), 3)),
        p_touch=np.zeros(len(m.p_att), bool), q_anchor=np.zeros((len(m.q_att), 3)),
        q_touch=np.zeros(len(m.q_att), bool), jack_y=jack_y,
        b_prev=np.zeros((nb, 7)), s_prev=np.zeros((len(m.s_nodes), 18)),
        c_prev=np.zeros((nc, 3)), c_alive=np.ones(nc, bool), loose=np.zeros(n, bool))


# ----------------------------------------------------------------------------- kernels


def _attach_pos(m: Model, st: State, att: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = m.a_node[att]
    arm = np.einsum("nij,nj->ni", st.R[nodes], m.a_r0[att])
    return st.x[nodes] + arm, arm


def _apply(F: np.ndarray, Mo: np.ndarray, nodes: np.ndarray, arm: np.ndarray,
           f: np.ndarray) -> None:
    np.add.at(F, nodes, f)
    np.add.at(Mo, nodes, np.cross(arm, f))


def beam_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray,
                update: bool = True, beta: float = 0.0, dt: float = 1.0) -> dict:
    if len(m.b_i) == 0:
        return {}
    bi, bj = m.b_i, m.b_j
    d = st.x[bj] - st.x[bi]
    l = np.linalg.norm(d, axis=1)
    e1 = d / l[:, None]
    Ti = st.R[bi] @ m.b_E0
    Tj = st.R[bj] @ m.b_E0
    q = 0.5 * (Ti[:, :, 1] + Tj[:, :, 1])
    e3 = _unit(np.cross(e1, q))
    e2 = np.cross(e3, e1)
    E = np.stack([e1, e2, e3], axis=2)
    Et = np.transpose(E, (0, 2, 1))
    th_i = logvec(Et @ Ti)
    th_j = logvec(Et @ Tj)
    L0 = m.b_L0
    eps0 = (l - L0) / L0
    N = np.zeros_like(l)
    Mzi = np.zeros_like(l)
    Mzj = np.zeros_like(l)
    Myi = np.zeros_like(l)
    Myj = np.zeros_like(l)
    E_ = m.b_E[:, None]
    ft = m.b_ft[:, None]
    fc = m.b_fc[:, None]
    et0 = ft / E_
    lch = (0.5 * L0)[:, None]
    eu = np.maximum(et0 + 2 * m.b_Gf[:, None] / (ft * lch), 1.5 * et0)
    soft = m.params.get("crush_softening", 0.3)
    ec0 = fc / E_
    for g, xi in enumerate(GP):
        kz = ((-4 + 6 * xi) * th_i[:, 2] + (-2 + 6 * xi) * th_j[:, 2]) / L0
        ky = ((-4 + 6 * xi) * th_i[:, 1] + (-2 + 6 * xi) * th_j[:, 1]) / L0
        eps = eps0[:, None] - m.b_fy * kz[:, None] + m.b_fz * ky[:, None]
        ep = st.eps_p[:, g]
        dm = st.dmg_t[:, g]
        ee = eps - ep
        dnew = np.where(ee > et0, np.clip(eu * (ee - et0) / (np.maximum(ee, 1e-30)
                                                             * (eu - et0)), 0, 1), 0.0)
        dm2 = np.maximum(dm, dnew)
        sig = np.where(ee > 0, (1 - dm2) * E_ * ee, E_ * ee)
        fce = fc * (1 - soft * np.clip(-ep / (5 * ec0), 0, 1))
        over = sig < -fce
        ep2 = np.where(over, eps + fce / E_, ep)
        sig = np.where(over, -fce, sig)
        if update:
            st.dmg_t[:, g] = dm2
            st.eps_p[:, g] = ep2
        A = m.b_fA
        Ng = (sig * A).sum(1)
        Mz = -(sig * A * m.b_fy).sum(1)
        My = (sig * A * m.b_fz).sum(1)
        N += 0.5 * Ng
        Mzi += 0.5 * Mz * (-4 + 6 * xi)
        Mzj += 0.5 * Mz * (-2 + 6 * xi)
        Myi += 0.5 * My * (-4 + 6 * xi)
        Myj += 0.5 * My * (-2 + 6 * xi)
    T = m.b_GJ / L0 * (th_j[:, 0] - th_i[:, 0])
    D = np.c_[l - L0, th_i, th_j]
    if beta > 0:
        rate = (D - st.b_prev) / dt
        EA = (m.b_E[:, None] * m.b_fA).sum(1)
        EIz = (m.b_E[:, None] * m.b_fA * m.b_fy**2).sum(1)
        EIy = (m.b_E[:, None] * m.b_fA * m.b_fz**2).sum(1)
        N = N + beta * EA / L0 * rate[:, 0]
        T = T + beta * m.b_GJ / L0 * (rate[:, 4] - rate[:, 1])
        Mzi = Mzi + beta * EIz / L0 * (4 * rate[:, 3] + 2 * rate[:, 6])
        Mzj = Mzj + beta * EIz / L0 * (2 * rate[:, 3] + 4 * rate[:, 6])
        Myi = Myi + beta * EIy / L0 * (4 * rate[:, 2] + 2 * rate[:, 5])
        Myj = Myj + beta * EIy / L0 * (2 * rate[:, 2] + 4 * rate[:, 5])
    if update:
        st.b_prev = D
    Vz = (Mzi + Mzj) / l
    Vy = (Myi + Myj) / l
    alive = st.b_alive
    if update:
        big = st.b_alive & (((th_i**2).sum(1) > 1.0) | ((th_j**2).sum(1) > 1.0))
        st.b_alive &= ~big
        st.b_fractured |= big
        tau = 1.5 * np.hypot(Vz, Vy) / m.b_A + np.abs(T) / m.b_Wt
        shear_fail = alive & (tau > m.b_fv)
        if shear_fail.any():
            st.b_shear |= shear_fail
            st.b_alive &= ~shear_fail
        cracked = (st.dmg_t > 0.99) * m.b_fA[:, None, :]
        frac = cracked.sum(2).max(1) / m.b_fA.sum(1)
        st.b_fractured |= frac > 0.9
    a = st.b_alive.astype(float)
    fl = np.stack([-N, Vz, -Vy], axis=1) * a[:, None]
    fi = np.einsum("nij,nj->ni", E, fl)
    mi = np.einsum("nij,nj->ni", E, np.stack([-T, Myi, Mzi], axis=1) * a[:, None])
    mj = np.einsum("nij,nj->ni", E, np.stack([T, Myj, Mzj], axis=1) * a[:, None])
    np.add.at(F, bi, fi)
    np.add.at(F, bj, -fi)
    np.add.at(Mo, bi, mi)
    np.add.at(Mo, bj, mj)
    return {"N": N * a, "Mz": 0.5 * (np.abs(Mzi) + np.abs(Mzj)) * a, "l": l}


def shell_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray,
                 update: bool = True, beta: float = 0.0, dt: float = 1.0) -> None:
    if len(m.s_nodes) == 0:
        return
    sn = m.s_nodes
    x = st.x[sn]  # (ns, 3, 3)
    c = x.mean(axis=1)
    ex = _unit(x[:, 1] - x[:, 0])
    ez = _unit(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]))
    ey = np.cross(ez, ex)
    E = np.stack([ex, ey, ez], axis=2)
    Et = np.transpose(E, (0, 2, 1))
    p = np.einsum("nij,naj->nai", Et, x - c[:, None, :])
    u = p - m.s_P
    th = logvec(Et[:, None] @ st.R[sn] @ m.s_E0[:, None])
    dloc = np.concatenate([u, th], axis=2).reshape(len(sn), 18)
    if update:
        st.s_dmg = np.where((th**2).sum(2).max(1) > 0.25, 1.0, st.s_dmg)
        strain = np.einsum("nij,nj->ni", m.s_R, dloc)  # noqa: E501
        em, kap = strain[:, :3], strain[:, 3:]
        fi = np.zeros(len(sn))
        for zf in (-0.5, 0.5):
            ez_ = em + (zf * m.s_t)[:, None] * kap
            for k in range(2):
                ang = m.s_ang[:, k]
                ok = ~np.isnan(ang)
                a = np.radians(np.nan_to_num(ang))
                cs, sn_ = np.cos(a), np.sin(a)
                e1 = ez_[:, 0] * cs**2 + ez_[:, 1] * sn_**2 + ez_[:, 2] * cs * sn_
                e2 = ez_[:, 0] * sn_**2 + ez_[:, 1] * cs**2 - ez_[:, 2] * cs * sn_
                g12 = 2 * (ez_[:, 1] - ez_[:, 0]) * cs * sn_ + ez_[:, 2] * (cs**2 - sn_**2)
                lim = m.s_lim
                r = np.maximum(np.where(e1 > 0, e1 / lim[:, 0], -e1 / lim[:, 1]),
                               np.abs(g12) / lim[:, 4])
                r2 = np.where(e2 > 0, e2 / lim[:, 2], -e2 / lim[:, 3])
                r = np.where(m.s_matrix, np.maximum(r, r2), r)
                fi = np.maximum(fi, np.where(ok, r, 0))
        live = (fi >= 1.0) & (st.s_dmg < 1.0)
        target = np.minimum(1.0, 2.0 * (fi - 1.0) + 0.1)
        st.s_dmg = np.where(live, np.maximum(st.s_dmg, target), st.s_dmg)
    dd = dloc + (beta * (dloc - st.s_prev) / dt if beta > 0 else 0.0)
    if update:
        st.s_prev = dloc
    f = np.einsum("nij,nj->ni", m.s_K, dd) * (1 - st.s_dmg)[:, None]
    f = f.reshape(len(sn), 3, 6)
    fg = np.einsum("nij,naj->nai", E, f[:, :, :3])
    mg = np.einsum("nij,naj->nai", E, f[:, :, 3:])
    np.add.at(F, sn.ravel(), fg.reshape(-1, 3))
    np.add.at(Mo, sn.ravel(), mg.reshape(-1, 3))


def glue_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray,
                update: bool = True, beta: float = 0.0, dt: float = 1.0) -> None:
    if len(m.c_a) == 0:
        return
    pa, arm_a = _attach_pos(m, st, m.c_a)
    pb, arm_b = _attach_pos(m, st, m.c_b)
    na, nb = m.a_node[m.c_a], m.a_node[m.c_b]
    n = _unit(np.einsum("nij,nj->ni", st.R[na] + st.R[nb], m.c_n0))
    delta = pb - pa
    dn = (delta * n).sum(1)
    ds = delta - dn[:, None] * n
    s = np.linalg.norm(ds, axis=1)
    dnp = np.maximum(dn, 0.0)
    Kn, Ks = m.c_Kn, m.c_Ks
    dn0 = m.c_sig / Kn
    ds0 = m.c_tau / Ks
    dm = np.sqrt(dnp**2 + s**2)
    mix = s / np.maximum(dnp, 1e-12)
    dm0 = np.where(dnp > 1e-12, dn0 * ds0 * np.sqrt((1 + mix**2)
                                                   / (ds0**2 + (mix * dn0) ** 2)), ds0)
    Gs = Ks * s**2
    Gn = Kn * dnp**2
    B = Gs / np.maximum(Gs + Gn, 1e-30)
    Gc = m.c_GIc + (m.c_GIIc - m.c_GIc) * B**2
    Keff = np.where(dm > 1e-12, (Gn + Gs) / np.maximum(dm**2, 1e-30), Ks)
    dmf = np.maximum(2 * Gc / (Keff * np.maximum(dm0, 1e-12)), 1.05 * dm0)
    dmax = np.maximum(st.c_dmax, dm)
    d = np.clip(dmf * (dmax - dm0) / (np.maximum(dmax, 1e-12) * (dmf - dm0)), 0, 1)
    d = np.where(m.c_breakable, np.maximum(st.c_d, d), 0.0)
    if update:
        st.c_dmax = np.where(m.c_breakable, dmax, 0.0)
        st.c_d = d
    tn = np.where(dn > 0, (1 - d) * Kn * dn, Kn * dn)
    # friction on the broken part (anchor follows the slip)
    rel = ds - st.c_slip
    rel = rel - (rel * n).sum(1)[:, None] * n
    tf = -(Ks[:, None]) * rel * -1  # traction from elastic tangential spring (+ dir)
    cap = m.c_mu * np.maximum(-tn, 0.0)
    mag = np.linalg.norm(tf, axis=1)
    slip = mag > cap
    scale = np.where(slip, cap / np.maximum(mag, 1e-30), 1.0)
    tf = tf * scale[:, None]
    if update:
        new_anchor = ds - tf / Ks[:, None]
        st.c_slip = np.where((d > 0)[:, None], new_anchor, ds)
    ts = (1 - d)[:, None] * Ks[:, None] * ds + d[:, None] * tf
    t = tn[:, None] * n + ts
    f = t * m.c_area[:, None] * st.c_alive[:, None]  # force pulling B back towards A
    if beta > 0:
        rate = (delta - st.c_prev) / dt
        rn = (rate * n).sum(1)
        kk = np.where(dn <= 0, Kn, (1 - d) * Kn)
        f = f + beta * (kk[:, None] * rn[:, None] * n
                        + ((1 - d) * Ks)[:, None] * (rate - rn[:, None] * n)) \
            * (m.c_area * st.c_alive)[:, None]
    if update:
        st.c_prev = delta
    # f pulls A towards B; F holds internal (resisting) forces, hence the signs
    _apply(F, Mo, na, arm_a, -f)
    _apply(F, Mo, nb, arm_b, f)


def cable_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray,
                 update: bool = True) -> None:
    if len(m.k_a) == 0:
        return
    pa, arm_a = _attach_pos(m, st, m.k_a)
    pb, arm_b = _attach_pos(m, st, m.k_b)
    d = pb - pa
    L = np.linalg.norm(d, axis=1)
    T = np.maximum(m.k_EA * (L - m.k_L0) / m.k_L0, 0.0)
    if update:
        st.k_alive &= T < m.k_Fb
    T = T * st.k_alive
    f = d / np.maximum(L, 1e-9)[:, None] * T[:, None]
    _apply(F, Mo, m.a_node[m.k_a], arm_a, -f)
    _apply(F, Mo, m.a_node[m.k_b], arm_b, f)


def platform_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray, k: float,
                    mu: float, update: bool = True, cdamp=0.0) -> None:
    p, arm = _attach_pos(m, st, m.p_att)
    pen = m.table_y - p[:, 1]
    touch = (pen > 0) & ~st.loose[m.a_node[m.p_att]]
    nd = m.a_node[m.p_att]
    vy = st.v[nd, 1] + st.w[nd, 2] * arm[:, 0] - st.w[nd, 0] * arm[:, 2]
    fn = np.where(touch, np.maximum(k * pen - cdamp * vy, 0.0), 0.0)
    k = np.broadcast_to(np.asarray(k, float), (len(p),))
    rel = p - st.p_anchor
    rel[:, 1] = 0
    ft = -k[:, None] * rel
    cap = mu * fn
    mag = np.linalg.norm(ft, axis=1)
    sc = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
    ft = ft * sc[:, None] * touch[:, None]
    if update:
        anc = p + ft / k[:, None]
        st.p_anchor = np.where(touch[:, None], anc, p)
        st.p_touch = touch
    f = ft.copy()
    f[:, 1] += fn
    _apply(F, Mo, m.a_node[m.p_att], arm, f)


def plate_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray, k: float,
                 mu: float, update: bool = True, cdamp=0.0) -> float:
    """Contact between bridge points and the rigid loading plate; returns the total
    downward force the plate puts on the bridge."""
    if len(m.q_att) == 0 or m.plate_node < 0:
        return 0.0
    P = m.plate_node
    Rp = st.R[P]
    xp = st.x[P]
    p, arm = _attach_pos(m, st, m.q_att)
    loc = (p - xp) @ Rp  # plate-frame coordinates
    k = np.broadcast_to(np.asarray(k, float), (len(p),))
    h = m.plate_half
    depth = h[None, :] - np.abs(loc)
    inside = (depth > 0).all(axis=1) & ~st.loose[m.a_node[m.q_att]]
    ax = np.argmin(depth, axis=1)
    sgn = np.sign(loc[np.arange(len(loc)), ax])
    sgn[sgn == 0] = -1.0
    nloc = np.zeros_like(loc)
    nloc[np.arange(len(loc)), ax] = sgn
    pen = depth[np.arange(len(loc)), ax]
    n = nloc @ Rp.T
    nd = m.a_node[m.q_att]
    vb = st.v[nd] + np.cross(st.w[nd], arm)
    vp = st.v[P] + np.cross(st.w[P], p - xp)
    vr = ((vb - vp) * n).sum(1)
    fn = np.where(inside, np.maximum(k * pen - cdamp * vr, 0.0), 0.0)
    rel = loc - st.q_anchor
    rel = rel - (rel * nloc).sum(1)[:, None] * nloc
    ftl = -k[:, None] * rel
    cap = mu * fn
    mag = np.linalg.norm(ftl, axis=1)
    sc = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
    ftl = ftl * (sc * inside)[:, None]
    if update:
        st.q_anchor = np.where(inside[:, None], loc + ftl / k[:, None], loc)
        st.q_touch = inside
    f = fn[:, None] * n + ftl @ Rp.T  # on the bridge point
    _apply(F, Mo, m.a_node[m.q_att], arm, f)
    np.add.at(F, P, -f.sum(0))
    Mo[P] -= np.cross(p - xp, f).sum(0)
    return float(-f[:, 1].sum())


def jack_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray, k: float,
                k_lat: float, cdamp: float = 2e-4, jack_v: float = 0.0) -> float:
    P = m.plate_node
    if P < 0:
        return 0.0
    seat = st.x[P] + st.R[P] @ np.array([0.0, m.plate_half[1], 0.0])
    pen = seat[1] - st.jack_y
    if pen <= 0:
        st.jack_force = 0.0
        st.jack_lat[:] = 0
        return 0.0
    arm = seat - st.x[P]
    vseat = st.v[P] + np.cross(st.w[P], arm)
    fy = min(0.0, -k * pen - cdamp * k * (vseat[1] - jack_v))
    lat = -k_lat * np.array([seat[0] - m.jack_xz[0], seat[2] - m.jack_xz[1]]) \
        - cdamp * k_lat * np.array([vseat[0], vseat[2]])
    f = np.array([lat[0], fy, lat[1]])
    F[P] += f
    Mo[P] += np.cross(seat - st.x[P], f)
    st.jack_force = -fy
    st.jack_lat = lat
    return -fy


@dataclass
class Settings:
    dt: float
    alpha: float = 30.0  # mass-proportional damping (1/s)
    k_platform: float = 2000.0  # N/mm per contact point
    k_plate: float = 2000.0
    k_jack: float = 20000.0  # crusher + load cell stiffness (N/mm)
    k_jack_lat: float = 2000.0
    mu_platform: float = 0.35
    mu_plate: float = 0.3
    gravity: float = G_MM_S2
    use_numba: bool = True
    contact_damping: float = 2e-4  # s: normal contact damping c = 2e-4 * k (~15 % of
    #                               critical for the plate on the deck); contacts dissipate
    beta_frac: float = 0.1  # material damping beta = beta_frac * dt (damps only the
    #                         highest, numerical frequencies; ~0 for the bridge's modes)

    @property
    def beta(self) -> float:
        return self.beta_frac * self.dt


def internal_and_contact(m: Model, st: State, s: Settings, F: np.ndarray, Mo: np.ndarray,
                         update: bool = True) -> dict:
    if s.use_numba and K.HAVE_NUMBA:
        return _internal_fast(m, st, s, F, Mo, update)
    return internal_and_contact_numpy(m, st, s, F, Mo, update)


def contact_damping_coeffs(m: Model, s: Settings, mass: np.ndarray) -> None:
    """Penalty stiffness and normal damping of every platform and plate contact point.

    Any part may land on the platform or the plate during a collapse, so contact must be
    stable for every node, not only those touching at the start: like the "soft"
    penalty of explicit codes, a point's stiffness is the nominal k but never more than
    its node can take at this time step (sum over the node's points of k <= 0.3 m/dt^2),
    and its damping c = contact_damping * k, likewise capped (sum c <= 0.4 m/dt)."""
    for key, att, k in (("p", m.p_att, s.k_platform), ("q", m.q_att, s.k_plate)):
        nd = m.a_node[att]
        cnt = np.bincount(nd, minlength=len(mass))[nd] if len(nd) else np.zeros(0)
        kk = np.minimum(k, 0.3 * mass[nd] / (s.dt**2 * np.maximum(cnt, 1)))
        c = np.minimum(s.contact_damping * kk, 0.4 * mass[nd] / (s.dt * np.maximum(cnt, 1)))
        m.params[f"cdamp_{key}"] = c
        m.params[f"kc_{key}"] = kk


def _kc(m: Model, key: str, s: Settings) -> np.ndarray:
    n = len(m.p_att) if key == "p" else len(m.q_att)
    k = m.params.get(f"kc_{key}")
    if k is None or len(k) != n:
        return np.full(n, s.k_platform if key == "p" else s.k_plate)
    return k


def _cd(m: Model, key: str, s: Settings, update: bool) -> np.ndarray:
    n = len(m.p_att) if key == "p" else len(m.q_att)
    c = m.params.get(f"cdamp_{key}")
    if not update or c is None or len(c) != n:
        return np.zeros(n)
    return c


def _internal_fast(m: Model, st: State, s: Settings, F: np.ndarray, Mo: np.ndarray,
                   update: bool) -> dict:
    F[:] = 0
    Mo[:] = 0
    x, R = st.x, st.R
    if len(m.b_i):
        K.beam_kernel(x, R, m.b_i, m.b_j, m.b_L0, m.b_E0, m.b_fy, m.b_fz, m.b_fA, m.b_E,
                      m.b_ft, m.b_fc, m.b_Gf, m.b_GJ, m.b_A, m.b_fv, m.b_Wt,
                      float(m.params.get("crush_softening", 0.3)), st.eps_p, st.dmg_t,
                      st.b_alive, st.b_fractured, st.b_shear, F, Mo, update,
                      s.beta if update else 0.0, s.dt, st.b_prev)
    if len(m.s_nodes):
        ns = len(m.s_nodes)
        if st.work is None or st.work[0].shape[0] != ns:
            st.work = (np.zeros((ns, 3, 3)), np.zeros((ns, 3, 3)), np.zeros((ns, 18)),
                       np.zeros((ns, 3, 3)), np.zeros((ns, 3)))
        fe, me, wd, wR, wth = st.work
        K.shell_kernel(x, R, m.s_nodes, m.s_K, m.s_E0, m.s_P, m.s_R, m.s_t, m.s_ang,
                       m.s_lim, m.s_matrix, st.s_dmg, F, Mo, update, fe, me,
                       s.beta if update else 0.0, s.dt, st.s_prev, wd, wR, wth)
    if len(m.c_a):
        K.glue_kernel(x, R, m.a_node, m.a_r0, m.c_a, m.c_b, m.c_area, m.c_n0, m.c_Kn,
                      m.c_Ks, m.c_sig, m.c_tau, m.c_GIc, m.c_GIIc, m.c_mu, m.c_breakable,
                      st.c_dmax, st.c_d, st.c_slip, F, Mo, update,
                      s.beta if update else 0.0, s.dt, st.c_prev, st.c_alive)
    if len(m.k_a):
        K.cable_kernel(x, R, m.a_node, m.a_r0, m.k_a, m.k_b, m.k_L0, m.k_EA, m.k_Fb,
                       st.k_alive, F, Mo, update)
    if st.scratch is None or st.scratch[0].shape != F.shape:
        st.scratch = (np.zeros_like(F), np.zeros_like(Mo))
    Fc, Mc = st.scratch
    Fc[:] = 0.0
    Mc[:] = 0.0
    K.platform_kernel(x, R, m.a_node, m.a_r0, m.p_att, m.table_y, _kc(m, "p", s),
                      s.mu_platform, st.p_anchor, st.p_touch, Fc, Mc, update, st.v, st.w,
                      _cd(m, "p", s, update), st.loose)
    plate_load = 0.0
    if len(m.q_att) and m.plate_node >= 0:
        plate_load = K.plate_kernel(x, R, m.a_node, m.a_r0, m.q_att, m.plate_node,
                                    m.plate_half, _kc(m, "q", s), s.mu_plate, st.q_anchor,
                                    st.q_touch, Fc, Mc, update, st.v, st.w,
                                    _cd(m, "q", s, update), st.loose)
    jack_forces(m, st, Fc, Mc, s.k_jack, s.k_jack_lat, s.contact_damping, st.jack_v)
    F -= Fc
    Mo -= Mc
    return {"plate_load": plate_load}


def internal_and_contact_numpy(m: Model, st: State, s: Settings, F: np.ndarray,
                               Mo: np.ndarray, update: bool = True) -> dict:
    """Reference NumPy implementation (slow, used to check the compiled kernels)."""
    F[:] = 0
    Mo[:] = 0
    b = s.beta if update else 0.0
    info = beam_forces(m, st, F, Mo, update, b, s.dt)
    shell_forces(m, st, F, Mo, update, b, s.dt)
    glue_forces(m, st, F, Mo, update, b, s.dt)
    cable_forces(m, st, F, Mo, update)
    # contacts produce forces *on* nodes; convert to "internal" sign convention below
    Fc = np.zeros_like(F)
    Mc = np.zeros_like(Mo)
    platform_forces(m, st, Fc, Mc, _kc(m, "p", s), s.mu_platform, update,
                    _cd(m, "p", s, update))
    plate_load = plate_forces(m, st, Fc, Mc, _kc(m, "q", s), s.mu_plate, update,
                              _cd(m, "q", s, update))
    jack_forces(m, st, Fc, Mc, s.k_jack, s.k_jack_lat, s.contact_damping, st.jack_v)
    F -= Fc
    Mo -= Mc
    info["plate_load"] = plate_load
    return info


def step(m: Model, st: State, s: Settings, F: np.ndarray, Mo: np.ndarray,
         grav_scale: float = 1.0, alpha: float | None = None) -> dict:
    """Advance one central-difference step (velocities at half steps)."""
    info = internal_and_contact(m, st, s, F, Mo)
    a_ = s.alpha if alpha is None else alpha
    fext = np.zeros_like(F)
    fext[:, 1] = -m.mass * s.gravity * grav_scale
    acc = (fext - F) / m.mass[:, None] - a_ * st.v
    st.v += s.dt * acc
    st.x += s.dt * st.v
    alpha_acc = -Mo / m.rot_inertia[:, None] - a_ * st.w
    st.w += s.dt * alpha_acc
    rotate(st, s.dt, s.use_numba)
    st.t += s.dt
    return info


def rotate(st: State, dt: float, use_numba: bool = True) -> None:
    if use_numba and K.HAVE_NUMBA:
        K.expmap_update(st.R, st.w, dt)
    else:
        st.R = expmap(st.w * dt) @ st.R


def kinetic_energy(m: Model, st: State) -> float:
    return 0.5 * float((m.mass * (st.v**2).sum(1)).sum()
                       + (m.rot_inertia * (st.w**2).sum(1)).sum())


def reorthonormalize(st: State) -> None:
    U, _, Vt = np.linalg.svd(st.R)
    st.R = U @ Vt
