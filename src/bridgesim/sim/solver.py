"""Explicit nonlinear dynamics of the bridge (central-difference time integration).

Every time step computes the internal forces of the deformed structure directly; there is
no stiffness matrix to factor, which is what lets the bridge crack, lose joints, buckle,
slide on the platform and collapse without convergence failures (the approach of
LS-DYNA and Abaqus/Explicit). Ingredients:

* **beams**: corotational 3D elements (the element frame follows the deformed chord, so
  rotations of any size are exact) with *fibre sections*: each section is a grid of
  wood fibres with their own stress-strain law, integrated at two Gauss points. A fibre is
  linear elastic, softens to zero after reaching f_t in tension (fracture energy G_f,
  regularised by element length) and in compression bends over from the proportional
  limit to f_c, then crushes with mild softening. Torsion is elastic. A member whose
  shear, averaged over a section depth, passes the strength of its glue lines or grain
  splits (:func:`split_members`): its plies then bend each about its own axis, and it
  keeps its axial strength. An element bent or twisted past 1 rad at an end is removed
  (erosion, reported as such).
* **shells**: corotational CST + DKMT (shear-deformable; DKT with plate.shear_deformable
  0) triangles with laminate stiffness; a ply strain limit reached anywhere in an
  element cracks it (its stiffness falls to zero over a small strain range); a node held
  only by broken elements becomes loose debris. A plate whose out-of-plane shear,
  averaged over its thickness, reaches the strength of its glue planes splits between
  its plies (:func:`split_plates`), as members do.
* **glue lines**: zero-thickness cohesive springs with a bilinear mixed-mode law
  (quadratic initiation, Benzeggagh-Kenane propagation). In compression they keep
  bearing (contact); once broken they slide with Coulomb friction.
* **floss**: tension-only cables that snap at their breaking load.
* **contact**: penalty contact with friction against the base platform and the steel
  loading plate (a rigid body free to tilt on the jack's hemisphere), and frictionless
  contact between parts that meet during the collapse (:mod:`bridgesim.sim.contact`).
  The jack rod above the plate is not a contact body (only drawn in the replay).
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
    # shells: per in-plane point, ply layer and through-thickness point, the largest
    # strain / onset strain reached in each failure mode (fibre t, fibre c, across t,
    # across c, shear); s_dmg is the through-crack indicator, s_gone the eroded elements
    s_hist: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 5)))
    s_gone: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # beams split in shear: factor on the bending lever arms across the width (s2) and
    # through the plies (s3); 1 intact, 1/2 and 1/n after splitting
    c_crush: np.ndarray = field(default_factory=lambda: np.zeros(0))  # glue-line set
    c_creep: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # creep slip
    creep_dt: float = 0.0  # physical test time of the current step (s): jack travel /
    #                        the real test rate; 0 while the jack stands
    t_test: float = 0.0  # physical test time so far (s): the sum of creep_dt
    c_t0: np.ndarray = field(default_factory=lambda: np.zeros(0))  # physical test time
    #   at which each glue point first took load (RATE_LOAD of its strength), -1 not yet:
    #   its time to failure, which its strength follows, runs from then
    b_s2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_s3: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # shear stress / strength of each beam (through the plies, across the width)
    b_rat2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_rat3: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Timoshenko shear strain correction of each beam, bending about e3 and e2
    b_gam: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    # shells split between their plies in out-of-plane shear (split_plates): factor on
    # the through-thickness lever arms (1 intact, 1/2, ... 1/n_plies), the interlaminar
    # shear over its strength of each element (before averaging) and whether it split
    s_split: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_rat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_shear: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    sc: object | None = None  # self-contact (bridgesim.sim.contact.SelfContact)
    t: float = 0.0
    jack_y: float = 0.0
    jack_v: float = 0.0  # jack speed (mm/s, negative = down)
    jack_force: float = 0.0
    jack_lat: np.ndarray = field(default_factory=lambda: np.zeros(2))
    work_ext: float = 0.0


def _d0(m: Model, nc: int) -> np.ndarray:
    d0 = m.params.get("c_d0")
    return np.array(d0, float) if d0 is not None and len(d0) == nc else np.zeros(nc)


def initial_state(m: Model, jack_y: float) -> State:
    n = m.n_nodes
    nb, nf = m.b_fA.shape if m.b_fA.size else (0, 0)
    nc = len(m.c_a)
    return State(
        x=m.X.copy(), R=np.repeat(np.eye(3)[None], n, axis=0), v=np.zeros((n, 3)),
        w=np.zeros((n, 3)), eps_p=np.zeros((nb, 2, nf)), dmg_t=np.zeros((nb, 2, nf)),
        b_alive=np.ones(nb, bool), b_fractured=np.zeros(nb, bool),
        b_shear=np.zeros(nb, bool), s_dmg=np.zeros(len(m.s_nodes)),
        c_dmax=np.zeros(nc), c_d=_d0(m, nc), c_slip=np.zeros((nc, 3)),
        k_alive=np.ones(len(m.k_a), bool), p_anchor=np.zeros((len(m.p_att), 3)),
        p_touch=np.zeros(len(m.p_att), bool), q_anchor=np.zeros((len(m.q_att), 3)),
        q_touch=np.zeros(len(m.q_att), bool), jack_y=jack_y,
        b_prev=np.zeros((nb, 7)), s_prev=np.zeros((len(m.s_nodes), 18)),
        c_prev=np.zeros((nc, 3)), c_alive=np.ones(nc, bool), loose=np.zeros(n, bool),
        s_hist=np.zeros((len(m.s_nodes), 9 * m.s_lay.shape[1], 5)),
        s_gone=np.zeros(len(m.s_nodes), bool), b_s2=np.ones(nb), b_s3=np.ones(nb),
        b_rat2=np.zeros(nb), b_rat3=np.zeros(nb),
        b_gam=np.zeros((nb, 2)), c_crush=np.zeros(nc), c_creep=np.zeros((nc, 3)),
        s_split=np.ones(len(m.s_nodes)), s_rat=np.zeros(len(m.s_nodes)),
        s_shear=np.zeros(len(m.s_nodes), bool))


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
    kap = float(m.params.get("fc_proportional", 1.0))
    ec0 = fc / E_
    # Timoshenko shear flexibility (see kernels.beam_kernel)
    EIz = (m.b_E[:, None] * m.b_fA * m.b_fy**2).sum(1) * st.b_s2**2
    EIy = (m.b_E[:, None] * m.b_fA * m.b_fz**2).sum(1) * st.b_s3**2
    kGA = m.b_kGA
    pos = kGA > 0
    kGAs = np.where(pos, kGA, 1.0)
    phz = np.where(pos, 12 * EIz / (kGAs * L0**2), 0.0)
    phy = np.where(pos, 12 * EIy / (kGAs * L0**2), 0.0)
    gz = 0.5 * phz / (1 + phz) * (th_i[:, 2] + th_j[:, 2]) + st.b_gam[:, 0]
    gy = 0.5 * phy / (1 + phy) * (th_i[:, 1] + th_j[:, 1]) + st.b_gam[:, 1]
    bzi, bzj = th_i[:, 2] - gz, th_j[:, 2] - gz
    byi, byj = th_i[:, 1] - gy, th_j[:, 1] - gy
    EIzt = np.zeros_like(l)
    EIyt = np.zeros_like(l)
    for g, xi in enumerate(GP):
        kz = ((-4 + 6 * xi) * bzi + (-2 + 6 * xi) * bzj) / L0
        ky = ((-4 + 6 * xi) * byi + (-2 + 6 * xi) * byj) / L0
        s2, s3 = st.b_s2[:, None], st.b_s3[:, None]
        eps = eps0[:, None] - s2 * m.b_fy * kz[:, None] + s3 * m.b_fz * ky[:, None]
        ep = st.eps_p[:, g]
        dm = st.dmg_t[:, g]
        ee = eps - ep
        dnew = np.where(ee > et0, np.clip(eu * (ee - et0) / (np.maximum(ee, 1e-30)
                                                             * (eu - et0)), 0, 1), 0.0)
        dm2 = np.maximum(dm, dnew)
        sig = np.where(ee > 0, (1 - dm2) * E_ * ee, E_ * ee)
        # compression: linear to kap f_c, rising to f_c, then crushing (kernels)
        q = np.maximum(-ep, 0.0)
        qp = (1 - kap) * ec0
        rising = q < qp
        u = np.sqrt(np.where(rising, q / np.maximum(qp, 1e-30), 1.0))
        fce = np.where(rising, fc * (1 - (1 - kap) * (1 - u) ** 2),
                       fc * (1 - soft * np.clip((q - qp) / (5 * ec0), 0, 1)))
        over = sig < -fce
        Et = np.where(over, np.where(rising, (1 - u) * E_, 0.0),
                      np.where(ee > 0, (1 - dm2) * E_, E_))
        ep2 = np.where(over, eps + fce / E_, ep)
        sig = np.where(over, -fce, sig)
        EIzt += 0.5 * (Et * m.b_fA * m.b_fy**2).sum(1) * st.b_s2**2
        EIyt += 0.5 * (Et * m.b_fA * m.b_fz**2).sum(1) * st.b_s3**2
        if update:
            st.dmg_t[:, g] = dm2
            st.eps_p[:, g] = ep2
        A = m.b_fA
        Ng = (sig * A).sum(1)
        Mz = -(sig * A * m.b_fy * s2).sum(1)
        My = (sig * A * m.b_fz * s3).sum(1)
        N += 0.5 * Ng
        Mzi += 0.5 * Mz * (-4 + 6 * xi)
        Mzj += 0.5 * Mz * (-2 + 6 * xi)
        Myi += 0.5 * My * (-4 + 6 * xi)
        Myj += 0.5 * My * (-2 + 6 * xi)
    if update:
        st.b_gam[:, 0] += np.where(pos, ((Mzi + Mzj) / L0 - kGA * gz)
                                   / (kGAs + 12 * EIzt / L0**2), 0)
        st.b_gam[:, 1] += np.where(pos, ((Myi + Myj) / L0 - kGA * gy)
                                   / (kGAs + 12 * EIyt / L0**2), 0)
    sj = (st.b_s2 * st.b_s3) ** 2
    T = sj * m.b_GJ / L0 * (th_j[:, 0] - th_i[:, 0])
    D = np.c_[l - L0, th_i, th_j]
    if beta > 0:
        rate = (D - st.b_prev) / dt
        EA = (m.b_E[:, None] * m.b_fA).sum(1)
        cz = beta * EIz / (L0 * (1 + phz))
        cy = beta * EIy / (L0 * (1 + phy))
        N = N + beta * EA / L0 * rate[:, 0]
        T = T + beta * sj * m.b_GJ / L0 * (rate[:, 4] - rate[:, 1])
        Mzi = Mzi + cz * ((4 + phz) * rate[:, 3] + (2 - phz) * rate[:, 6])
        Mzj = Mzj + cz * ((2 - phz) * rate[:, 3] + (4 + phz) * rate[:, 6])
        Myi = Myi + cy * ((4 + phy) * rate[:, 2] + (2 - phy) * rate[:, 5])
        Myj = Myj + cy * ((2 - phy) * rate[:, 2] + (4 + phy) * rate[:, 5])
    if update:
        st.b_prev = D
    Vz = (Mzi + Mzj) / l
    Vy = (Myi + Myj) / l
    alive = st.b_alive
    if update:
        big = st.b_alive & (((th_i**2).sum(1) > 1.0) | ((th_j**2).sum(1) > 1.0))
        st.b_alive &= ~big  # erosion (member_erosion), not a fracture of the fibres
        st.b_rat3 = np.where(alive, (1.5 * np.abs(Vy) / m.b_A + np.abs(T) / m.b_Wt[:, 1])
                             / m.b_fv3, 0.0)
        st.b_rat2 = np.where(alive, (1.5 * np.abs(Vz) / m.b_A + np.abs(T) / m.b_Wt[:, 0])
                             / m.b_fv, 0.0)
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


def _rc(m: Model) -> float:
    """Crushing plateau of the fibres, as a fraction of the compressive strength."""
    return 1.0 - float(m.params.get("crush_softening", 0.3))


def _soft(r: np.ndarray, rf: np.ndarray) -> np.ndarray:
    d = rf / (rf - 1.0) * (1.0 - 1.0 / np.maximum(r, 1.0))
    return np.where(r > 1.0, np.minimum(d, 1.0), 0.0)


def _plateau(r: np.ndarray, rc: float) -> np.ndarray:
    u = np.minimum(np.maximum(r - 1.0, 0.0), 1.0)
    return np.where(r > 1.0, 1.0 - (1.0 - (1.0 - rc) * u) / np.maximum(r, 1.0), 0.0)


def _ply_strain(e, k, z, cc2, ss2, cs):
    """Strain at height z (membrane e + z x curvature k) in the grain axes."""
    exx = e[:, 0] + z * k[:, 0]
    eyy = e[:, 1] + z * k[:, 1]
    gxy = e[:, 2] + z * k[:, 2]
    return (exx * cc2 + eyy * ss2 + gxy * cs, exx * ss2 + eyy * cc2 - gxy * cs,
            2 * (eyy - exx) * cs + gxy * (cc2 - ss2))


_IM = np.array([0, 1, 6, 7, 12, 13])
_IB = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
_IR = np.array([5, 11, 17])


def shell_shear_arrays(m: Model) -> tuple[np.ndarray, ...]:
    """The DKMT arrays of the shells (s_R, s_KH, s_Gs, s_phi, s_H, s_fil), zeros (DKT,
    no splitting) for a model built without them."""
    from bridgesim.sim.shell import SHEAR_SHAPES

    ns = len(m.s_nodes)
    if m.s_H.shape[0] != ns:
        for k, shp in SHEAR_SHAPES.items():
            setattr(m, f"s_{k}", np.zeros((ns, *shp)))
    return m.s_R, m.s_KH, m.s_Gs, m.s_phi, m.s_H, m.s_fil


def shell_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray,
                 update: bool = True, beta: float = 0.0, dt: float = 1.0) -> None:
    """Layered shells (see :mod:`bridgesim.sim.shell`): forces integrated ply by ply with
    a damage state per failure mode at every integration point. A shear-deformable
    element (s_H > 0) adds the DKMT correction to its curvatures and its transverse shear
    force Q = H gamma (elastic), and records its interlaminar shear 1.5 |Q| / t over the
    strength of its glue planes (resolved along and across each ply's grain) in
    ``st.s_rat``; a split element (``st.s_split`` < 1) bends with its lever arms x s."""
    if len(m.s_nodes) == 0:
        return
    sR, sKH, sGs, sphi, sH, sfil = shell_shear_arrays(m)
    sn = m.s_nodes
    ns = len(sn)
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
    dloc = np.concatenate([u, th], axis=2).reshape(ns, 18)
    if update:
        torn = (th**2).sum(2).max(1) > 0.25
        st.s_dmg = np.where(torn & ~st.s_gone, 1.0, st.s_dmg)
        st.s_gone = st.s_gone | torn
    dd = dloc + (beta * (dloc - st.s_prev) / dt if beta > 0 else 0.0)
    if update:
        st.s_prev = dloc
    em = np.einsum("nik,nk->ni", m.s_Bm, dloc[:, _IM])
    emt = np.einsum("nik,nk->ni", m.s_Bm, dd[:, _IM])
    kap = np.einsum("ngik,nk->ngi", m.s_Bb, dloc[:, _IB])
    kapt = np.einsum("ngik,nk->ngi", m.s_Bb, dd[:, _IB])
    # DKMT: the sides' share of their Kirchhoff residual that is shear, om = s^2 phi /
    # (1 + s^2 phi) (0 for DKT), corrects the curvatures and gives the shear strain
    shear = sH[:, 0] > 0.0
    ssp = st.s_split if len(st.s_split) == ns else np.ones(ns)
    s2 = (ssp * ssp)[:, None]
    om = np.where(shear[:, None], s2 * sphi / (1.0 + s2 * sphi), 0.0)
    rd = om * np.einsum("nkj,nj->nk", sR, dloc[:, _IB])
    rw = om * np.einsum("nkj,nj->nk", sR, dd[:, _IB])
    kap = kap + 1.5 * np.einsum("ngik,nk->ngi", sKH, rd)
    kapt = kapt + 1.5 * np.einsum("ngik,nk->ngi", sKH, rw)
    gd = np.einsum("nik,nk->ni", sGs, rd)
    gw = np.einsum("nik,nk->ni", sGs, rw)
    Qs = np.stack([sH[:, 0] * gw[:, 0] + sH[:, 2] * gw[:, 1],
                   sH[:, 2] * gw[:, 0] + sH[:, 1] * gw[:, 1]], axis=1)
    if update:
        qd = np.stack([sH[:, 0] * gd[:, 0] + sH[:, 2] * gd[:, 1],
                       sH[:, 2] * gd[:, 0] + sH[:, 1] * gd[:, 1]], axis=1)
        rat = np.sqrt((qd**2).sum(1)) * sfil[:, 2]
        for ll in range(m.s_lay.shape[1]):
            c_, s_ = m.s_lay[:, ll, 2], m.s_lay[:, ll, 3]
            tl = (c_ * qd[:, 0] + s_ * qd[:, 1]) * sfil[:, 0]
            tt = (-s_ * qd[:, 0] + c_ * qd[:, 1]) * sfil[:, 1]
            rat = np.where(ll < m.s_nl, np.maximum(rat, np.sqrt(tl * tl + tt * tt)), rat)
        st.s_rat = np.where(st.s_gone | ~shear, 0.0, rat)
    q11, q22, q12, q66 = m.s_Q.T
    lim, rf, rc = m.s_lim, m.s_rf, _rc(m)
    nlay = m.s_lay.shape[1]
    N = np.zeros((ns, 3, 3))
    M = np.zeros((ns, 3, 3))
    crack = np.zeros(ns)
    all_open = np.ones(ns, bool)
    for g in range(3):
        mincr = np.ones(ns)
        for ll in range(nlay):
            act = ll < m.s_nl
            zb, zt, cg, sg = m.s_lay[:, ll].T
            hh = zt - zb
            cc2, ss2, cs = cg * cg, sg * sg, cg * sg
            for q, (z0, wq) in enumerate(((zb, hh / 6), ((zb + zt) / 2, 4 * hh / 6),
                                          (zt, hh / 6))):
                z = z0 * ssp  # a split plate's sub-laminates bend about their own planes
                e1, e2, g12 = _ply_strain(em, kap[:, g], z, cc2, ss2, cs)
                e1t, e2t, g12t = _ply_strain(emt, kapt[:, g], z, cc2, ss2, cs)
                h = st.s_hist[:, (g * nlay + ll) * 3 + q]
                if update:
                    upd = (act & ~st.s_gone)[:, None]
                    new = np.stack([np.where(e1 > 0, e1 / lim[:, 0], 0),
                                    np.where(e1 > 0, 0, -e1 / lim[:, 1]),
                                    np.where(e2 > 0, e2 / lim[:, 2], 0),
                                    np.where(e2 > 0, 0, -e2 / lim[:, 3]),
                                    np.abs(g12) / lim[:, 4]], axis=1)
                    h[:] = np.where(upd, np.maximum(h, new), h)
                dft = _soft(h[:, 0], rf[:, 0])
                dmt = _soft(h[:, 2], rf[:, 1])
                ds = _soft(h[:, 4], rf[:, 2])
                d1 = np.where(e1t >= 0, dft, _plateau(h[:, 1], rc))
                d2 = np.where(e2t >= 0, dmt, _plateau(h[:, 3], 1.0))
                dsh = 1 - (1 - ds) * (1 - dft) * (1 - dmt)
                cr = np.maximum(np.maximum(dft, dmt), ds)
                mincr = np.where(act, np.minimum(mincr, cr), mincr)
                op = np.maximum(np.maximum(h[:, 0] / rf[:, 0], h[:, 2] / rf[:, 1]),
                                h[:, 4] / rf[:, 2])
                all_open &= ~act | (op >= 2.0)
                k1, k2 = 1 - d1, 1 - d2
                s1 = k1 * q11 * e1t + k1 * k2 * q12 * e2t
                s2 = k1 * k2 * q12 * e1t + k2 * q22 * e2t
                t12 = (1 - dsh) * q66 * g12t
                sig = np.stack([s1 * cc2 + s2 * ss2 - 2 * t12 * cs,
                                s1 * ss2 + s2 * cc2 + 2 * t12 * cs,
                                (s1 - s2) * cs + t12 * (cc2 - ss2)], axis=1)
                wa = np.where(act, wq, 0.0)[:, None]
                N[:, g] += wa * sig
                M[:, g] += (wa * z[:, None]) * sig
        crack = np.maximum(crack, mincr)
    if update:
        st.s_dmg = np.where(st.s_gone, st.s_dmg, np.maximum(st.s_dmg, crack))
        newly = all_open & (m.s_nl > 0) & ~st.s_gone
    f = np.zeros((ns, 18))
    f[:, _IM] = m.s_area[:, None] * np.einsum("nik,ni->nk", m.s_Bm, N.mean(axis=1))
    f[:, _IB] = (m.s_area / 3)[:, None] * np.einsum("ngik,ngi->nk", m.s_Bb, M)
    if shear.any():  # DKMT: R^T om (1.5 A/3 sum_g KH_g^T M_g + A Gs^T Q)
        coef = om * (1.5 * (m.s_area / 3)[:, None] * np.einsum("ngik,ngi->nk", sKH, M)
                     + m.s_area[:, None] * np.einsum("nik,ni->nk", sGs, Qs))
        f[:, _IB] += np.einsum("nkj,nk->nj", sR, coef)
    # drilling (Hughes-Brezzi): theta_z against the in-plane rigid rotation omega
    g = np.zeros((ns, 6))
    g[:, 0::2] = -0.5 * m.s_Bm[:, 1, 1::2]
    g[:, 1::2] = 0.5 * m.s_Bm[:, 0, 0::2]
    om = (g * dd[:, _IM]).sum(1)
    rel = dd[:, _IR] - om[:, None]
    kde = m.s_kd * np.maximum(0.0, 1.0 - 10.0 * st.s_dmg)  # a cracking element: no drilling
    f[:, _IR] = kde[:, None] * rel
    f[:, _IM] -= (kde * rel.sum(1))[:, None] * g
    f[st.s_gone] = 0.0
    if update:
        st.s_gone = st.s_gone | newly
    f = f.reshape(ns, 3, 6)
    fg = np.einsum("nij,naj->nai", E, f[:, :, :3])
    mg = np.einsum("nij,naj->nai", E, f[:, :, 3:])
    np.add.at(F, sn.ravel(), fg.reshape(-1, 3))
    np.add.at(Mo, sn.ravel(), mg.reshape(-1, 3))


def glue_creep(m: Model, st: State) -> tuple[float, float]:
    """The creep of the glue lines for this step: (phi / (1 + phi), exp(-(1 + phi)
    dt_test / t_r)) for a Kelvin element of final creep phi x the elastic slip and
    retardation time t_r (glue.creep_factor, glue.creep_time_s), over the physical test
    time of the step (``st.creep_dt``)."""
    phi = float(m.params.get("glue_creep_factor", 0.0))
    tr = float(m.params.get("glue_creep_time_s", 600.0))
    if phi <= 0.0 or tr <= 0.0:
        return 0.0, 1.0
    return phi / (1.0 + phi), math.exp(-(1.0 + phi) * max(st.creep_dt, 0.0) / tr)


def glue_shape(m: Model) -> tuple[float, float]:
    """The glue law's pre-peak shape: (onset, peak slip ratio), glue.prepeak_onset and
    glue.peak_slip_ratio; (1, 1) is the bilinear law."""
    return (float(m.params.get("glue_prepeak_onset", 1.0)),
            float(m.params.get("glue_peak_slip_ratio", 1.0)))


def trilinear_damage(dmx, dm0, keff, Gc, ga: float, gp: float) -> np.ndarray:
    """The glue law of kernels._trilinear_damage, over arrays: the damage at the largest
    separation ``dmx`` reached (elastic to ga x the strength, bending over to the
    strength at gp dm0, linear softening to nothing where the work done is Gc)."""
    dmx, dm0 = np.asarray(dmx, float), np.asarray(dm0, float)
    e, q = ga * dm0, gp * dm0
    lf = np.maximum(gp + 2.0 * Gc / (keff * dm0 * dm0) - ga * ga - (ga + 1.0) * (gp - ga),
                    1.05 * gp)
    dmf = lf * dm0
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(dmx <= q, dm0 * (ga + (1.0 - ga) * (dmx - e) / np.maximum(q - e, 1e-30)),
                     dm0 * (dmf - dmx) / np.maximum(dmf - q, 1e-30))
        d = np.where(dmx <= e, 0.0, np.where(dmx >= dmf, 1.0,
                                             1.0 - T / np.maximum(dmx, 1e-30)))
    return np.clip(d, 0.0, 1.0)


RATE_LOAD = 0.1  # share of its strength at which a glue line's own clock starts


def glue_rate(m: Model, st: State) -> tuple[np.ndarray, float, float, np.ndarray]:
    """The glue points' rate exponents, the physical test time so far, t_ref and each
    point's start time: a glue line's strength depends on its time to failure, tau(t) =
    tau(t_ref) (t / t_ref)^-n (n 0.1: +26 % per decade faster; PVAc +20 % at 16x the rate,
    Wernersson 1994; -40 % at a tenth of the rate, Serrano 2000), t_ref the time to failure
    of the lap tests tau_g describes (glue.coupon_time_s), t the line's own time under
    load: from when it first took RATE_LOAD of its strength (at least t_ref / RATE_RANGE,
    3.75 s: no faster than 16x its coupons, the law's data), so a line that
    picks up load when its neighbour breaks is loaded fast, not as old as the test (the
    realism and adversarial reviews of round 21-22). Before any physical time has passed
    (the screen) tau_g; a coupon (lab) has every line loaded from its start."""
    nc = len(m.c_a)
    rate = m.c_rate if len(m.c_rate) == nc else np.zeros(nc)
    tref = float(m.params.get("glue_coupon_time_s", 60.0))
    if len(st.c_t0) != nc:
        st.c_t0 = np.full(nc, -1.0)
    return rate, float(st.t_test), tref, st.c_t0


def glue_rate_factor(rate, tnow: float, tref: float, t0: np.ndarray) -> np.ndarray:
    """Each glue point's strength factor at its own time under load (glue_rate), taken
    no faster than kernels.RATE_RANGE x the coupons' time (the law's data)."""
    from bridgesim.sim.kernels import RATE_RANGE

    if tnow <= 0.0 or tref <= 0.0:
        return np.ones(len(t0))
    tl = np.where(t0 >= 0.0, tnow - t0, 0.0)
    return np.exp(-rate * np.log(np.maximum(tl, tref / RATE_RANGE) / tref))


def glue_forces(m: Model, st: State, F: np.ndarray, Mo: np.ndarray,
                update: bool = True, beta: float = 0.0, dt: float = 1.0) -> None:
    if len(m.c_a) == 0:
        return
    pa, arm_a = _attach_pos(m, st, m.c_a)
    pb, arm_b = _attach_pos(m, st, m.c_b)
    na, nb = m.a_node[m.c_a], m.a_node[m.c_b]
    nv = np.einsum("nij,nj->ni", st.R[na] + st.R[nb], m.c_n0)
    nn = np.linalg.norm(nv, axis=1)
    n = nv / nn[:, None]
    delta = pb - pa
    dn = (delta * n).sum(1)
    ds = delta - dn[:, None] * n
    # creep of the glue line in shear (kernels.glue_kernel): the elastic slip carries
    # the stress and drives the damage
    cfac, cdec = glue_creep(m, st)
    es = ds - st.c_creep
    es = es - (es * n).sum(1)[:, None] * n
    s = np.linalg.norm(es, axis=1)
    Kn, Ks = m.c_Kn, m.c_Ks
    # the wood under a glue line crushes (permanent set), hardening as it compacts and
    # densified past eps_d of its depth (see kernels.glue_kernel)
    fc90 = _glue_fc90(m)
    cdep, eps_d, E_d, fcap = _glue_crush(m)
    dn = dn - st.c_crush
    q = -st.c_crush / cdep
    lo = q < eps_d
    fy = np.where(lo, fc90 * (1 + q), fc90 * (1 + eps_d) + E_d * (q - eps_d))
    hh = np.where(lo, fc90 / cdep, E_d / cdep)
    capped = fy > fcap
    fy = np.where(capped, fcap, fy)
    hh = np.where(capped, 0.0, hh)
    over = Kn * dn < -fy
    dp = np.where(over, (-Kn * dn - fy) / (Kn + hh), 0.0)
    if update:
        st.c_crush = st.c_crush - dp
    dn = dn + dp
    dnp = np.maximum(dn, 0.0)
    rate, tnow, tref, t0 = glue_rate(m, st)
    dm = np.sqrt(dnp**2 + s**2)
    if update and tnow > 0.0:
        # each line's clock starts at RATE_LOAD of its mixed-mode onset
        dn0b, ds0b = m.c_sig / Kn, m.c_tau / Ks
        mixb = s / np.maximum(dnp, 1e-12)
        dm0b = np.where(dnp > 1e-12, dn0b * ds0b * np.sqrt(
            (1 + mixb**2) / (ds0b**2 + (mixb * dn0b) ** 2)), ds0b)
        on = (t0 < 0.0) & (dm > RATE_LOAD * dm0b)
        st.c_t0 = t0 = np.where(on, tnow, t0)
    rs = glue_rate_factor(rate, tnow, tref, t0)
    dn0 = m.c_sig * rs / Kn
    ds0 = m.c_tau * rs / Ks
    mix = s / np.maximum(dnp, 1e-12)
    dm0 = np.where(dnp > 1e-12, dn0 * ds0 * np.sqrt((1 + mix**2)
                                                   / (ds0**2 + (mix * dn0) ** 2)), ds0)
    Gs = Ks * s**2
    Gn = Kn * dnp**2
    B = Gs / np.maximum(Gs + Gn, 1e-30)
    Gc = m.c_GIc + (m.c_GIIc - m.c_GIc) * B**2
    Keff = np.where(dm > 1e-12, (Gn + Gs) / np.maximum(dm**2, 1e-30), Ks)
    dmax = np.maximum(st.c_dmax, dm)
    d = trilinear_damage(dmax, np.maximum(dm0, 1e-12), Keff, Gc, *glue_shape(m))
    d = np.where(m.c_breakable, np.maximum(st.c_d, d), 0.0)
    if update:
        st.c_dmax = np.where(m.c_breakable, dmax, 0.0)
        st.c_d = d
        if cfac > 0.0:
            go = (m.c_breakable & (d < 0.999))[:, None]
            st.c_creep = np.where(go, cfac * ds + (st.c_creep - cfac * ds) * cdec,
                                  st.c_creep)
    # a debonded pair that has turned, parted or passed through stops acting as contact
    gone = m.c_breakable & (d >= 0.999) & ((nn < 1.6) | (st.c_crush < -cdep)
                                           | (dn**2 + s**2 > 9.0))
    live = st.c_alive & ~gone
    if update:
        st.c_alive = live
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
    ts = (1 - d)[:, None] * Ks[:, None] * es + d[:, None] * tf
    t = tn[:, None] * n + ts
    f = t * m.c_area[:, None] * live[:, None]  # force pulling B back towards A
    if beta > 0:
        rate = (delta - st.c_prev) / dt
        rn = (rate * n).sum(1)
        kk = np.where(dn <= 0, Kn, (1 - d) * Kn)
        f = f + beta * (kk[:, None] * rn[:, None] * n
                        + ((1 - d) * Ks)[:, None] * (rate - rn[:, None] * n)) \
            * (m.c_area * live)[:, None]
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
    pen = m.table_y - p[:, 1] - _lift(m)
    touch = (pen > 0) & ~st.loose[m.a_node[m.p_att]]
    nd = m.a_node[m.p_att]
    vy = st.v[nd, 1] + st.w[nd, 2] * arm[:, 0] - st.w[nd, 0] * arm[:, 2]
    fe = seated(k, np.maximum(pen, 0.0), _seat(m))
    fn = np.where(touch, np.maximum(np.minimum(fe - cdamp * vy, _fmax(m, "p")), 0.0), 0.0)
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
    fe = seated(k, np.maximum(pen, 0.0), _seat(m))
    fn = np.where(inside, np.maximum(np.minimum(fe - cdamp * vr, _fmax(m, "q")), 0.0), 0.0)
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


def split_averaging(m: Model):
    """The averaging of the split check (cached on the model): each beam element's
    shear stress over strength is averaged over a window of one section depth (the larger
    of its width and its thickness) centred on it, each element of the same body weighted
    by the length of it inside the window, so that a split starts where the shear acts
    over a member's depth, not where one element resolves a peak next to a glued joint
    (a local onset would come earlier the finer the mesh). The window is the same on
    every mesh whose elements are no longer than a section depth; a longer element
    averages over itself. Returns a sparse averaging matrix."""
    W = getattr(m, "_split_W", None)
    if W is not None and W.shape[0] == len(m.b_i):
        return W
    from scipy.sparse import csr_matrix

    nb = len(m.b_i)
    c = 0.5 * (m.X[m.b_i] + m.X[m.b_j])
    rows, cols, vals = [], [], []
    for bd in np.unique(m.b_body):
        idx = np.where(m.b_body == bd)[0]
        sec = m.bodies[bd].section_mm if 0 <= bd < len(m.bodies) else None
        depth = float(max(sec)) if sec and max(sec) > 0 else \
            2.0 * float(np.abs(np.r_[m.b_fy[idx].ravel(), m.b_fz[idx].ravel()]).max())
        d = np.linalg.norm(c[idx, None, :] - c[None, idx, :], axis=2)
        r = 0.5 * depth
        half = 0.5 * m.b_L0[idx]
        for a, e in enumerate(idx):
            # the length of each element inside [-r, r] around this one's centre
            ov = np.clip(np.minimum(d[a] + half, r) - np.maximum(d[a] - half, -r), 0.0, None)
            ov[a] = max(ov[a], 1e-9)
            near = np.where(ov > 0)[0]
            rows += [e] * len(near)
            cols += list(idx[near])
            vals += list(ov[near] / ov[near].sum())
    W = csr_matrix((vals, (rows, cols)), shape=(nb, nb))
    m._split_W = W
    return W


def split_members(m: Model, st: State) -> None:
    """Members split in shear where their averaged shear (``split_averaging``) reaches
    the strength: through the plies the member splits along its most stressed glue line
    into two halves (lever arms x 1/2, bending stiffness x 1/4), and again in each half
    while the shear, which falls with the stiffness, still reaches the strength, down to
    its separate plies (1/n: until round 22 at once, which left a 13-ply member 1/169 of
    its stiffness at the first split, the adversarial review of round 21); across the
    width the wood splits in two. Either way it keeps its axial strength."""
    W = split_averaging(m)
    r3, r2 = W @ st.b_rat3, W @ st.b_rat2
    alive = st.b_alive
    last = 1.0 / m.b_nply
    f3 = alive & (st.b_s3 > last * (1.0 + 1e-9)) & (r3 > 1.0)
    f2 = alive & (st.b_s2 == 1.0) & (r2 > 1.0)
    if f3.any() or f2.any():
        st.b_s3 = np.where(f3, np.maximum(0.5 * st.b_s3, last), st.b_s3)
        st.b_s2 = np.where(f2, 0.5, st.b_s2)
        st.b_shear |= f2 | f3


def plate_split_averaging(m: Model):
    """The averaging of the plates' split check (cached on the model), as
    :func:`split_averaging` does for members: each shell element's interlaminar shear
    over strength is averaged over a disc one plate thickness across (radius t / 2)
    centred on it, each element of the same plate weighted by its area inside the disc
    (4 x 4 sub-triangles each), at least the element itself: a plate splits where the
    shear acts over its thickness, not where one small element resolves a peak at the
    edge of a patch load (a local onset would come earlier the finer the mesh). With
    elements larger than the thickness (the usual case) it is about the element itself.
    Returns a sparse averaging matrix."""
    ns = len(m.s_nodes)
    W = getattr(m, "_psplit_W", None)
    if W is not None and W.shape[0] == ns:
        return W
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    P = m.X[m.s_nodes]  # (ns, 3, 3)
    c = P.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    k = 4  # sub-triangles' centroids of a k x k subdivision, in area coordinates
    bary = []
    for i in range(k):
        for j in range(k - i):
            bary.append(((i + 1 / 3), (j + 1 / 3), (k - i - j - 2 / 3)))
            if i + j < k - 1:
                bary.append(((i + 2 / 3), (j + 2 / 3), (k - i - j - 4 / 3)))
    bary = np.array(bary) / k  # (k^2, 3)
    pts = np.einsum("qa,nak->nqk", bary, P)
    nq = len(bary)
    rows, cols, vals = [], [], []
    for bd in np.unique(m.s_body):
        idx = np.where(m.s_body == bd)[0]
        tree = cKDTree(pts[idx].reshape(-1, 3))
        owner = np.repeat(np.arange(len(idx)), nq)
        hits = tree.query_ball_point(c[idx], 0.5 * m.s_t[idx])
        for a, (e, hit) in enumerate(zip(idx, hits, strict=True)):
            ov = np.bincount(owner[np.asarray(hit, int)], minlength=len(idx)) \
                * area[idx] / nq
            ov[a] = max(ov[a], 1e-9)
            near = np.nonzero(ov)[0]
            rows += [e] * len(near)
            cols += list(idx[near])
            vals += list(ov[near] / ov[near].sum())
    W = csr_matrix((vals, (rows, cols)), shape=(ns, ns))
    m._psplit_W = W
    return W


def split_plates(m: Model, st: State) -> None:
    """Plates split between their plies in out-of-plane shear, as members do
    (:func:`split_members`): where the interlaminar shear 1.5 |Q| / t, averaged over a
    plate thickness (:func:`plate_split_averaging`), reaches the strength of the glue
    planes, the element splits along its most stressed one into two sub-laminates that
    bend each about its own mid-plane (lever arms x 1/2, bending stiffness x 1/4), at
    once (brittle, as a pressed line in a short-beam shear test), and again in each part
    while the shear still reaches it, down to its separate plies (1/n). It keeps its
    membrane stiffness and strength and its transverse shear stiffness (the parts share
    the shear, each with its own: kappa G t in all). A single ply has no glue plane."""
    if len(m.s_nodes) == 0 or len(st.s_split) != len(m.s_nodes) or \
            len(st.s_rat) != len(m.s_nodes) or not st.s_rat.max() > 1.0:
        return  # (a mean of values none above 1 is not above 1)
    r = plate_split_averaging(m) @ st.s_rat
    last = 1.0 / np.maximum(m.s_nply, 1.0)
    f = ~st.s_gone & (st.s_split > last * (1.0 + 1e-9)) & (r > 1.0)
    if f.any():
        st.s_split = np.where(f, np.maximum(0.5 * st.s_split, last), st.s_split)
        st.s_shear = st.s_shear | f


def internal_and_contact(m: Model, st: State, s: Settings, F: np.ndarray, Mo: np.ndarray,
                         update: bool = True) -> dict:
    if s.use_numba and K.HAVE_NUMBA:
        out = _internal_fast(m, st, s, F, Mo, update)
    else:
        out = internal_and_contact_numpy(m, st, s, F, Mo, update)
    if st.sc is not None:
        from bridgesim.sim import contact

        contact.forces(m, st, st.sc, F, Mo, s.dt, update)
    return out


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


def _glue_crush(m: Model) -> tuple[np.ndarray, float, float, float]:
    """Depth of wood that can crush at each glue point (mm), the strain at which crushed
    wood is densified, its stiffness then (MPa), and the most any face bears (MPa: the
    wood's crushing strength along the grain, which neither end grain nor densified wood
    across the grain exceeds)."""
    d = m.params.get("glue_cdep")
    if d is None or len(d) != len(m.c_a):
        d = np.full(len(m.c_a), 1e30)
        m.params["glue_cdep"] = d
    return d, float(m.params.get("densification_strain", 0.45)), \
        float(m.params.get("densified_E", 500.0)), \
        float(m.params.get("bearing_cap", 1e30))


def _glue_fc90(m: Model) -> np.ndarray:
    """Crushing strength of the wood at each glue point (MPa), by grain direction."""
    f = m.params.get("glue_fc")
    if f is None or len(f) != len(m.c_a):
        f = np.full(len(m.c_a), 1e30)
        m.params["glue_fc"] = f
    return f


def seated(k, pen: np.ndarray, seat: float) -> np.ndarray:
    """Bearing-face normal force with seating (see kernels._seated)."""
    if seat <= 0:
        return k * pen
    return np.where(pen < seat, 0.5 * k * pen**2 / seat, k * (pen - 0.5 * seat))


def _seat(m: Model) -> float:
    return float(m.params.get("seating_mm", 0.0))


def _lift(m: Model) -> np.ndarray:
    """How far each platform contact point sits above the platform when the bridge is
    level: the support corners of a built bridge are not quite in one plane."""
    v = m.params.get("p_lift")
    if v is None or len(v) != len(m.p_att):
        v = np.zeros(len(m.p_att))
        m.params["p_lift"] = v
    return v


def _fmax(m: Model, key: str) -> np.ndarray:
    """Bearing capacity of each platform ("p") or plate ("q") contact point: crushing
    strength of the wood under it x the area the point stands for."""
    n = len(m.p_att) if key == "p" else len(m.q_att)
    f = m.params.get(f"fmax_{key}")
    if f is None or len(f) != n:
        f = np.full(n, np.inf)
        m.params[f"fmax_{key}"] = f
    return f


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
                      m.b_ft, m.b_fc, m.b_Gf, m.b_GJ, m.b_A, m.b_fv, m.b_fv3, m.b_Wt,
                      m.b_nply, m.b_kGA, st.b_gam, st.b_s2, st.b_s3,
                      float(m.params.get("crush_softening", 0.3)), st.eps_p, st.dmg_t,
                      st.b_alive, st.b_fractured, st.b_shear, F, Mo, update,
                      s.beta if update else 0.0, s.dt, st.b_prev, st.b_rat2, st.b_rat3,
                      float(m.params.get("fc_proportional", 1.0)))
        if update:
            split_members(m, st)
    if len(m.s_nodes):
        ns = len(m.s_nodes)
        if st.work is None or st.work[0].shape[0] != ns:
            st.work = (np.zeros((ns, 3, 3)), np.zeros((ns, 3, 3)), np.zeros((ns, 18)),
                       np.zeros((ns, 3, 3)), np.zeros((ns, 3)), np.zeros((ns, 72)))
        fe, me, wd, wR, wth, ws = st.work
        if len(st.s_split) != ns:
            st.s_split, st.s_rat = np.ones(ns), np.zeros(ns)
            st.s_shear = np.zeros(ns, bool)
        K.shell_kernel(x, R, m.s_nodes, m.s_E0, m.s_P, m.s_Bm, m.s_Bb, m.s_area, m.s_kd,
                       m.s_lay, m.s_nl, m.s_Q, m.s_lim, m.s_rf, _rc(m), st.s_hist,
                       st.s_dmg, st.s_gone, F, Mo, update, fe, me,
                       s.beta if update else 0.0, s.dt, st.s_prev, wd, wR, wth, ws,
                       *shell_shear_arrays(m), st.s_split, st.s_rat)
        if update:
            split_plates(m, st)
    if len(m.c_a):
        K.glue_kernel(x, R, m.a_node, m.a_r0, m.c_a, m.c_b, m.c_area, m.c_n0, m.c_Kn,
                      m.c_Ks, m.c_sig, m.c_tau, m.c_GIc, m.c_GIIc, m.c_mu, m.c_breakable,
                      st.c_dmax, st.c_d, st.c_slip, F, Mo, update,
                      s.beta if update else 0.0, s.dt, st.c_prev, st.c_alive, st.c_crush,
                      _glue_fc90(m), *_glue_crush(m), st.c_creep, *glue_creep(m, st),
                      *glue_shape(m), *glue_rate(m, st))
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
                      _cd(m, "p", s, update), st.loose, _fmax(m, "p"), _lift(m), _seat(m))
    plate_load = 0.0
    if len(m.q_att) and m.plate_node >= 0:
        plate_load = K.plate_kernel(x, R, m.a_node, m.a_r0, m.q_att, m.plate_node,
                                    m.plate_half, _kc(m, "q", s), s.mu_plate, st.q_anchor,
                                    st.q_touch, Fc, Mc, update, st.v, st.w,
                                    _cd(m, "q", s, update), st.loose, _fmax(m, "q"),
                                    _seat(m))
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
    if update and len(m.b_i):
        split_members(m, st)
    shell_forces(m, st, F, Mo, update, b, s.dt)
    if update:
        split_plates(m, st)
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
