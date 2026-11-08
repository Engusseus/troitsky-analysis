"""Linear pre-screen: the bridge's elastic response to the plate load, in seconds.

The structural model the crush simulates (members, plates, glue springs, floss, rigid
blocks, the platform and the loading plate) is assembled into one sparse linear
stiffness matrix and solved once for a reference load on the plate: no damage, no
contact changes, no imperfections, small displacements. From that solution:

* the bridge's stiffness under the plate (load / plate displacement);
* every glued joint's load against its capacity (normal and shear, the joint's resultant
  over its glued area), every member's fibre stress against its strength in tension or
  compression, its shear against its shear strength and its axial force against its
  Euler load (the member's own length, pinned ends), every plate's ply strains against
  their limits;
* each one's utilisation, and the load at which it would reach its limit if the bridge
  stayed linear.

It ranks the weak spots and compares design variants in seconds. It does not predict
F_u: the real bridge redistributes load after the first joint lets go, softens, and has
imperfections (crush does all of that); the first limit it gives is usually below F_u.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bridgesim.sim.model import Model

KGF = 9.80665


@dataclass
class Limit:
    part: str  # a body, or 'A | B' for a glued joint
    mode: str  # "glue", "bearing", "tension", "compression", "shear", "buckling", "plate ..."
    utilisation: float  # at the reference load
    load_kgf: float  # plate load at which it reaches its limit (linear)
    where_mm: tuple[float, float, float]


def _skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _att_T(m: Model, att: int) -> tuple[int, np.ndarray]:
    """Node and 3 x 6 matrix of an attachment: its displacement from the node's."""
    nd = int(m.a_node[att])
    T = np.zeros((3, 6))
    T[:, :3] = np.eye(3)
    T[:, 3:] = -_skew(m.a_r0[att])  # u + theta x r = u - r x theta
    return nd, T


class _Assembler:
    def __init__(self, n: int):
        self.n = n
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []

    def add(self, dofs: np.ndarray, K: np.ndarray) -> None:
        d = np.asarray(dofs)
        self.rows.append(np.repeat(d, len(d)))
        self.cols.append(np.tile(d, len(d)))
        self.vals.append(np.asarray(K).ravel())

    def spring(self, m: Model, a: int, b: int | None, k3: np.ndarray,
               b_node: int | None = None, b_r: np.ndarray | None = None) -> None:
        """A 3 x 3 spring between attachment ``a`` and attachment ``b`` (or a point at
        offset ``b_r`` from node ``b_node``)."""
        na, Ta = _att_T(m, a)
        if b is not None:
            nb, Tb = _att_T(m, b)
        else:
            nb = int(b_node)
            Tb = np.zeros((3, 6))
            Tb[:, :3] = np.eye(3)
            Tb[:, 3:] = -_skew(b_r)
        B = np.hstack([-Ta, Tb])  # relative displacement b - a
        dofs = np.r_[6 * na + np.arange(6), 6 * nb + np.arange(6)]
        self.add(dofs, B.T @ k3 @ B)

    def matrix(self):
        from scipy.sparse import coo_matrix

        N = 6 * self.n
        return coo_matrix((np.concatenate(self.vals), (np.concatenate(self.rows),
                                                       np.concatenate(self.cols))),
                          shape=(N, N)).tocsr()


def _beam_K(m: Model) -> np.ndarray:
    """Linear 12 x 12 local stiffness of every beam, Timoshenko (shear-flexible)."""
    nb = len(m.b_i)
    A = m.b_fA.sum(1)
    Iz = (m.b_fA * m.b_fy**2).sum(1)
    Iy = (m.b_fA * m.b_fz**2).sum(1)
    L, E = m.b_L0, m.b_E
    K = np.zeros((nb, 12, 12))

    def put(i, j, v):
        K[:, i, j] += v
        if i != j:
            K[:, j, i] += v

    ka = E * A / L
    put(0, 0, ka); put(6, 6, ka); put(0, 6, -ka)  # noqa: E702
    kt = m.b_GJ / L
    put(3, 3, kt); put(9, 9, kt); put(3, 9, -kt)  # noqa: E702
    for I_, (v, r, sg) in ((Iz, (1, 5, 1.0)), (Iy, (2, 4, -1.0))):
        EI = E * I_
        phi = np.where(m.b_kGA > 0, 12 * EI / np.maximum(m.b_kGA, 1e-30) / L**2, 0.0)
        a = 12 * EI / L**3 / (1 + phi)
        b = 6 * EI / L**2 / (1 + phi) * sg
        c = (4 + phi) * EI / L / (1 + phi)
        d = (2 - phi) * EI / L / (1 + phi)
        put(v, v, a); put(v + 6, v + 6, a); put(v, v + 6, -a)  # noqa: E702
        put(v, r, b); put(v, r + 6, b); put(v + 6, r, -b); put(v + 6, r + 6, -b)  # noqa: E702
        put(r, r, c); put(r + 6, r + 6, c); put(r, r + 6, d)  # noqa: E702
    return K


def _unglued(m: Model) -> np.ndarray:
    """The glue points of faces that touch but are not glued (glue_factor 0: the crush
    starts them broken, bearing and sliding with friction only)."""
    d0 = m.params.get("c_d0")
    if d0 is None or len(d0) != len(m.c_a):
        return np.zeros(len(m.c_a), bool)
    return np.asarray(d0) >= 1.0


def _block(Rm: np.ndarray, k: int) -> np.ndarray:
    """Block-diagonal matrix of k copies of the 3 x 3 rotation Rm (global -> local)."""
    T = np.zeros((3 * k, 3 * k))
    for i in range(k):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Rm
    return T


def solve(m: Model, load_N: float, k_contact: float = 2000.0,
          tie_tol_mm: float = 1.0) -> dict:
    """Assemble and solve. Returns the nodal displacements (n, 6), the plate's
    displacement, and the pieces needed to evaluate utilisations."""

    n = m.n_nodes
    asm = _Assembler(n)
    Kb = _beam_K(m)
    beamT = []
    for e in range(len(m.b_i)):
        T = _block(m.b_E0[e].T, 4)
        dofs = np.r_[6 * m.b_i[e] + np.arange(6), 6 * m.b_j[e] + np.arange(6)]
        asm.add(dofs, T.T @ Kb[e] @ T)
        beamT.append(T)
    for e in range(len(m.s_nodes)):
        T = _block(m.s_E0[e].T, 6)
        dofs = np.concatenate([6 * nd + np.arange(6) for nd in m.s_nodes[e]])
        asm.add(dofs, T.T @ m.s_K[e] @ T)
    unglued = _unglued(m)
    for i in range(len(m.c_a)):
        if unglued[i]:  # touching, not glued: bears in compression only (below)
            continue
        nrm = m.c_n0[i] / max(np.linalg.norm(m.c_n0[i]), 1e-12)
        P = np.outer(nrm, nrm)
        k3 = m.c_area[i] * (m.c_Kn[i] * P + m.c_Ks[i] * (np.eye(3) - P))
        asm.spring(m, int(m.c_a[i]), int(m.c_b[i]), k3)
    for i in range(len(m.k_a)):
        na, nb = int(m.a_node[m.k_a[i]]), int(m.a_node[m.k_b[i]])
        pa = m.X[na] + m.a_r0[m.k_a[i]]
        pb = m.X[nb] + m.a_r0[m.k_b[i]]
        d = pb - pa
        Ld = np.linalg.norm(d)
        if Ld > 1e-9:
            t = d / Ld
            asm.spring(m, int(m.k_a[i]), int(m.k_b[i]),
                       m.k_EA[i] / max(m.k_L0[i], 1e-9) * np.outer(t, t))
    # contacts, compression only (solved again without the ones that would pull, until
    # none does): the points resting on the platform, held in every direction while they
    # press on it (friction sticks), unless the bridge's thrust at one end is more than
    # friction holds (contact.friction_platform x the reaction there): that end slides
    # along the span (pinned at one end, a roller at the other; held everywhere a truss
    # read as an arch, 42 % too stiff on a light truss); and the loading plate, a rigid
    # body bearing on the deck points under it, which tilts freely on the jack's
    # hemisphere (the jack holds it across and against turning)
    contacts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    sup = np.where(m.p_support)[0] if len(m.p_support) else np.zeros(0, int)
    sup_T = []
    for i in sup:
        att = int(m.p_att[i])
        nd = int(m.a_node[att])
        _, T = _att_T(m, att)
        dofs = 6 * nd + np.arange(6)
        sup_T.append(T)
        # opening = the point rising off the platform
        contacts.append((dofs, T.T @ (k_contact * np.eye(3)) @ T, T[1].copy(), 1.0))
    sup_x = np.array([float((m.X[m.a_node[m.p_att[i]]] + m.a_r0[m.p_att[i]])[0])
                      for i in sup]) if len(sup) else np.zeros(0)
    mu = float(m.params.get("friction_platform", 0.2))
    # faces that touch but are not glued (glue_factor 0: a cut end, a part only resting
    # on another) bear across their faces while pressed, and hold along them as friction
    # sticks (as the platform's contacts do: a loading bar resting on two chords is held
    # in place, not left free to slide, which left the linear system all but singular
    # and a published truss's period at 5.7 s: the realism review of round 21), and
    # open freely
    for i in np.where(unglued)[0]:
        na, Ta = _att_T(m, int(m.c_a[i]))
        nb, Tb = _att_T(m, int(m.c_b[i]))
        if na == nb:
            continue
        nrm = m.c_n0[i] / max(np.linalg.norm(m.c_n0[i]), 1e-12)
        T = np.hstack([-Ta, Tb])
        B = nrm @ T  # opening: B . u > 0
        P = np.outer(nrm, nrm)
        k3 = m.c_area[i] * (m.c_Kn[i] * P + m.c_Ks[i] * (np.eye(3) - P))
        dofs = np.r_[6 * na + np.arange(6), 6 * nb + np.arange(6)]
        contacts.append((dofs, T.T @ k3 @ T, B, 1.0))
    P_ = m.plate_node
    q_on = []
    if P_ >= 0 and len(m.q_att):
        Xp = m.X[P_]
        h = m.plate_half
        pts = m.X[m.a_node[m.q_att]] + m.a_r0[m.q_att]
        loc = pts - Xp  # plate frame = global at the start
        # the deck points under the plate's footprint, at or below its underside: parts
        # above it (top struts, the top chords' bracing) are what the plate slides in
        # under, not what it rests on (the crush's plate is a box, touched from below)
        under = ((np.abs(loc[:, 0]) <= h[0] + 1e-6) & (np.abs(loc[:, 2]) <= h[2] + 1e-6)
                 & (loc[:, 1] <= -h[1] + tie_tol_mm))
        if under.any():
            top = pts[under, 1].max()
            q_on = np.where(under & (pts[:, 1] >= top - tie_tol_mm))[0]
        for i in q_on:
            att = int(m.q_att[i])
            na, Ta = _att_T(m, att)
            Tb = np.zeros((3, 6))
            Tb[:, :3] = np.eye(3)
            Tb[:, 3:] = -_skew(pts[i] - Xp)
            B = np.hstack([-Ta, Tb])[1]  # vertical gap change: plate point - deck point
            dofs = np.r_[6 * na + np.arange(6), 6 * P_ + np.arange(6)]
            contacts.append((dofs, k_contact * np.outer(B, B), B, 1.0))
        hold = np.zeros((6, 6))
        hold[0, 0] = hold[2, 2] = hold[4, 4] = 1e3 * k_contact
        asm.add(6 * P_ + np.arange(6), hold)
    K0 = asm.matrix()
    diag = K0.diagonal()
    scale = float(np.mean(np.abs(diag[diag != 0]))) if np.any(diag) else 1.0
    from scipy.sparse import diags

    K0 = K0 + diags(np.full(6 * n, 1e-9 * scale))  # loose debris, unused rotations
    f = np.zeros(6 * n)
    if P_ >= 0:
        f[6 * P_ + 1] = -load_N
    active = np.ones(len(contacts), bool)
    u = np.zeros(6 * n)
    slide = np.zeros(len(sup), bool)  # supports free to slide along the span
    for _pass in range(3):
        for k_s in range(len(sup)):
            kx = 0.0 if slide[k_s] else k_contact
            T = sup_T[k_s]
            contacts[k_s] = (contacts[k_s][0], T.T @ np.diag([kx, k_contact, k_contact]) @ T,
                             contacts[k_s][2], 1.0)
        u, active, K = _solve_contacts(K0, f, contacts, active)
        if not len(sup):
            break
        # the reactions at the supports that bear, by end of the bridge
        uu = np.array([sup_T[k_s] @ u[contacts[k_s][0]] for k_s in range(len(sup))])
        V = np.where(active[:len(sup)], -k_contact * uu[:, 1], 0.0)
        H = np.where(active[:len(sup)] & ~slide, -k_contact * uu[:, 0], 0.0)
        mid = 0.5 * (sup_x.min() + sup_x.max())
        ratio = {}
        for end, sel in (("lo", sup_x < mid), ("hi", sup_x >= mid)):
            if sel.any() and not slide[sel].all():
                ratio[end] = abs(H[sel].sum()) / max(V[sel].sum(), 1e-9)
        over = {e: r for e, r in ratio.items() if r > mu}
        if not over or len(ratio) < 2:
            break
        end = max(over, key=over.get)  # that end slides; the other keeps the bridge
        slide |= (sup_x >= mid) if end == "hi" else (sup_x < mid)
    u = u.reshape(n, 6)
    # how far the points the plate rests on went down (the bridge under the plate): the
    # plate's own drop is that, unless it rests on a line (a loading bar across two
    # chords), where it pivots freely in a linear solve and its drop means nothing
    deck_dy = 0.0
    if len(q_on):
        ys = []
        for i in q_on:
            att = int(m.q_att[i])
            nd, Ta = _att_T(m, att)
            ys.append(-(Ta @ u[nd])[1])
        deck_dy = float(np.mean(ys))
    plate_dy = float(-u[P_, 1]) if P_ >= 0 else 0.0
    if deck_dy > 0 and plate_dy > 3.0 * deck_dy:
        plate_dy = deck_dy
    return {"u": u, "beamT": beamT, "Kb": Kb, "q_on": q_on, "K": K,
            "n_contact": int(active.sum()), "sliding_supports": int(slide.sum()),
            "plate_dy": plate_dy, "deck_dy": deck_dy}


def _solve_contacts(K0, f, contacts, active):
    """Solve with the compression-only contacts (an active set, at most 8 passes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    u = np.zeros(len(f))
    K = K0
    for _ in range(8):
        rows, cols, vals = [], [], []
        for (dofs, Kc, _B, _s), on in zip(contacts, active, strict=True):
            if on:
                rows.append(np.repeat(dofs, len(dofs)))
                cols.append(np.tile(dofs, len(dofs)))
                vals.append(Kc.ravel())
        K = K0
        if rows:
            K = K0 + coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                        np.concatenate(cols))),
                                shape=K0.shape).tocsr()
        u = spsolve(K.tocsc(), f)
        opening = np.array([float(B @ u[dofs]) for dofs, _K, B, _s in contacts])
        # a platform point opens when it rises (B = its vertical displacement); a plate
        # point when the plate point moves up relative to the deck point
        new = opening <= 1e-9 * max(1.0, float(np.abs(u).max()))
        if np.array_equal(new, active):
            break
        active = new
    return u, active, K


def modes(m: Model, sol: dict, k: int = 6) -> list[dict]:
    """The lowest natural frequencies of the bridge standing on its supports with the
    loading plate on its deck (the stiffness of :func:`solve`, its contacts as they bear
    under the load; the model's lumped masses, the plate's included), and for each mode
    the parts that move most (their share of its kinetic energy): what a tap test of the
    built bridge measures (a phone's accelerometer or microphone), so its stick and glue
    stiffness and its support can be checked before the crush."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    n = m.n_nodes
    mv = np.zeros(6 * n)
    mv[0::6] = mv[1::6] = mv[2::6] = np.maximum(m.mass, 1e-12)
    rot = np.maximum(np.asarray(m.rot_inertia, float), 1e-12)
    mv[3::6] = mv[4::6] = mv[5::6] = rot
    K = sol["K"].tocsc()
    try:
        vals, vecs = eigsh(K, k=min(k + 6, 6 * n - 2), M=diags(mv), sigma=0.0, which="LM")
    except Exception:  # noqa: BLE001 - a singular or tiny model: no modes
        return []
    out = []
    names = [b.name for b in m.bodies]
    for lam, v in sorted(zip(vals, vecs.T, strict=True), key=lambda t: t[0]):
        f = float(np.sqrt(max(lam, 0.0)) / (2 * np.pi))
        u = v.reshape(n, 6)[:, :3]
        ke = mv[0::6] * (u**2).sum(1)
        if f < 1.0 or ke.sum() <= 0:  # a loose part or a rigid-body motion
            continue
        share: dict[str, float] = {}
        for i in np.argsort(ke)[::-1][:200]:
            b = int(m.node_body[i]) if i < len(m.node_body) else -1
            nm = names[b] if 0 <= b < len(names) else "loading plate"
            share[nm] = share.get(nm, 0.0) + float(ke[i] / ke.sum())
        top = sorted(share.items(), key=lambda kv: -kv[1])[:3]
        out.append({"frequency_hz": round(f, 1),
                    "moves": [f"{nm} ({sh:.0%})" for nm, sh in top]})
        if len(out) >= k:
            break
    return out


def utilisations(m: Model, sol: dict, load_N: float, ply=None,
                 peak_over_avg: float = 1.065) -> list[Limit]:
    """Every joint's, member's and plate's utilisation at ``load_N``."""
    from bridgesim.sim import shell as shellmod

    u = sol["u"]
    out: list[Limit] = []
    lim_load = load_N / KGF

    def add(part, mode, util, where):
        if util > 1e-9 and math.isfinite(util):
            out.append(Limit(part, mode, float(util), float(lim_load / util),
                             tuple(round(float(v), 1) for v in where)))

    # glued joints: the resultant over the joint against its capacity
    if len(m.c_a):
        fN = np.zeros(len(m.patches))
        fT = np.zeros((len(m.patches), 3))
        capN = np.zeros(len(m.patches))
        capT = np.zeros(len(m.patches))
        # bearing: the wood under a face pressed together (glued or dry) starting to
        # crush where its pressure first reaches what the grain there takes (Hankinson,
        # glue_fc): the peak over the face's glue points, as a face pressed off-centre
        # crushes at its edge first
        uB = np.zeros(len(m.patches))
        fc = np.asarray(m.params.get("glue_fc", np.zeros(len(m.c_a))), float)
        if len(fc) != len(m.c_a):
            fc = np.zeros(len(m.c_a))
        unglued = _unglued(m)
        for i in range(len(m.c_a)):
            if not m.c_breakable[i]:
                continue
            ua, ub = (u[m.a_node[a], :3] + np.cross(u[m.a_node[a], 3:], m.a_r0[a])
                      for a in (int(m.c_a[i]), int(m.c_b[i])))
            d = ub - ua
            nrm = m.c_n0[i] / max(np.linalg.norm(m.c_n0[i]), 1e-12)
            dn = float(d @ nrm)
            p = int(m.c_patch[i])
            if fc[i] > 0:
                uB[p] = max(uB[p], m.c_Kn[i] * max(-dn, 0.0) / fc[i])
            if unglued[i]:
                continue
            dt = d - dn * nrm
            fN[p] += m.c_area[i] * m.c_Kn[i] * dn
            fT[p] += m.c_area[i] * m.c_Ks[i] * dt
            capN[p] += m.c_area[i] * m.c_sig[i] / peak_over_avg
            capT[p] += m.c_area[i] * m.c_tau[i] / peak_over_avg
        for p, pa in enumerate(m.patches):
            if capT[p] <= 0:
                continue
            ut = math.hypot(max(fN[p], 0.0) / max(capN[p], 1e-9),
                            float(np.linalg.norm(fT[p])) / capT[p])
            add(f"{m.bodies[pa.body_a].name} | {m.bodies[pa.body_b].name}",
                "wood" if pa.kind == "solid" else "glue", ut, pa.centroid)
        for p, pa in enumerate(m.patches):
            if uB[p] > 0:
                add(f"{m.bodies[pa.body_a].name} | {m.bodies[pa.body_b].name}", "bearing",
                    uB[p], pa.centroid)
    # members: fibre stresses, shear, buckling over the member's length
    if len(m.b_i):
        A = m.b_fA.sum(1)
        Iz = (m.b_fA * m.b_fy**2).sum(1)
        Iy = (m.b_fA * m.b_fz**2).sum(1)
        # extreme fibre of the rectangular section (the fibres sit inside it, placed to
        # give its exact second moment: I = A c^2 / 3 with c the half depth)
        ymax = np.sqrt(3 * Iz / np.maximum(A, 1e-12))
        zmax = np.sqrt(3 * Iy / np.maximum(A, 1e-12))
        worst: dict[int, tuple] = {}
        Nmin: dict[int, float] = {}
        Nel = np.zeros(len(m.b_i))
        for e in range(len(m.b_i)):
            dl = sol["beamT"][e] @ np.r_[u[m.b_i[e]], u[m.b_j[e]]]
            fl = sol["Kb"][e] @ dl
            N = fl[6]  # tension positive at the far end
            Mz = max(abs(fl[5]), abs(fl[11]))
            My = max(abs(fl[4]), abs(fl[10]))
            V = max(math.hypot(fl[1], fl[2]), math.hypot(fl[7], fl[8]))
            sb = My * zmax[e] / max(Iy[e], 1e-9) + Mz * ymax[e] / max(Iz[e], 1e-9)
            ut_t = (N / A[e] + sb) / max(m.b_ft[e], 1e-9)
            ut_c = (-N / A[e] + sb) / max(m.b_fc[e], 1e-9)
            ut_v = 1.5 * V / A[e] / max(min(m.b_fv[e], m.b_fv3[e]), 1e-9)
            bd = int(m.b_body[e])
            mid = 0.5 * (m.X[m.b_i[e]] + m.X[m.b_j[e]])
            for mode, ut in (("tension", ut_t), ("compression", ut_c), ("shear", ut_v)):
                if ut > worst.get((bd, mode), (0,))[0]:
                    worst[(bd, mode)] = (ut, mid)
            Nmin[bd] = min(Nmin.get(bd, 0.0), N)
            Nel[e] = N
        for (bd, mode), (ut, mid) in worst.items():
            add(m.bodies[bd].name, mode, ut, mid)
        for bd, N in Nmin.items():
            if N >= 0:
                continue
            info = m.bodies[bd]
            sel = np.where(m.b_body == bd)[0]
            # a member buckles over its length (pinned); a frame arm by arm, each between
            # its junctions with its own compression
            groups = [np.asarray(g, int) for g in getattr(info, "chains", []) if len(g)] \
                if info.kind == "frame" else [sel]
            best = None
            for g in groups or [sel]:
                Ng = float(Nel[g].min())
                if Ng >= 0:
                    continue
                EI = float(np.min(m.b_E[g] * np.minimum(Iy[g], Iz[g])))
                Lm = float(m.b_L0[g].sum()) if info.kind == "frame" else \
                    max(info.length_mm, float(m.b_L0[g].sum()))
                ut = -Ng / max(math.pi**2 * EI / Lm**2, 1e-9)
                if best is None or ut > best[0]:
                    mid = m.X[np.r_[m.b_i[g], m.b_j[g]]].mean(0)
                    best = (ut, mid)
            if best is not None:
                add(info.name, "buckling", best[0], best[1])
    # plates: ply strains at each element's centre against the ply's limits
    if len(m.s_nodes):
        worst_s: dict[tuple, tuple] = {}
        names = ("plate fibre tension", "plate fibre crushing", "plate across-grain "
                 "tension", "plate across-grain crushing", "plate shear")
        for e in range(len(m.s_nodes)):
            T = _block(m.s_E0[e].T, 6)
            dg = np.concatenate([u[nd] for nd in m.s_nodes[e]])
            dl = T @ dg
            Rr = shellmod.recovery(m.s_P[e][:, :2])
            eps = Rr @ dl
            lim = m.s_lim[e]
            ut = np.zeros(5)
            for ll in range(int(m.s_nl[e])):
                zb, zt, c, s = m.s_lay[e, ll]
                for z in (zb, zt):
                    ex, ey_, gxy = eps[:3] + z * eps[3:]
                    e1 = ex * c * c + ey_ * s * s + gxy * c * s
                    e2 = ex * s * s + ey_ * c * c - gxy * c * s
                    g12 = 2 * (ey_ - ex) * c * s + gxy * (c * c - s * s)
                    ut = np.maximum(ut, [max(e1, 0) / lim[0], max(-e1, 0) / lim[1],
                                         max(e2, 0) / lim[2], max(-e2, 0) / lim[3],
                                         abs(g12) / lim[4]])
            bd = int(m.s_body[e])
            ctr = m.X[m.s_nodes[e]].mean(0)
            for k in range(5):
                if ut[k] > worst_s.get((bd, k), (0,))[0]:
                    worst_s[(bd, k)] = (ut[k], ctr)
        for (bd, k), (ut, ctr) in worst_s.items():
            add(m.bodies[bd].name, names[k], ut, ctr)
    out.sort(key=lambda lm: -lm.utilisation)
    return out
