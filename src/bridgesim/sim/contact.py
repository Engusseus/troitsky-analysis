"""Contact between parts of the bridge that were not touching at the start.

Glued and touching parts interact through their glue springs (which turn into contact
with friction once the glue has broken). Everything else - a broken chord falling onto a
cross-beam, a deck sagging onto a strut below it, a buckled diagonal pressing on its
neighbour - interacts here, by penalty contact as in explicit codes:

* slave points: every beam and shell node (a sphere of the member's radius or half the
  plate thickness), the mid-point of every beam element, and the corners of rigid
  blocks;
* master surfaces: beam elements as capsules (radius of the circle with the section's
  area) and shell elements as slabs of their thickness with rounded edges (distance to
  the closest point of the triangle);
* a point never pushes on its own body, nor on a body it was already touching at the
  start (those pairs are glued, or tied, and handled by their springs);
* search: a hashed uniform grid lists the candidate pairs closer than contact + 3 mm;
  it is searched again whenever two parts may have closed by half that margin since the
  last search (every few thousand steps while the bridge is loaded, every few dozen
  while it collapses), and the forces are evaluated on the candidates every step.

The normal force is a penalty k x penetration, with k capped by the slave node's mass at
the time step (soft contact) and a little normal damping; contact is frictionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bridgesim.sim.kernels import njit
from bridgesim.sim.model import Model

K_EXCL = 12  # bodies excluded per slave point


@dataclass
class SelfContact:
    s_att: np.ndarray  # slave attachment ids
    s_r: np.ndarray  # slave radius
    s_body: np.ndarray
    excl: np.ndarray  # (ns, K_EXCL) bodies the slave does not push on (-1: none)
    el_nodes: np.ndarray  # (ne, 3)
    el_kind: np.ndarray  # 0 beam capsule, 1 shell slab
    el_ref: np.ndarray  # index into the beam or shell arrays (alive check)
    el_r: np.ndarray
    el_body: np.ndarray
    k: np.ndarray  # penalty per slave
    c: np.ndarray  # normal damping per slave
    h: float  # grid cell
    nbuckets: int
    start: np.ndarray | None = None
    count: np.ndarray | None = None
    ids: np.ndarray | None = None
    margin: float = 3.0  # candidate search distance beyond contact (mm)
    seen: np.ndarray | None = None
    buf_i: np.ndarray | None = None
    buf_e: np.ndarray | None = None
    pair_i: np.ndarray | None = None
    pair_e: np.ndarray | None = None
    npair: int = 0
    travel: float = 0.0  # largest closing distance since the last search
    searches: int = 0
    n_active: int = 0  # contact pairs pushing at the last step
    Fc: np.ndarray | None = None
    Mc: np.ndarray | None = None


@njit(cache=True)
def _hash(i, j, k, nb):
    return ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) % nb


@njit(cache=True)
def _cells(x, nodes, kind, pad, h):
    lo0 = lo1 = lo2 = 1e30
    hi0 = hi1 = hi2 = -1e30
    nn = 2 if kind == 0 else 3
    for a in range(nn):
        nd = nodes[a]
        lo0 = min(lo0, x[nd, 0])
        lo1 = min(lo1, x[nd, 1])
        lo2 = min(lo2, x[nd, 2])
        hi0 = max(hi0, x[nd, 0])
        hi1 = max(hi1, x[nd, 1])
        hi2 = max(hi2, x[nd, 2])
    return (int(math.floor((lo0 - pad) / h)), int(math.floor((hi0 + pad) / h)),
            int(math.floor((lo1 - pad) / h)), int(math.floor((hi1 + pad) / h)),
            int(math.floor((lo2 - pad) / h)), int(math.floor((hi2 + pad) / h)))


@njit(cache=True)
def build_grid(x, el_nodes, el_kind, el_r, alive, pad, h, nb):
    """Bucket start/count and element ids sorted by bucket (two passes, no lists). Each
    element goes in every cell its box, grown by its radius + ``pad``, overlaps."""
    ne = el_nodes.shape[0]
    total = 0
    for e in range(ne):
        if not alive[e]:
            continue
        i0, i1, j0, j1, k0, k1 = _cells(x, el_nodes[e], el_kind[e], el_r[e] + pad, h)
        total += (i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1)
    keys = np.empty(total, np.int64)
    ids = np.empty(total, np.int64)
    t = 0
    for e in range(ne):
        if not alive[e]:
            continue
        i0, i1, j0, j1, k0, k1 = _cells(x, el_nodes[e], el_kind[e], el_r[e] + pad, h)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    keys[t] = _hash(i, j, k, nb)
                    ids[t] = e
                    t += 1
    order = np.argsort(keys)
    keys = keys[order]
    ids = ids[order]
    start = np.full(nb, -1, np.int64)
    count = np.zeros(nb, np.int64)
    for t in range(total):
        kk = keys[t]
        if start[kk] < 0:
            start[kk] = t
        count[kk] += 1
    return start, count, ids


@njit(cache=True, fastmath=True, error_model="numpy")
def _closest_on_triangle(p, x, n0, n1, n2):
    """Barycentric weights of the point of triangle (n0, n1, n2) closest to p (Ericson,
    Real-Time Collision Detection, 5.1.5)."""
    abx = x[n1, 0] - x[n0, 0]
    aby = x[n1, 1] - x[n0, 1]
    abz = x[n1, 2] - x[n0, 2]
    acx = x[n2, 0] - x[n0, 0]
    acy = x[n2, 1] - x[n0, 1]
    acz = x[n2, 2] - x[n0, 2]
    apx = p[0] - x[n0, 0]
    apy = p[1] - x[n0, 1]
    apz = p[2] - x[n0, 2]
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return 1.0, 0.0, 0.0
    bpx = p[0] - x[n1, 0]
    bpy = p[1] - x[n1, 1]
    bpz = p[2] - x[n1, 2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return 1.0 - v, v, 0.0
    cpx = p[0] - x[n2, 0]
    cpy = p[1] - x[n2, 1]
    cpz = p[2] - x[n2, 2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return 1.0 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return 0.0, 1.0 - w, w
    den = va + vb + vc
    if abs(den) < 1e-30:
        return 1.0, 0.0, 0.0
    v = vb / den
    w = vc / den
    return 1.0 - v - w, v, w


@njit(cache=True, fastmath=True, error_model="numpy")
def _touch(i, e, p, arm, nd, x, v, s_r, s_body, excl, el_nodes, el_kind, el_r, el_body,
           kpt, cpt, loose, F, Mo, record):
    """Slave point i (at p, arm from its node nd) against element e: push it out, or with
    ``record`` note e's body as touching. Returns True if it pushed."""
    eb = el_body[e]
    if eb == s_body[i]:
        return False
    for q in range(excl.shape[1]):
        if excl[i, q] == eb:
            return False
        if excl[i, q] < 0:
            break
    n0 = el_nodes[e, 0]
    n1 = el_nodes[e, 1]
    n2 = el_nodes[e, 2]
    if loose[n0] or loose[n1] or (el_kind[e] == 1 and loose[n2]):
        return False
    if el_kind[e] == 0:  # capsule
        dx = x[n1, 0] - x[n0, 0]
        dy = x[n1, 1] - x[n0, 1]
        dz = x[n1, 2] - x[n0, 2]
        L2 = dx * dx + dy * dy + dz * dz
        u = ((p[0] - x[n0, 0]) * dx + (p[1] - x[n0, 1]) * dy
             + (p[2] - x[n0, 2]) * dz) / max(L2, 1e-12)
        u = min(1.0, max(0.0, u))
        w0 = 1.0 - u
        w1 = u
        w2 = 0.0
        cx = x[n0, 0] + u * dx
        cy = x[n0, 1] + u * dy
        cz = x[n0, 2] + u * dz
    else:  # slab over a triangle: closest point of the triangle, edges included
        w0, w1, w2 = _closest_on_triangle(p, x, n0, n1, n2)
        cx = w0 * x[n0, 0] + w1 * x[n1, 0] + w2 * x[n2, 0]
        cy = w0 * x[n0, 1] + w1 * x[n1, 1] + w2 * x[n2, 1]
        cz = w0 * x[n0, 2] + w1 * x[n1, 2] + w2 * x[n2, 2]
    gx = p[0] - cx
    gy = p[1] - cy
    gz = p[2] - cz
    dist = math.sqrt(gx * gx + gy * gy + gz * gz)
    reach = el_r[e] + s_r[i]
    if record == 2:  # candidate search: within reach + margin (kpt holds the margin)
        return dist < reach + kpt[0]
    if record == 1:
        if dist < reach + 1.0:  # touching (or nearly) at the start: excluded
            for q in range(excl.shape[1]):
                if excl[i, q] == eb:
                    break
                if excl[i, q] < 0:
                    excl[i, q] = eb
                    break
        return False
    pen = reach - dist
    if pen <= 0.0 or dist < 1e-9:
        return False
    nx = gx / dist
    ny = gy / dist
    nz = gz / dist
    vn = ((v[nd, 0] - w0 * v[n0, 0] - w1 * v[n1, 0] - w2 * v[n2, 0]) * nx
          + (v[nd, 1] - w0 * v[n0, 1] - w1 * v[n1, 1] - w2 * v[n2, 1]) * ny
          + (v[nd, 2] - w0 * v[n0, 2] - w1 * v[n1, 2] - w2 * v[n2, 2]) * nz)
    fn = kpt[i] * pen - cpt[i] * vn
    if fn <= 0.0:
        return False
    f0 = fn * nx
    f1 = fn * ny
    f2 = fn * nz
    F[nd, 0] += f0
    F[nd, 1] += f1
    F[nd, 2] += f2
    Mo[nd, 0] += arm[1] * f2 - arm[2] * f1
    Mo[nd, 1] += arm[2] * f0 - arm[0] * f2
    Mo[nd, 2] += arm[0] * f1 - arm[1] * f0
    F[n0, 0] -= w0 * f0
    F[n0, 1] -= w0 * f1
    F[n0, 2] -= w0 * f2
    F[n1, 0] -= w1 * f0
    F[n1, 1] -= w1 * f1
    F[n1, 2] -= w1 * f2
    if el_kind[e] == 1:
        F[n2, 0] -= w2 * f0
        F[n2, 1] -= w2 * f1
        F[n2, 2] -= w2 * f2
    return True


@njit(cache=True, fastmath=True, error_model="numpy")
def contact_kernel(x, R, v, a_node, a_r0, s_att, s_r, s_body, excl, el_nodes, el_kind,
                   el_r, el_body, alive, start, count, ids, h, nb, kpt, cpt, loose, F, Mo,
                   record, pad, seen, pair_i, pair_e):
    """Penalty forces of slave points on capsules and slabs of other bodies (F, Mo hold
    forces ON the nodes, contact convention). Each slave looks in the cells its sphere
    (radius + ``pad``) covers; ``seen`` keeps an element from being met twice. With
    ``record`` = 1 nothing is pushed: each slave's excl row is filled with the bodies it
    touches (initial state); with 2 the pairs closer than reach + kpt[0] are listed in
    pair_i, pair_e (returns how many; more than their size means the lists overflowed)."""
    p = np.empty(3)
    arm = np.empty(3)
    n_act = 0
    seen[:] = -1
    for i in range(s_att.shape[0]):
        nd = a_node[s_att[i]]
        if loose[nd]:
            continue
        for c in range(3):
            vv = 0.0
            for j in range(3):
                vv += R[nd, c, j] * a_r0[s_att[i], j]
            arm[c] = vv
            p[c] = x[nd, c] + vv
        rq = s_r[i] + pad
        i0 = int(math.floor((p[0] - rq) / h))
        i1 = int(math.floor((p[0] + rq) / h))
        j0 = int(math.floor((p[1] - rq) / h))
        j1 = int(math.floor((p[1] + rq) / h))
        k0 = int(math.floor((p[2] - rq) / h))
        k1 = int(math.floor((p[2] + rq) / h))
        pushed = False
        for ci in range(i0, i1 + 1):
            for cj in range(j0, j1 + 1):
                for ck in range(k0, k1 + 1):
                    b = _hash(ci, cj, ck, nb)
                    s0 = start[b]
                    if s0 < 0:
                        continue
                    for t in range(s0, s0 + count[b]):
                        e = ids[t]
                        if seen[e] == i or not alive[e]:
                            continue
                        seen[e] = i
                        if _touch(i, e, p, arm, nd, x, v, s_r, s_body, excl, el_nodes,
                                  el_kind, el_r, el_body, kpt, cpt, loose, F, Mo, record):
                            if record == 2:
                                if n_act < pair_i.shape[0]:
                                    pair_i[n_act] = i
                                    pair_e[n_act] = e
                                n_act += 1
                            else:
                                pushed = True
        if pushed:
            n_act += 1
    return n_act


@njit(cache=True, fastmath=True, error_model="numpy")
def push_pairs(x, R, v, a_node, a_r0, s_att, s_r, s_body, excl, el_nodes, el_kind, el_r,
               el_body, alive, pair_i, pair_e, npair, kpt, cpt, loose, F, Mo):
    """Contact forces of the candidate pairs (slave i, element e) found by the last
    search. Returns the number of pairs pushing."""
    p = np.empty(3)
    arm = np.empty(3)
    n_act = 0
    for t in range(npair):
        i = pair_i[t]
        e = pair_e[t]
        if not alive[e]:
            continue
        nd = a_node[s_att[i]]
        if loose[nd]:
            continue
        for c in range(3):
            vv = 0.0
            for j in range(3):
                vv += R[nd, c, j] * a_r0[s_att[i], j]
            arm[c] = vv
            p[c] = x[nd, c] + vv
        if _touch(i, e, p, arm, nd, x, v, s_r, s_body, excl, el_nodes, el_kind, el_r,
                  el_body, kpt, cpt, loose, F, Mo, 0):
            n_act += 1
    return n_act


def setup(m: Model, x: np.ndarray, mass: np.ndarray, dt: float, k_nom: float = 2000.0
          ) -> SelfContact | None:
    """Slave points, master elements and the pairs excluded because they touch at the
    start. Adds the slave attachments (beam mid-points, nodes) to the model."""
    nb_, ns_ = len(m.b_i), len(m.s_nodes)
    if nb_ + ns_ == 0:
        return None
    # masters
    sec = np.array([m.bodies[b].section_mm for b in m.b_body]) if nb_ else np.zeros((0, 2))
    r_b = np.sqrt(np.prod(sec, axis=1) / np.pi) if nb_ else np.zeros(0)
    el_nodes = np.zeros((nb_ + ns_, 3), np.int64)
    if nb_:
        el_nodes[:nb_, 0], el_nodes[:nb_, 1], el_nodes[:nb_, 2] = m.b_i, m.b_j, m.b_j
    if ns_:
        el_nodes[nb_:] = m.s_nodes
    el_kind = np.r_[np.zeros(nb_, np.int64), np.ones(ns_, np.int64)]
    el_ref = np.r_[np.arange(nb_), np.arange(ns_)].astype(np.int64)
    el_r = np.r_[r_b, m.s_t / 2]
    el_body = np.r_[m.b_body, m.s_body].astype(np.int64)
    # slaves: nodes with their member's radius, beam mid-points, block corners
    node_r = np.full(m.n_nodes, -1.0)
    if nb_:
        np.maximum.at(node_r, m.b_i, r_b)
        np.maximum.at(node_r, m.b_j, r_b)
    if ns_:
        for a in range(3):
            np.maximum.at(node_r, m.s_nodes[:, a], m.s_t / 2)
    nodes = np.where(node_r >= 0)[0]
    a_node = list(m.a_node)
    a_r0 = list(m.a_r0)
    s_att, s_r, s_body = [], [], []
    for nd in nodes:
        a_node.append(nd)
        a_r0.append(np.zeros(3))
        s_att.append(len(a_node) - 1)
        s_r.append(node_r[nd])
        s_body.append(m.node_body[nd])
    for e in range(nb_):
        a_node.append(m.b_i[e])
        a_r0.append(0.5 * (m.X[m.b_j[e]] - m.X[m.b_i[e]]))
        s_att.append(len(a_node) - 1)
        s_r.append(r_b[e])
        s_body.append(m.b_body[e])
    blocks = {i for i, b in enumerate(m.bodies) if b.kind == "block"}
    for a in m.p_att:
        if m.node_body[m.a_node[a]] in blocks:
            s_att.append(int(a))
            s_r.append(0.0)
            s_body.append(m.node_body[m.a_node[a]])
    m.a_node = np.array(a_node, np.int64)
    m.a_r0 = np.array(a_r0, float).reshape(-1, 3)
    s_att = np.array(s_att, np.int64)
    # soft penalty: no stiffer than the slave node, or a light master node, can follow at
    # this time step
    m_master = np.percentile(mass[np.unique(el_nodes)], 5)
    sm = np.minimum(mass[m.a_node[s_att]], m_master)
    k = np.minimum(k_nom, 0.2 * sm / dt**2)
    c = np.minimum(0.2 * np.sqrt(k * sm), 0.3 * sm / dt)
    size = np.r_[m.b_L0, np.sqrt(2 * m.s_area)] if (nb_ or ns_) else np.ones(1)
    h = float(max(np.median(size), 2 * np.median(el_r), 5.0))
    sc = SelfContact(s_att, np.array(s_r), np.array(s_body, np.int64),
                     np.full((len(s_att), K_EXCL), -1, np.int64), el_nodes, el_kind, el_ref,
                     el_r, el_body, k, c, h, nbuckets=int(max(4096, 16 * len(el_kind))))
    # exclusions: whatever each point touches (or nearly) in the initial geometry
    R = np.repeat(np.eye(3)[None], m.n_nodes, axis=0)
    alive = np.ones(len(el_kind), bool)
    start, count, ids = build_grid(x, el_nodes, el_kind, el_r, alive, 1.0, h, sc.nbuckets)
    F = np.zeros((m.n_nodes, 3))
    sc.seen = np.full(len(el_kind), -1, np.int64)
    sc.buf_i = np.zeros(4096, np.int64)
    sc.buf_e = np.zeros(4096, np.int64)
    contact_kernel(x, R, np.zeros_like(x), m.a_node, m.a_r0, s_att, sc.s_r, sc.s_body,
                   sc.excl, el_nodes, el_kind, el_r, el_body, alive, start, count, ids, h,
                   sc.nbuckets, k, c, np.zeros(m.n_nodes, bool), F, F.copy(), 1, 1.0,
                   sc.seen, np.zeros(1, np.int64), np.zeros(1, np.int64))
    # the glued and tied pairs too (their springs carry the contact)
    pairs = {(p.body_a, p.body_b) for p in m.patches} | {(p.body_b, p.body_a)
                                                         for p in m.patches}
    for i in range(len(s_att)):
        row = [b for b in sc.excl[i] if b >= 0]
        extra = [bb for (ba, bb) in pairs if ba == sc.s_body[i] and bb not in row]
        row = (row + extra)[:K_EXCL]
        sc.excl[i, :] = -1
        sc.excl[i, : len(row)] = row
    return sc


@njit(cache=True)
def _vmax(v, loose):
    out = 0.0
    for i in range(v.shape[0]):
        if not loose[i]:
            for c in range(3):
                a = abs(v[i, c])
                if a > out:
                    out = a
    return out


def forces(m: Model, st, sc: SelfContact, F: np.ndarray, Mo: np.ndarray, dt: float,
           update: bool) -> None:
    """Add the self-contact forces (internal-force convention: F holds resisting forces,
    so the contact forces are subtracted). The candidate pairs (closer than reach +
    margin) are searched again once anything may have moved half the margin."""
    alive = np.r_[st.b_alive, ~st.s_gone]
    if update:
        vmax = _vmax(st.v, st.loose)
        sc.travel += 2.0 * vmax * dt  # two bodies closing on each other
        if sc.pair_i is None or sc.travel > 0.5 * sc.margin:
            start, count, ids = build_grid(st.x, sc.el_nodes, sc.el_kind, sc.el_r, alive,
                                           sc.margin, sc.h, sc.nbuckets)
            while True:
                n = contact_kernel(st.x, st.R, st.v, m.a_node, m.a_r0, sc.s_att, sc.s_r,
                                   sc.s_body, sc.excl, sc.el_nodes, sc.el_kind, sc.el_r,
                                   sc.el_body, alive, start, count, ids, sc.h, sc.nbuckets,
                                   np.full(1, sc.margin), sc.c, st.loose, F, Mo, 2,
                                   sc.margin, sc.seen, sc.buf_i, sc.buf_e)
                if n <= len(sc.buf_i):
                    break
                sc.buf_i = np.zeros(2 * n, np.int64)
                sc.buf_e = np.zeros(2 * n, np.int64)
            sc.pair_i, sc.pair_e, sc.npair = sc.buf_i, sc.buf_e, int(n)
            sc.travel = 0.0
            sc.searches += 1
    if sc.pair_i is None or sc.npair == 0:
        if update:
            sc.n_active = 0
        return
    if sc.Fc is None or sc.Fc.shape != F.shape:
        sc.Fc, sc.Mc = np.zeros_like(F), np.zeros_like(Mo)
    Fc, Mc = sc.Fc, sc.Mc
    Fc[:] = 0.0
    Mc[:] = 0.0
    n = push_pairs(st.x, st.R, st.v, m.a_node, m.a_r0, sc.s_att, sc.s_r, sc.s_body,
                   sc.excl, sc.el_nodes, sc.el_kind, sc.el_r, sc.el_body, alive, sc.pair_i,
                   sc.pair_e, sc.npair, sc.k, sc.c if update else np.zeros_like(sc.c),
                   st.loose, Fc, Mc)
    if update:
        sc.n_active = int(n)
    F -= Fc
    Mo -= Mc
