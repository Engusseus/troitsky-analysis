"""Contact between parts of the bridge that were not touching at the start.

Glued and touching parts interact through their glue springs (which turn into contact
with friction once the glue has broken), and faces drawn a little apart through dry
bearing points held apart by their gap (idealize: ``dry`` patches). Everything else - a
broken chord falling onto a cross-beam, a deck sagging onto a strut below it, a buckled
diagonal pressing on its neighbour, a member that seats on a part it was drawn clear of
- interacts here, by penalty contact as in explicit codes (since round 28):

* master surfaces: a beam element is a **box**, its real rectangular section (the width
  and depth its fibres are laid over, centred where they are, so a tapered member or a
  frame's arm has its own per element) swept along its chord in the element's current
  frame (built as the beam kernel builds it); its ends are faces only where the member
  ends (an end shared with the next element of the same body is open, the box reaching
  on past it, so a point near a joint between elements is pushed out through the
  member's side, not along its axis). A shell element is a slab: its triangle times
  the plate's thickness, square-edged. Until round 27 a beam was a capsule of the radius
  of the circle with the section's area (a 21 x 47 mm chord 6 mm too slim one way and 7
  mm too fat the other) and a slab's edges were rounded by half its thickness (a 50 mm
  pier read as a plate reached 25 mm past its top);
* slave points: the outline of each member's section at each beam node and element
  mid-point (its four corners and the middle of each side, and more along a side longer
  than an element, so no two are farther apart than an element), carried on the node by
  rigid arms; a point on each face of a plate at each of its nodes, and its mid-surface
  at the nodes along its edges; the corners of rigid blocks. (Corners alone miss two
  members of one width lying flush: each corner is on the other's side face, at no depth
  however far they are pressed together);
* a point never pushes on its own body. Two bodies joined by a patch (glue, a tie, one
  piece drawn as two, a dry face drawn apart) do not push on each other near that patch:
  a point closer to one of the patch's glue points than the larger section depth of the
  two bodies plus the patch's point spacing is kept off the other body (the patch's
  springs carry the contact there; points beyond it may touch the other body, as two
  members glued at one end and bent together do). Every other pair starts force-free: a
  point drawn inside another body (an overlap in the CAD, a bowed member) is allowed that
  penetration and pushes only beyond it; the allowance shrinks as the point comes out,
  so a pair drawn interpenetrating that separates later bears at its true surface. A
  pair drawn apart pushes once its real gap has closed;
* a point touching two elements of one body (at a joint between elements) is pushed by
  the deeper one only, so the joint is not twice as stiff;
* search: a hashed uniform grid lists the candidate pairs closer than contact + 3 mm;
  it is searched again whenever two parts may have closed by half that margin since the
  last search (every few thousand steps while the bridge is loaded, every few dozen
  while it collapses), and the forces are evaluated on the candidates every step.

The normal force is a penalty k x penetration, with k capped by the slave node's mass at
the time step (soft contact: the stiffness of all the points on a node together is held
to what one point had until round 27, so contact does not lower the stable time step)
and a little normal damping. Friction is Coulomb's, mu = ``contact.friction_parts``
(0.4, wood on wood as ``glue.friction``), by an elastic-plastic tangential spring of the
normal penalty's stiffness: each point keeps an anchor on the body it touches, in that
element's own coordinates, so the anchor moves with the part; the point sticks while the
spring's force is below mu x the normal force and drags the anchor once it passes it (as
the platform's contact does). The anchor is kept per point and body, across searches and
from element to element of that body; it is let go when the point leaves the body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bridgesim.sim.kernels import njit
from bridgesim.sim.model import Model

K_SLOT = 12  # contact slots per slave point (bodies it is kept off, allowances, anchors)
EXCLUDED = -1.0  # a slot's allowance: the point is kept off that body (a shared patch)
PEN_TOL = 1e-6  # mm: faces drawn touching meet within round-off, not a push


@dataclass
class SelfContact:
    s_att: np.ndarray  # slave attachment ids
    s_r: np.ndarray  # slave radius (0 for a member's outline points and block corners)
    s_body: np.ndarray
    s_el: np.ndarray  # the beam element a mid-point ring belongs to (-1: none)
    # contact slots per slave point: the body (-1 free), the allowance (EXCLUDED: kept
    # off that body; >= 0: the penetration it was drawn with), the element its friction
    # anchor is on (-1: not in contact) and the anchor in that element's coordinates
    sl_body: np.ndarray
    sl_d0: np.ndarray
    sl_e: np.ndarray
    sl_anc: np.ndarray
    el_nodes: np.ndarray  # (ne, 3)
    el_kind: np.ndarray  # 0 beam box, 1 shell slab
    el_ref: np.ndarray  # index into the beam or shell arrays (alive check)
    el_r: np.ndarray  # bounding radius about the chord (beam), half thickness (shell)
    el_body: np.ndarray
    el_h: np.ndarray  # (ne, 2) half width (e2) and half depth (e3) of a beam's section
    el_c: np.ndarray  # (ne, 2) centre of a beam's section from its chord (e2, e3)
    el_q0: np.ndarray  # (ne, 3) a beam's e2 at the start (the frame's reference)
    el_cap: np.ndarray  # (ne, 2) the beam's end at node i / j is the member's end
    k: np.ndarray  # penalty per slave
    c: np.ndarray  # normal damping per slave
    mu: float  # friction between parts (contact.friction_parts)
    h: float  # grid cell
    nbuckets: int
    margin: float = 3.0  # candidate search distance beyond contact (mm)
    seen: np.ndarray | None = None
    buf_i: np.ndarray | None = None
    buf_e: np.ndarray | None = None
    pair_i: np.ndarray | None = None
    pair_e: np.ndarray | None = None
    pen_buf: np.ndarray | None = None
    npair: int = 0
    travel: float = 0.0  # largest closing distance since the last search
    searches: int = 0
    n_active: int = 0  # (point, body) pairs pushing at the last step
    n_excluded: int = 0  # (point, body) pairs kept apart near a shared patch
    n_allowed: int = 0  # (point, body) pairs drawn interpenetrating (an allowance)
    slot_overflow: int = 0  # (point, body) pairs that found no free slot at the start
    Fc: np.ndarray | None = None
    Mc: np.ndarray | None = None
    notes: list = field(default_factory=list)


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
    element goes in every cell its box, grown by its bounding radius (twice, for the
    open ends that reach past its nodes) + ``pad``, overlaps."""
    ne = el_nodes.shape[0]
    total = 0
    for e in range(ne):
        if not alive[e]:
            continue
        i0, i1, j0, j1, k0, k1 = _cells(x, el_nodes[e], el_kind[e], 2.0 * el_r[e] + pad, h)
        total += (i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1)
    keys = np.empty(total, np.int64)
    ids = np.empty(total, np.int64)
    t = 0
    for e in range(ne):
        if not alive[e]:
            continue
        i0, i1, j0, j1, k0, k1 = _cells(x, el_nodes[e], el_kind[e], 2.0 * el_r[e] + pad, h)
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
def _sgn(v):
    return 1.0 if v >= 0.0 else -1.0


@njit(cache=True, fastmath=True, error_model="numpy")
def box_sdf(s, y, z, hx, hy, hz, cap_lo, cap_hi, ext):
    """Signed distance from the point (s, y, z), in the frame of a box centred at the
    origin with half sizes hx, hy, hz, to the box's surface (negative inside), and the
    outward unit normal (in that frame) at its closest point. An end that is not a face
    (``cap_lo``/``cap_hi`` False: the member continues past it) is moved out by ``ext``
    and never taken as the nearest face of a point inside. The exact Euclidean distance
    to a box outside it (the length of the excess over each half size), the nearest
    face inside it."""
    lo = -hx if cap_lo else -hx - ext
    hi = hx if cap_hi else hx + ext
    qlo = lo - s
    qhi = s - hi
    qy = abs(y) - hy
    qz = abs(z) - hz
    ox = max(qlo, qhi, 0.0)
    sx = -1.0 if qlo > qhi else 1.0
    oy = max(qy, 0.0)
    oz = max(qz, 0.0)
    out2 = ox * ox + oy * oy + oz * oz
    if out2 > 0.0:
        g = math.sqrt(out2)
        return g, sx * ox / g, _sgn(y) * oy / g, _sgn(z) * oz / g
    g = qy
    n0, n1, n2 = 0.0, _sgn(y), 0.0
    if qz > g:
        g = qz
        n0, n1, n2 = 0.0, 0.0, _sgn(z)
    if cap_lo and qlo > g:
        g = qlo
        n0, n1, n2 = -1.0, 0.0, 0.0
    if cap_hi and qhi > g:
        g = qhi
        n0, n1, n2 = 1.0, 0.0, 0.0
    return g, n0, n1, n2


@njit(cache=True, fastmath=True, error_model="numpy")
def _beam_axes(n0, n1, q0, x, R):
    """A beam element's current chord length and frame, as the beam kernel builds it:
    e1 along the chord, e3 normal to e1 and the mean of its two nodes' rotated e2, e2 =
    e3 x e1."""
    d0 = x[n1, 0] - x[n0, 0]
    d1 = x[n1, 1] - x[n0, 1]
    d2 = x[n1, 2] - x[n0, 2]
    ll = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if ll < 1e-12:
        ll = 1e-12
    a0, a1, a2 = d0 / ll, d1 / ll, d2 / ll
    q = np.empty(3)
    for k in range(3):
        q[k] = 0.5 * (R[n0, k, 0] * q0[0] + R[n0, k, 1] * q0[1] + R[n0, k, 2] * q0[2]
                      + R[n1, k, 0] * q0[0] + R[n1, k, 1] * q0[1] + R[n1, k, 2] * q0[2])
    c0 = a1 * q[2] - a2 * q[1]
    c1 = a2 * q[0] - a0 * q[2]
    c2 = a0 * q[1] - a1 * q[0]
    cn = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    if cn < 1e-12:
        cn = 1e-12
    c0 /= cn
    c1 /= cn
    c2 /= cn
    b0 = c1 * a2 - c2 * a1
    b1 = c2 * a0 - c0 * a2
    b2 = c0 * a1 - c1 * a0
    return ll, a0, a1, a2, b0, b1, b2, c0, c1, c2


@njit(cache=True, fastmath=True, error_model="numpy")
def _gap(p, e, x, R, el_nodes, el_kind, el_r, el_h, el_c, el_q0, el_cap):
    """Signed distance from p to element e's surface (negative inside it), the outward
    unit normal there, the reference point on the element's chord or mid-surface the
    force is carried to (moment arms and velocity), and the weights of its nodes."""
    n0 = el_nodes[e, 0]
    n1 = el_nodes[e, 1]
    n2 = el_nodes[e, 2]
    if el_kind[e] == 0:
        ll, a0, a1, a2, b0, b1, b2, c0, c1, c2 = _beam_axes(n0, n1, el_q0[e], x, R)
        r0 = p[0] - x[n0, 0]
        r1 = p[1] - x[n0, 1]
        r2 = p[2] - x[n0, 2]
        s = r0 * a0 + r1 * a1 + r2 * a2
        yy = r0 * b0 + r1 * b1 + r2 * b2 - el_c[e, 0]
        zz = r0 * c0 + r1 * c1 + r2 * c2 - el_c[e, 1]
        g, m0, m1, m2 = box_sdf(s - 0.5 * ll, yy, zz, 0.5 * ll, el_h[e, 0], el_h[e, 1],
                                el_cap[e, 0], el_cap[e, 1], el_r[e])
        nx = m0 * a0 + m1 * b0 + m2 * c0
        ny = m0 * a1 + m1 * b1 + m2 * c1
        nz = m0 * a2 + m1 * b2 + m2 * c2
        u = min(1.0, max(0.0, s / ll))
        ax = x[n0, 0] + u * ll * a0
        ay = x[n0, 1] + u * ll * a1
        az = x[n0, 2] + u * ll * a2
        return g, nx, ny, nz, ax, ay, az, 1.0 - u, u, 0.0
    # a slab: the triangle times the plate's thickness, square-edged (a prism): its
    # closest point is the triangle's closest point to p's projection, lifted by p's
    # height clamped to the half thickness; inside, the nearest face (its edges, shared
    # with the next element, open)
    w0, w1, w2 = _closest_on_triangle(p, x, n0, n1, n2)
    ax = w0 * x[n0, 0] + w1 * x[n1, 0] + w2 * x[n2, 0]
    ay = w0 * x[n0, 1] + w1 * x[n1, 1] + w2 * x[n2, 1]
    az = w0 * x[n0, 2] + w1 * x[n1, 2] + w2 * x[n2, 2]
    u0 = x[n1, 0] - x[n0, 0]
    u1 = x[n1, 1] - x[n0, 1]
    u2 = x[n1, 2] - x[n0, 2]
    v0 = x[n2, 0] - x[n0, 0]
    v1 = x[n2, 1] - x[n0, 1]
    v2 = x[n2, 2] - x[n0, 2]
    m0 = u1 * v2 - u2 * v1
    m1 = u2 * v0 - u0 * v2
    m2 = u0 * v1 - u1 * v0
    mn = max(math.sqrt(m0 * m0 + m1 * m1 + m2 * m2), 1e-30)
    m0 /= mn
    m1 /= mn
    m2 /= mn
    hgt = (p[0] - x[n0, 0]) * m0 + (p[1] - x[n0, 1]) * m1 + (p[2] - x[n0, 2]) * m2
    d0 = p[0] - hgt * m0 - ax  # in the plane: from the triangle to p's projection
    d1 = p[1] - hgt * m1 - ay
    d2 = p[2] - hgt * m2 - az
    din = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    ht = el_r[e]
    if din <= 1e-9:
        sg = _sgn(hgt)
        return abs(hgt) - ht, sg * m0, sg * m1, sg * m2, ax, ay, az, w0, w1, w2
    hc = min(ht, max(-ht, hgt))
    gx = d0 + (hgt - hc) * m0
    gy = d1 + (hgt - hc) * m1
    gz = d2 + (hgt - hc) * m2
    g = math.sqrt(gx * gx + gy * gy + gz * gz)
    return g, gx / g, gy / g, gz / g, ax, ay, az, w0, w1, w2


@njit(cache=True, fastmath=True, error_model="numpy")
def _anchor_global(e, loc, x, R, el_nodes, el_kind, el_q0, out):
    """A friction anchor, held in element e's own coordinates (a beam: along its chord
    and its section's axes from node i; a shell: barycentric weights and the height
    above the mid-surface), at its current place."""
    n0 = el_nodes[e, 0]
    n1 = el_nodes[e, 1]
    n2 = el_nodes[e, 2]
    if el_kind[e] == 0:
        ll, a0, a1, a2, b0, b1, b2, c0, c1, c2 = _beam_axes(n0, n1, el_q0[e], x, R)
        out[0] = x[n0, 0] + loc[0] * a0 + loc[1] * b0 + loc[2] * c0
        out[1] = x[n0, 1] + loc[0] * a1 + loc[1] * b1 + loc[2] * c1
        out[2] = x[n0, 2] + loc[0] * a2 + loc[1] * b2 + loc[2] * c2
        return
    u0 = x[n1, 0] - x[n0, 0]
    u1 = x[n1, 1] - x[n0, 1]
    u2 = x[n1, 2] - x[n0, 2]
    v0 = x[n2, 0] - x[n0, 0]
    v1 = x[n2, 1] - x[n0, 1]
    v2 = x[n2, 2] - x[n0, 2]
    m0 = u1 * v2 - u2 * v1
    m1 = u2 * v0 - u0 * v2
    m2 = u0 * v1 - u1 * v0
    mn = max(math.sqrt(m0 * m0 + m1 * m1 + m2 * m2), 1e-30)
    hh = loc[3] / mn
    out[0] = loc[0] * x[n0, 0] + loc[1] * x[n1, 0] + loc[2] * x[n2, 0] + hh * m0
    out[1] = loc[0] * x[n0, 1] + loc[1] * x[n1, 1] + loc[2] * x[n2, 1] + hh * m1
    out[2] = loc[0] * x[n0, 2] + loc[1] * x[n1, 2] + loc[2] * x[n2, 2] + hh * m2


@njit(cache=True, fastmath=True, error_model="numpy")
def _anchor_local(e, g, x, R, el_nodes, el_kind, el_q0, loc):
    """The inverse of _anchor_global: the point g in element e's own coordinates."""
    n0 = el_nodes[e, 0]
    n1 = el_nodes[e, 1]
    n2 = el_nodes[e, 2]
    r0 = g[0] - x[n0, 0]
    r1 = g[1] - x[n0, 1]
    r2 = g[2] - x[n0, 2]
    if el_kind[e] == 0:
        ll, a0, a1, a2, b0, b1, b2, c0, c1, c2 = _beam_axes(n0, n1, el_q0[e], x, R)
        loc[0] = r0 * a0 + r1 * a1 + r2 * a2
        loc[1] = r0 * b0 + r1 * b1 + r2 * b2
        loc[2] = r0 * c0 + r1 * c1 + r2 * c2
        loc[3] = 0.0
        return
    u0 = x[n1, 0] - x[n0, 0]
    u1 = x[n1, 1] - x[n0, 1]
    u2 = x[n1, 2] - x[n0, 2]
    v0 = x[n2, 0] - x[n0, 0]
    v1 = x[n2, 1] - x[n0, 1]
    v2 = x[n2, 2] - x[n0, 2]
    d00 = u0 * u0 + u1 * u1 + u2 * u2
    d01 = u0 * v0 + u1 * v1 + u2 * v2
    d11 = v0 * v0 + v1 * v1 + v2 * v2
    d20 = r0 * u0 + r1 * u1 + r2 * u2
    d21 = r0 * v0 + r1 * v1 + r2 * v2
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-30:
        den = 1e-30
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    m0 = u1 * v2 - u2 * v1
    m1 = u2 * v0 - u0 * v2
    m2 = u0 * v1 - u1 * v0
    mn = max(math.sqrt(m0 * m0 + m1 * m1 + m2 * m2), 1e-30)
    loc[0] = 1.0 - bv - bw
    loc[1] = bv
    loc[2] = bw
    loc[3] = (r0 * m0 + r1 * m1 + r2 * m2) / mn


@njit(cache=True, fastmath=True, error_model="numpy")
def _slot(i, body, sl_body, sl_d0, sl_e, create):
    """Slave i's slot for body (-1 if none; with ``create`` a free one is taken: a slot
    never used, or one holding nothing: no allowance, no anchor; never one that keeps
    the point off a body)."""
    free = -1
    for q in range(sl_body.shape[1]):
        b = sl_body[i, q]
        if b == body:
            return q
        if b < 0 and free < 0:
            free = q
    if not create:
        return -1
    if free < 0:
        for q in range(sl_body.shape[1]):
            if sl_d0[i, q] == 0.0 and sl_e[i, q] < 0:
                free = q
                break
    if free >= 0:
        sl_body[i, free] = body
        sl_d0[i, free] = 0.0
        sl_e[i, free] = -1
    return free


@njit(cache=True, fastmath=True, error_model="numpy")
def _point(i, x, R, a_node, a_r0, s_att, p, arm):
    nd = a_node[s_att[i]]
    for c in range(3):
        vv = 0.0
        for j in range(3):
            vv += R[nd, c, j] * a_r0[s_att[i], j]
        arm[c] = vv
        p[c] = x[nd, c] + vv
    return nd


@njit(cache=True, fastmath=True, error_model="numpy")
def search_pairs(x, R, a_node, a_r0, s_att, s_r, s_body, s_el, sl_body, sl_d0, sl_e,
                 el_nodes, el_kind, el_r, el_body, el_h, el_c, el_q0, el_cap, alive,
                 start, count, ids, h, nb, margin, loose, seen, pair_i, pair_e, reset):
    """The candidate pairs (slave i, element e of another body it is not kept off) closer
    than contact + ``margin``, in pair_i, pair_e, sorted by slave; returns how many (more
    than the lists hold: they overflowed, search again with longer ones). With ``reset``
    a slave's allowance and friction anchor on a body it is no longer near are let go
    (it has come out of that body by more than the margin)."""
    p = np.empty(3)
    arm = np.empty(3)
    n_act = 0
    seen[:] = -1
    cap = pair_i.shape[0]
    for i in range(s_att.shape[0]):
        nd = a_node[s_att[i]]
        if loose[nd] or (s_el[i] >= 0 and not alive[s_el[i]]):
            continue
        _point(i, x, R, a_node, a_r0, s_att, p, arm)
        rq = s_r[i] + margin
        i0 = int(math.floor((p[0] - rq) / h))
        i1 = int(math.floor((p[0] + rq) / h))
        j0 = int(math.floor((p[1] - rq) / h))
        j1 = int(math.floor((p[1] + rq) / h))
        k0 = int(math.floor((p[2] - rq) / h))
        k1 = int(math.floor((p[2] + rq) / h))
        n_start = n_act
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
                        eb = el_body[e]
                        if eb == s_body[i]:
                            continue
                        q = _slot(i, eb, sl_body, sl_d0, sl_e, False)
                        if q >= 0 and sl_d0[i, q] < 0.0:
                            continue  # kept off that body (a shared patch)
                        n0 = el_nodes[e, 0]
                        n1 = el_nodes[e, 1]
                        n2 = el_nodes[e, 2]
                        if loose[n0] or loose[n1] or (el_kind[e] == 1 and loose[n2]):
                            continue
                        # quick reject: farther from the element's centre than its box (or
                        # slab) reaches, open ends included
                        c0 = (x[n0, 0] + x[n1, 0] + x[n2, 0]) / 3.0
                        c1 = (x[n0, 1] + x[n1, 1] + x[n2, 1]) / 3.0
                        c2 = (x[n0, 2] + x[n1, 2] + x[n2, 2]) / 3.0
                        if el_kind[e] == 0:  # (n2 = n1: c is a third of the way along)
                            c0 = 0.5 * (x[n0, 0] + x[n1, 0])
                            c1 = 0.5 * (x[n0, 1] + x[n1, 1])
                            c2 = 0.5 * (x[n0, 2] + x[n1, 2])
                        rr = 0.0
                        for a in range(3):
                            na_ = el_nodes[e, a]
                            dd = math.sqrt((x[na_, 0] - c0) ** 2 + (x[na_, 1] - c1) ** 2
                                           + (x[na_, 2] - c2) ** 2)
                            if dd > rr:
                                rr = dd
                        reach = rr + 2.0 * el_r[e] + s_r[i] + margin
                        if (p[0] - c0) ** 2 + (p[1] - c1) ** 2 + (p[2] - c2) ** 2 > reach * reach:
                            continue
                        g = _gap(p, e, x, R, el_nodes, el_kind, el_r, el_h, el_c, el_q0,
                                 el_cap)[0]
                        if g < s_r[i] + margin:
                            if n_act < cap:
                                pair_i[n_act] = i
                                pair_e[n_act] = e
                            n_act += 1
        if reset and n_act <= cap:
            for q in range(sl_body.shape[1]):
                bq = sl_body[i, q]
                if bq < 0 or sl_d0[i, q] < 0.0:
                    continue
                found = False
                for t in range(n_start, n_act):
                    if el_body[pair_e[t]] == bq:
                        found = True
                        break
                if not found:
                    sl_d0[i, q] = 0.0
                    sl_e[i, q] = -1
    return n_act


@njit(cache=True, fastmath=True, error_model="numpy")
def pair_gaps(x, R, a_node, a_r0, s_att, el_nodes, el_kind, el_r, el_h, el_c, el_q0,
              el_cap, pair_i, pair_e, npair, out):
    """The signed distance of each listed pair (slave point to element surface)."""
    p = np.empty(3)
    arm = np.empty(3)
    for t in range(npair):
        _point(pair_i[t], x, R, a_node, a_r0, s_att, p, arm)
        out[t] = _gap(p, pair_e[t], x, R, el_nodes, el_kind, el_r, el_h, el_c, el_q0,
                      el_cap)[0]


@njit(cache=True, fastmath=True, error_model="numpy")
def push_pairs(x, R, v, om, a_node, a_r0, s_att, s_r, s_body, s_el, sl_body, sl_d0, sl_e,
               sl_anc, el_nodes, el_kind, el_r, el_body, el_h, el_c, el_q0, el_cap, alive,
               pair_i, pair_e, npair, pen_buf, kpt, cpt, mu, loose, F, Mo, update):
    """Contact forces of the candidate pairs found by the last search (F, Mo hold forces
    ON the nodes). For each slave point and each body it touches, the deepest element
    pushes: k (penetration - allowance) - c v_n along the normal, and friction from the
    point's anchor on that body, capped at mu times the normal force. Returns the number
    of (point, body) pairs pushing."""
    p = np.empty(3)
    arm = np.empty(3)
    anc = np.empty(3)
    loc = np.empty(4)
    ft = np.empty(3)
    f = np.empty(3)
    hit = np.zeros(sl_body.shape[1], np.bool_)
    n_act = 0
    t = 0
    while t < npair:
        i = pair_i[t]
        t1 = t
        while t1 < npair and pair_i[t1] == i:
            t1 += 1
        nd = a_node[s_att[i]]
        if loose[nd] or (s_el[i] >= 0 and not alive[s_el[i]]):
            t = t1
            continue
        _point(i, x, R, a_node, a_r0, s_att, p, arm)
        for tt in range(t, t1):
            e = pair_e[tt]
            if not alive[e] or loose[el_nodes[e, 0]] or loose[el_nodes[e, 1]] or (
                    el_kind[e] == 1 and loose[el_nodes[e, 2]]):
                pen_buf[tt] = -1e30
                continue
            pen_buf[tt] = s_r[i] - _gap(p, e, x, R, el_nodes, el_kind, el_r, el_h, el_c,
                                        el_q0, el_cap)[0]
        hit[:] = False
        for tt in range(t, t1):
            pen = pen_buf[tt]
            if pen <= -1e29:
                continue
            e = pair_e[tt]
            eb = el_body[e]
            best = True
            for t2 in range(t, t1):
                if t2 != tt and el_body[pair_e[t2]] == eb and (
                        pen_buf[t2] > pen or (pen_buf[t2] == pen and t2 < tt)):
                    best = False
                    break
            if not best:
                continue
            q = _slot(i, eb, sl_body, sl_d0, sl_e, False)
            d0 = 0.0
            if q >= 0:
                d0 = sl_d0[i, q]
                if d0 < 0.0:
                    continue
                if update and d0 > 0.0 and pen < d0:  # coming out: the allowance shrinks
                    d0 = pen if pen > 0.0 else 0.0
                    sl_d0[i, q] = d0
            if pen <= d0 + PEN_TOL:
                continue
            g, nx, ny, nz, ax, ay, az, w0, w1, w2 = _gap(p, e, x, R, el_nodes, el_kind,
                                                         el_r, el_h, el_c, el_q0, el_cap)
            n0 = el_nodes[e, 0]
            n1 = el_nodes[e, 1]
            n2 = el_nodes[e, 2]
            # closing speed: the slave point's velocity over that of the master's material
            # point there (its nodes' by their weights, turning about the chord or
            # mid-surface point)
            r0 = p[0] - ax
            r1 = p[1] - ay
            r2 = p[2] - az
            vsx = v[nd, 0] + om[nd, 1] * arm[2] - om[nd, 2] * arm[1]
            vsy = v[nd, 1] + om[nd, 2] * arm[0] - om[nd, 0] * arm[2]
            vsz = v[nd, 2] + om[nd, 0] * arm[1] - om[nd, 1] * arm[0]
            o0 = w0 * om[n0, 0] + w1 * om[n1, 0] + w2 * om[n2, 0]
            o1 = w0 * om[n0, 1] + w1 * om[n1, 1] + w2 * om[n2, 1]
            o2 = w0 * om[n0, 2] + w1 * om[n1, 2] + w2 * om[n2, 2]
            vmx = w0 * v[n0, 0] + w1 * v[n1, 0] + w2 * v[n2, 0] + o1 * r2 - o2 * r1
            vmy = w0 * v[n0, 1] + w1 * v[n1, 1] + w2 * v[n2, 1] + o2 * r0 - o0 * r2
            vmz = w0 * v[n0, 2] + w1 * v[n1, 2] + w2 * v[n2, 2] + o0 * r1 - o1 * r0
            vn = (vsx - vmx) * nx + (vsy - vmy) * ny + (vsz - vmz) * nz
            fn = kpt[i] * (pen - d0) - cpt[i] * vn
            if fn <= 0.0:
                continue
            # friction: the tangential spring from the point's anchor on this body
            ft[0] = 0.0
            ft[1] = 0.0
            ft[2] = 0.0
            if mu > 0.0:
                if update and q < 0:
                    q = _slot(i, eb, sl_body, sl_d0, sl_e, True)
                if q >= 0:
                    ea = sl_e[i, q]
                    if ea >= 0 and alive[ea]:
                        for k in range(4):
                            loc[k] = sl_anc[i, q, k]
                        _anchor_global(ea, loc, x, R, el_nodes, el_kind, el_q0, anc)
                    else:
                        anc[0] = p[0]
                        anc[1] = p[1]
                        anc[2] = p[2]
                    d_0 = p[0] - anc[0]
                    d_1 = p[1] - anc[1]
                    d_2 = p[2] - anc[2]
                    dn_ = d_0 * nx + d_1 * ny + d_2 * nz
                    kt = kpt[i]
                    ft[0] = -kt * (d_0 - dn_ * nx)
                    ft[1] = -kt * (d_1 - dn_ * ny)
                    ft[2] = -kt * (d_2 - dn_ * nz)
                    mag = math.sqrt(ft[0] * ft[0] + ft[1] * ft[1] + ft[2] * ft[2])
                    cap = mu * fn
                    if mag > cap:
                        scl = cap / mag
                        ft[0] *= scl
                        ft[1] *= scl
                        ft[2] *= scl
                    if update:
                        # the anchor where the spring's force is ft (dragged when slipping)
                        anc[0] = p[0] + ft[0] / kt
                        anc[1] = p[1] + ft[1] / kt
                        anc[2] = p[2] + ft[2] / kt
                        _anchor_local(e, anc, x, R, el_nodes, el_kind, el_q0, loc)
                        for k in range(4):
                            sl_anc[i, q, k] = loc[k]
                        sl_e[i, q] = e
                    hit[q] = True
            f[0] = fn * nx + ft[0]
            f[1] = fn * ny + ft[1]
            f[2] = fn * nz + ft[2]
            # on the slave point (its node, through its arm)
            F[nd, 0] += f[0]
            F[nd, 1] += f[1]
            F[nd, 2] += f[2]
            Mo[nd, 0] += arm[1] * f[2] - arm[2] * f[1]
            Mo[nd, 1] += arm[2] * f[0] - arm[0] * f[2]
            Mo[nd, 2] += arm[0] * f[1] - arm[1] * f[0]
            # and the reaction on the master's nodes, by their weights, with the moment of
            # its arm from the chord or mid-surface to the slave point (so that the pair
            # of forces has no moment of its own)
            m0 = r1 * f[2] - r2 * f[1]
            m1 = r2 * f[0] - r0 * f[2]
            m2 = r0 * f[1] - r1 * f[0]
            nn = 2 if el_kind[e] == 0 else 3
            for a in range(nn):
                na = el_nodes[e, a]
                wa = w0 if a == 0 else (w1 if a == 1 else w2)
                F[na, 0] -= wa * f[0]
                F[na, 1] -= wa * f[1]
                F[na, 2] -= wa * f[2]
                Mo[na, 0] -= wa * m0
                Mo[na, 1] -= wa * m1
                Mo[na, 2] -= wa * m2
            n_act += 1
        if update:
            for q in range(sl_body.shape[1]):
                if not hit[q] and sl_e[i, q] >= 0:
                    sl_e[i, q] = -1  # not touching that body this step: the anchor goes
        t = t1
    return n_act


# ------------------------------------------------------------------------------ setup


def beam_sections(m: Model) -> tuple[np.ndarray, np.ndarray]:
    """Each beam element's section as its fibres are laid over it: (half width along e2,
    half depth along e3) and the section's centre from the chord (e2, e3). The fibres sit
    on an evenly spaced grid spread to the rectangle's exact second moment
    (materials.fibre_spread), so the variance of their coordinates is (side)^2 / 12
    whatever their number; a net section's cut plies keep their place."""
    nb_ = len(m.b_i)
    if not nb_:
        return np.zeros((0, 2)), np.zeros((0, 2))
    fy, fz = np.asarray(m.b_fy, float), np.asarray(m.b_fz, float)
    cy, cz = fy.mean(1), fz.mean(1)
    wy = np.sqrt(12.0 * fy.var(1))
    wz = np.sqrt(12.0 * fz.var(1))
    A = np.maximum(np.asarray(m.b_A, float), 1e-9)
    # one column of fibres (no width given by them): the area over the other side
    wy = np.where(wy > 1e-6, wy, np.where(wz > 1e-6, A / np.maximum(wz, 1e-6), np.sqrt(A)))
    wz = np.where(wz > 1e-6, wz, A / np.maximum(wy, 1e-6))
    return np.c_[wy / 2, wz / 2], np.c_[cy, cz]


def _outline(hy: float, hz: float, cy: float, cz: float, L: float) -> list[tuple]:
    """Points on a section's outline in its (e2, e3) plane: the four corners, the middle
    of each side, and more along a side longer than the element (``L``) so that none is
    farther apart."""
    pts = []
    ny = max(2, int(math.ceil(2 * hy / max(L, 1e-6))))
    nz = max(2, int(math.ceil(2 * hz / max(L, 1e-6))))
    for a in range(ny + 1):
        y = -hy + 2 * hy * a / ny
        for zz in (-hz, hz):
            pts.append((cy + y, cz + zz))
    for b in range(1, nz):
        z = -hz + 2 * hz * b / nz
        for yy in (-hy, hy):
            pts.append((cy + yy, cz + z))
    return pts


def _depths(m: Model, half: np.ndarray) -> np.ndarray:
    """Each body's section depth: a member's larger section side, a plate's thickness
    (0 for a block, which has no surface others push on)."""
    out = np.zeros(max(len(m.bodies), int(m.node_body.max()) + 1 if len(m.node_body) else 0))
    if len(m.b_i):
        np.maximum.at(out, m.b_body, 2.0 * half.max(1))
    if len(m.s_nodes):
        np.maximum.at(out, m.s_body, m.s_t)
    return out


def setup(m: Model, x: np.ndarray, mass: np.ndarray, dt: float, k_nom: float = 2000.0
          ) -> SelfContact | None:
    """Slave points, master elements, and each point's start state: the bodies it is
    kept off (near a patch they share) and the penetration it was drawn with in any other
    (its allowance). Adds the slave attachments (section outlines) to the model."""
    from scipy.spatial import cKDTree

    nb_, ns_ = len(m.b_i), len(m.s_nodes)
    if nb_ + ns_ == 0:
        return None
    half, cen = beam_sections(m)
    # masters
    el_nodes = np.zeros((nb_ + ns_, 3), np.int64)
    if nb_:
        el_nodes[:nb_, 0], el_nodes[:nb_, 1], el_nodes[:nb_, 2] = m.b_i, m.b_j, m.b_j
    if ns_:
        el_nodes[nb_:] = m.s_nodes
    el_kind = np.r_[np.zeros(nb_, np.int64), np.ones(ns_, np.int64)]
    el_ref = np.r_[np.arange(nb_), np.arange(ns_)].astype(np.int64)
    r_b = np.sqrt((np.abs(cen[:, 0]) + half[:, 0]) ** 2 + (np.abs(cen[:, 1]) + half[:, 1]) ** 2) \
        if nb_ else np.zeros(0)
    el_r = np.r_[r_b, np.asarray(m.s_t, float) / 2]
    el_body = np.r_[m.b_body, m.s_body].astype(np.int64)
    el_h = np.r_[half, np.zeros((ns_, 2))] if ns_ else half.copy()
    el_c = np.r_[cen, np.zeros((ns_, 2))] if ns_ else cen.copy()
    el_q0 = np.zeros((nb_ + ns_, 3))
    el_cap = np.ones((nb_ + ns_, 2), np.bool_)
    if nb_:
        el_q0[:nb_] = np.asarray(m.b_E0)[:, :, 1]
        # an end shared with another element of the same body is not a face
        deg: dict[tuple[int, int], int] = {}
        for e in range(nb_):
            for nd in (int(m.b_i[e]), int(m.b_j[e])):
                deg[(nd, int(m.b_body[e]))] = deg.get((nd, int(m.b_body[e])), 0) + 1
        for e in range(nb_):
            bd = int(m.b_body[e])
            el_cap[e, 0] = deg[(int(m.b_i[e]), bd)] <= 1
            el_cap[e, 1] = deg[(int(m.b_j[e]), bd)] <= 1
    # slaves: each member's section outline at each beam node (from its first element
    # there) and at each element's mid-point; shell nodes (half the thickness); block
    # corners
    a_node = list(m.a_node)
    a_r0 = list(m.a_r0)
    s_att, s_r, s_body, s_el = [], [], [], []
    X = np.asarray(x, float)

    def ring(nd: int, e: int, base: np.ndarray, el: int) -> None:
        E0 = np.asarray(m.b_E0[e])
        for yy, zz in _outline(half[e, 0], half[e, 1], cen[e, 0], cen[e, 1],
                               float(m.b_L0[e])):
            a_node.append(nd)
            a_r0.append(base + yy * E0[:, 1] + zz * E0[:, 2])
            s_att.append(len(a_node) - 1)
            s_r.append(0.0)
            s_body.append(int(m.b_body[e]))
            s_el.append(el)

    if nb_:
        first: dict[int, int] = {}
        for e in range(nb_):
            for nd in (int(m.b_i[e]), int(m.b_j[e])):
                first.setdefault(nd, e)
        for nd, e in sorted(first.items()):
            ring(nd, e, np.zeros(3), -1)
        for e in range(nb_):
            ring(int(m.b_i[e]), e, 0.5 * (X[m.b_j[e]] - X[m.b_i[e]]), e)
    if ns_:
        # a plate: a point on each of its two faces at every node (its element's normal
        # times half its thickness), and its mid-surface at the nodes along its edges
        # (until round 27 a sphere of half its thickness at each node, which a thick
        # plate - a 50 mm pier read as one - pushed half its thickness past its edges)
        sn = np.asarray(m.s_nodes)
        half_t = np.full(m.n_nodes, -1.0)
        for a in range(3):
            np.maximum.at(half_t, sn[:, a], np.asarray(m.s_t, float) / 2)
        nrm = np.cross(X[sn[:, 1]] - X[sn[:, 0]], X[sn[:, 2]] - X[sn[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
        n_of = np.zeros((m.n_nodes, 3))
        for a in (2, 1, 0):  # each node: the normal of the first element it is in
            n_of[sn[::-1, a]] = nrm[::-1]
        edges = np.sort(np.r_[sn[:, [0, 1]], sn[:, [1, 2]], sn[:, [2, 0]]], axis=1)
        uq, cnt_e = np.unique(edges, axis=0, return_counts=True)
        rim = np.zeros(m.n_nodes, bool)
        rim[uq[cnt_e == 1].ravel()] = True
        for nd in np.where(half_t >= 0)[0]:
            offs = [half_t[nd] * n_of[nd], -half_t[nd] * n_of[nd]]
            if rim[nd]:
                offs.append(np.zeros(3))
            for o in offs:
                a_node.append(int(nd))
                a_r0.append(o)
                s_att.append(len(a_node) - 1)
                s_r.append(0.0)
                s_body.append(int(m.node_body[nd]))
                s_el.append(-1)
    blocks = {i for i, b in enumerate(m.bodies) if b.kind == "block"}
    for a in m.p_att:
        if m.node_body[m.a_node[a]] in blocks:
            s_att.append(int(a))
            s_r.append(0.0)
            s_body.append(int(m.node_body[m.a_node[a]]))
            s_el.append(-1)
    m.a_node = np.array(a_node, np.int64)
    m.a_r0 = np.array(a_r0, float).reshape(-1, 3)
    s_att = np.array(s_att, np.int64)
    s_r = np.array(s_r, float)
    s_body = np.array(s_body, np.int64)
    ns = len(s_att)
    # soft penalty: no stiffer than the slave node, or a light master node, can follow at
    # this time step, shared among the points on that node (their sum is what one point
    # had until round 27, so contact does not lower the stable time step)
    s_nd = m.a_node[s_att]
    n_on = np.bincount(s_nd, minlength=m.n_nodes)[s_nd].astype(float)
    m_master = np.percentile(mass[np.unique(el_nodes)], 5)
    sm = np.minimum(mass[s_nd], m_master)
    k = np.minimum(k_nom, 0.2 * sm / (dt**2 * n_on))
    c = np.minimum(0.2 * np.sqrt(k * sm / n_on), 0.3 * sm / (dt * n_on))
    size = np.r_[m.b_L0, np.sqrt(2 * np.asarray(m.s_area, float))]
    h = float(max(np.median(size), 2 * np.median(el_r), 5.0))
    sc = SelfContact(s_att, s_r, s_body, np.array(s_el, np.int64),
                     np.full((ns, K_SLOT), -1, np.int64), np.zeros((ns, K_SLOT)),
                     np.full((ns, K_SLOT), -1, np.int64), np.zeros((ns, K_SLOT, 4)),
                     el_nodes, el_kind, el_ref, el_r, el_body, el_h, el_c, el_q0, el_cap,
                     k, c, float(m.params.get("friction_parts", 0.4)), h,
                     nbuckets=int(max(4096, 16 * len(el_kind))))
    sc.seen = np.full(len(el_kind), -1, np.int64)
    sc.buf_i = np.zeros(4096, np.int64)
    sc.buf_e = np.zeros(4096, np.int64)
    R0 = np.repeat(np.eye(3)[None], m.n_nodes, axis=0)
    P = X[s_nd] + m.a_r0[s_att]
    count = np.zeros(ns, np.int64)

    def put(i: int, body: int, d0: float) -> bool:
        row = sc.sl_body[i]
        hitq = np.where(row == body)[0]
        if len(hitq):
            q = int(hitq[0])
            if sc.sl_d0[i, q] >= 0.0:  # an exclusion wins over an allowance
                sc.sl_d0[i, q] = d0 if d0 < 0 else max(d0, sc.sl_d0[i, q])
            return True
        if count[i] >= K_SLOT:
            sc.slot_overflow += 1
            return False
        q = int(count[i])
        row[q] = body
        sc.sl_d0[i, q] = d0
        count[i] += 1
        return True

    # (b) kept off a body they share a patch with, near that patch: within the larger
    # section depth of the two plus the patch's point spacing of one of its glue points
    depth = _depths(m, half)
    by_body: dict[int, np.ndarray] = {}
    for b in np.unique(s_body):
        by_body[int(b)] = np.where(s_body == b)[0]
    if len(m.c_a):
        gpts = X[m.a_node[m.c_a]] + m.a_r0[m.c_a]
    for pa in m.patches:
        idx = np.asarray(pa.springs, int)
        if not len(idx):
            continue
        G = gpts[idx]
        area = float(np.asarray(m.c_area)[idx].sum()) or float(pa.area_mm2)
        spacing = math.sqrt(max(area, 1e-9) / len(idx))
        dA = depth[pa.body_a] if pa.body_a < len(depth) else 0.0
        dB = depth[pa.body_b] if pa.body_b < len(depth) else 0.0
        D = max(dA, dB) + spacing
        tree = None
        lo, hi = G.min(0) - D, G.max(0) + D
        for own, other in ((pa.body_a, pa.body_b), (pa.body_b, pa.body_a)):
            cand = by_body.get(int(own))
            if cand is None or not len(cand):
                continue
            Pc = P[cand]
            box_ = ((Pc >= lo) & (Pc <= hi)).all(1)
            if not box_.any():
                continue
            if tree is None:
                tree = cKDTree(G)
            d, _ = tree.query(Pc[box_], distance_upper_bound=D)
            for i in cand[box_][d < D]:
                put(int(i), int(other), EXCLUDED)
    # (c) every other pair: the penetration it is drawn with, in the box geometry
    alive = np.ones(len(el_kind), bool)
    start, cnt, ids = build_grid(X, el_nodes, el_kind, el_r, alive, 0.0, h, sc.nbuckets)
    loose = np.zeros(m.n_nodes, bool)
    while True:
        n = search_pairs(X, R0, m.a_node, m.a_r0, s_att, s_r, s_body, sc.s_el, sc.sl_body,
                         sc.sl_d0, sc.sl_e, el_nodes, el_kind, el_r, el_body, el_h, el_c,
                         el_q0, el_cap, alive, start, cnt, ids, h, sc.nbuckets, 0.0, loose,
                         sc.seen, sc.buf_i, sc.buf_e, False)
        if n <= len(sc.buf_i):
            break
        sc.buf_i = np.zeros(2 * n, np.int64)
        sc.buf_e = np.zeros(2 * n, np.int64)
    if n:
        gaps = np.zeros(n)
        pair_gaps(X, R0, m.a_node, m.a_r0, s_att, el_nodes, el_kind, el_r, el_h, el_c,
                  el_q0, el_cap, sc.buf_i, sc.buf_e, n, gaps)
        pi, pb = sc.buf_i[:n], el_body[sc.buf_e[:n]]
        pen = s_r[pi] - gaps
        key = pi * (int(el_body.max()) + 1) + pb
        order = np.argsort(key, kind="stable")
        uk, first_ = np.unique(key[order], return_index=True)
        deepest = np.maximum.reduceat(pen[order], first_)
        for kk, d0 in zip(uk, deepest, strict=True):
            if d0 > PEN_TOL:
                i = int(kk // (int(el_body.max()) + 1))
                put(i, int(kk % (int(el_body.max()) + 1)), float(d0))
    sc.n_excluded = int((sc.sl_d0 < 0).sum())
    sc.n_allowed = int((sc.sl_d0 > 0).sum())
    if sc.slot_overflow:
        sc.notes.append(f"contact: {sc.slot_overflow} (point, body) start states did not "
                        f"fit in {K_SLOT} slots and were left out")
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


def search(m: Model, st, sc: SelfContact, alive: np.ndarray) -> None:
    """List the candidate pairs again (and let go of what the points have left)."""
    start, count, ids = build_grid(st.x, sc.el_nodes, sc.el_kind, sc.el_r, alive,
                                   sc.margin, sc.h, sc.nbuckets)
    while True:
        n = search_pairs(st.x, st.R, m.a_node, m.a_r0, sc.s_att, sc.s_r, sc.s_body,
                         sc.s_el, sc.sl_body, sc.sl_d0, sc.sl_e, sc.el_nodes, sc.el_kind,
                         sc.el_r, sc.el_body, sc.el_h, sc.el_c, sc.el_q0, sc.el_cap, alive,
                         start, count, ids, sc.h, sc.nbuckets, sc.margin, st.loose,
                         sc.seen, sc.buf_i, sc.buf_e, True)
        if n <= len(sc.buf_i):
            break
        sc.buf_i = np.zeros(2 * n, np.int64)
        sc.buf_e = np.zeros(2 * n, np.int64)
    sc.pair_i, sc.pair_e, sc.npair = sc.buf_i, sc.buf_e, int(n)
    if sc.pen_buf is None or len(sc.pen_buf) < len(sc.buf_i):
        sc.pen_buf = np.zeros(len(sc.buf_i))
    sc.travel = 0.0
    sc.searches += 1


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
            search(m, st, sc, alive)
    if sc.pair_i is None or sc.npair == 0:
        if update:
            sc.n_active = 0
        return
    if sc.Fc is None or sc.Fc.shape != F.shape:
        sc.Fc, sc.Mc = np.zeros_like(F), np.zeros_like(Mo)
    Fc, Mc = sc.Fc, sc.Mc
    Fc[:] = 0.0
    Mc[:] = 0.0
    n = push_pairs(st.x, st.R, st.v, st.w, m.a_node, m.a_r0, sc.s_att, sc.s_r, sc.s_body,
                   sc.s_el, sc.sl_body, sc.sl_d0, sc.sl_e, sc.sl_anc, sc.el_nodes,
                   sc.el_kind, sc.el_r, sc.el_body, sc.el_h, sc.el_c, sc.el_q0, sc.el_cap,
                   alive, sc.pair_i, sc.pair_e, sc.npair, sc.pen_buf, sc.k,
                   sc.c if update else np.zeros_like(sc.c), sc.mu, st.loose, Fc, Mc, update)
    if update:
        sc.n_active = int(n)
    F -= Fc
    Mo -= Mc
