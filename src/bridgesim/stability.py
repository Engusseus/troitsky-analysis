"""Global elastic buckling (linear eigenvalue) analysis: the "instability" failure mode.

With K the elastic stiffness and K_g the geometric stiffness built from the member axial
forces at P_ref, the structure becomes unstable at the load factor lambda where

    (K + lambda K_g) phi = 0    ->    -K_g phi = (1/lambda) K phi

which is a symmetric generalised eigenproblem with K positive definite (stable structure).
The critical load is F_cr = lambda_cr P_ref, with lambda_cr the smallest positive root.
Source: e.g. McGuire, Gallagher & Ziemian, *Matrix Structural Analysis*, 2nd ed., Ch. 9.

This captures system modes that single-member checks miss: sway of legs, lateral buckling
of an unbraced top chord, and so on. Each member is split into two cubic elements
(``model.py``), so system modes are within about 0.2 % of a fine mesh; buckling of a single
member between joints is still over-estimated by a few percent, and that case is covered by
the member Euler check instead. F_cr is an *elastic, perfect-geometry* upper bound: real
crooked sticks buckle earlier (imperfections are planned for v0.2).

Boundary conditions: during buckling the pier bases are assumed held in X by friction at
both ends (no horizontal force is needed at the onset of buckling). The roller used in the
static analysis exists only so the bridge is not artificially tied by the table.

Each mode is described by how the bridge's nodes move (:func:`describe_mode`), not by the
member with the most strain energy alone: a small rib can hold the most energy while the
whole span bows sideways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from bridgesim.model import COMBO
from bridgesim.schema import Bridge, Stick

_DOFS = ("DX", "DY", "DZ", "RX", "RY", "RZ")

#: Above this many free degrees of freedom the sparse (ARPACK) solver is used instead of a
#: dense one, so memory stays proportional to the number of members.
DENSE_MAX_DOF = 1500
#: If the sparse solver fails to converge, retry densely up to this size (two ~50 MB
#: matrices); larger failures are reported as "not evaluated".
DENSE_FALLBACK_MAX_DOF = 2500


#: A mode whose largest node movement is below this fraction of the largest movement
#: anywhere (including member midpoints) is a member bowing between its joints.
LOCAL_MODE_RATIO = 0.5
#: Nodes moving at least this fraction of the largest node movement are named.
NAMED_NODE_RATIO = 0.8
_DIRECTIONS = ("lengthwise (X)", "vertical (Y)", "sideways (Z)")

#: Extra elastic stiffness for the buckling analysis: node ids and a matrix over their
#: translations (3 per node, in the order given), added to K but not to K_g.
StiffnessTerm = tuple[list[str], np.ndarray]


@dataclass
class BucklingResult:
    lambda_cr: float  # critical load factor on P_ref (inf if no compression-driven mode)
    lambdas: list[float] = field(default_factory=list)  # first few positive factors
    #: Mode 1 at the bridge nodes, scaled so the largest movement in the model is 1.
    mode_mm: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    member_energy: dict[str, float] = field(default_factory=dict)  # mode 1, max = 1
    key_member: str | None = None  # most strain energy in mode 1
    note: str = ""
    #: Every mode in ``lambdas``, scaled like ``mode_mm``, and a description of each.
    modes_mm: list[dict[str, tuple[float, float, float]]] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        """How mode 1 moves, e.g. "sideways (Z) movement, largest at T1n, T1f; most strain
        energy in RIB1" ("" if there is no mode)."""
        return self.descriptions[0] if self.descriptions else ""


def describe_mode(mode: dict[str, tuple[float, float, float]], key_member: str | None) -> str:
    """Describe a buckling mode by the movement of the bridge nodes.

    ``mode`` holds the node movements scaled so that the largest movement anywhere in the
    model (including the midpoints of split members) is 1. If no node moves more than
    ``LOCAL_MODE_RATIO`` of that, the joints stay nearly put and the mode is the key member
    bowing between them. Otherwise the main direction is the axis with the largest sum of
    squared node movements (a second axis is named if its sum is at least a quarter of
    that), and the nodes moving at least ``NAMED_NODE_RATIO`` of the largest node movement
    are listed, largest first (or counted, if they are more than half of all nodes).
    """
    mags = {n: math.sqrt(sum(c * c for c in v)) for n, v in mode.items()}
    peak = max(mags.values(), default=0.0)
    if peak < LOCAL_MODE_RATIO:
        who = key_member or "a member"
        return (f"local: {who} bows between its joints (they move at most {peak:.0%} as "
                f"much as the member)")
    share = [sum(v[k] ** 2 for v in mode.values()) for k in range(3)]
    axes = sorted(range(3), key=lambda k: -share[k])
    main = [_DIRECTIONS[axes[0]]]
    if share[axes[1]] >= 0.25 * share[axes[0]]:
        main.append(_DIRECTIONS[axes[1]])
    moving = sorted((n for n, m in mags.items() if m >= NAMED_NODE_RATIO * peak),
                    key=lambda n: -mags[n])
    if len(moving) > 6 and 2 * len(moving) > len(mode):
        where = f" of most of the bridge ({len(moving)} of {len(mode)} nodes)"
    else:
        where = ", largest at " + ", ".join(moving[:6]) + (
            f" and {len(moving) - 6} more" if len(moving) > 6 else "")
    energy = f"; most strain energy in {key_member}" if key_member else ""
    return f"{' and '.join(main)} movement{where}{energy}"


def diaphragm_terms(bridge: Bridge) -> list[StiffnessTerm]:
    """Shear stiffness of each deck diaphragm panel (``Deck.diaphragm``).

    A panel has corners n0, n1 on the -Z edge and f0, f1 on the +Z edge, a length a along
    the bridge (unit vector e1) and a distance b between its edges (e2, across the bridge
    in the panel's plane). It is a constant-shear panel: from the corner movements u its
    average engineering shear strain is

        gamma = (u_f0 + u_f1 - u_n0 - u_n1) . e1 / (2 b)
              + (u_n1 + u_f1 - u_n0 - u_f0) . e2 / (2 a)

    and its strain energy U = 1/2 G t a w gamma^2, with w the width of the plate (the deck's
    clear width unless the diaphragm gives width_mm) and t its thickness. So the panel's
    stiffness is k = G t a w g g^T, with g = d gamma / d u. Racking, one end moving Delta
    across the bridge, gives gamma = Delta / a and a force G t w Delta / a. Stretching or
    shortening the chords, and moving or turning the panel as a whole, give gamma = 0: the
    deck resists shear only and takes no chord force.
    """
    dia = bridge.deck.diaphragm
    if dia is None:
        return []
    t = dia.thickness_mm if dia.thickness_mm is not None else bridge.deck.thickness_mm
    w = dia.width_mm if dia.width_mm is not None else bridge.deck.clear_width_mm
    P = {n.id: np.array(n.xyz, dtype=float) for n in bridge.nodes}
    terms = []
    for n0, n1, f0, f1 in bridge.diaphragm_panels():
        e1 = (P[n1] - P[n0]) + (P[f1] - P[f0])
        e1 /= np.linalg.norm(e1)
        e2 = (P[f0] - P[n0]) + (P[f1] - P[n1])
        e2 -= (e2 @ e1) * e1
        e2 /= np.linalg.norm(e2)
        a = 0.5 * ((P[n1] - P[n0]) @ e1 + (P[f1] - P[f0]) @ e1)
        b = 0.5 * ((P[f0] - P[n0]) @ e2 + (P[f1] - P[n1]) @ e2)
        u, v = e1 / (2 * b), e2 / (2 * a)
        g = np.concatenate([-u - v, -u + v, u - v, u + v])  # n0, n1, f0, f1
        terms.append(([n0, n1, f0, f1], dia.G_MPa * t * a * w * np.outer(g, g)))
    return terms


def crossing_pairs(bridge: Bridge, ids, stick: Stick) -> list[tuple[str, str]]:
    """Pairs among members ``ids`` whose centre lines cross inside both, as in an X.

    Two members cross when they share no node, their closest points lie strictly inside
    both, and those points are less than the larger section dimension of the two apart
    (crossing straps are often drawn one on top of the other).
    """
    nodes, secs = bridge.node_map(), bridge.section_map()
    mem = {m.id: m for m in bridge.members}
    ids = [m.id for m in bridge.members if m.id in set(ids)]  # in file order
    seg = {k: (np.array(nodes[mem[k].i].xyz), np.array(nodes[mem[k].j].xyz)) for k in ids}
    size = {k: max(secs[mem[k].section].dims(stick)) for k in ids}
    pairs = []
    for x, k1 in enumerate(ids):
        for k2 in ids[x + 1:]:
            if {mem[k1].i, mem[k1].j} & {mem[k2].i, mem[k2].j}:
                continue
            (p, p2), (q, q2) = seg[k1], seg[k2]
            d1, d2, r = p2 - p, q2 - q, p - q
            a, b, e = d1 @ d1, d1 @ d2, d2 @ d2
            den = a * e - b * b
            if den <= 1e-12 * a * e:  # parallel
                continue
            s_ = (b * (d2 @ r) - e * (d1 @ r)) / den
            t_ = (a * (d2 @ r) - b * (d1 @ r)) / den
            gap = np.linalg.norm(p + s_ * d1 - q - t_ * d2)
            if 0 < s_ < 1 and 0 < t_ < 1 and gap < max(size[k1], size[k2]):
                pairs.append((k1, k2))
    return pairs


def braces_working_terms(model, bridge: Bridge, slack, stick: Stick
                         ) -> tuple[list[StiffnessTerm], list[str]]:
    """Axial stiffness, at half its value, of every slack member that crosses another slack
    member (``crossing_pairs``), and the ids of those members.

    The straps of a tension-only X both go slack under the crusher load when the chords
    shorten. When the X then racks in a buckling mode, one strap stretches by delta and the
    other shortens by delta. The stretched one straightens after a tiny movement and works
    at its full stiffness E A / L; the shortened one stays slack. That stores
    1/2 (E A / L) delta^2, and so do both straps at half stiffness,
    2 x 1/2 (E A / 2 L) delta^2, whichever way the X racks. So a linear analysis needs no
    mode direction. It is exact for racking and too stiff for a mode that shortens both
    straps (the chords moving closer together), so it is the upper end of a bracket whose
    lower end leaves the straps out.
    """
    ids = sorted({m for pair in crossing_pairs(bridge, slack, stick) for m in pair},
                 key=[m.id for m in bridge.members].index)
    nodes = bridge.node_map()
    mem = {m.id: m for m in bridge.members}
    terms = []
    for mid in ids:
        pm = model.members[mid]
        d = np.array(nodes[mem[mid].j].xyz) - np.array(nodes[mem[mid].i].xyz)
        L = float(np.linalg.norm(d))
        nn = np.outer(d, d) / L**2
        k = 0.5 * pm.material.E * pm.section.A / L
        terms.append(([mem[mid].i, mem[mid].j], k * np.block([[nn, -nn], [-nn, nn]])))
    return terms, ids


def _extra_matrix(model, terms, size: int) -> sps.csr_matrix:
    """The stiffness terms assembled into a size x size matrix on the model's DOFs."""
    rows, cols, vals = [], [], []
    for nids, k in terms:
        dofs = np.concatenate([np.arange(3) + model.nodes[n].ID * 6 for n in nids])
        r, c = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(np.asarray(k, dtype=float).ravel())
    return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(size, size))


def _free_dofs(model, bridge: Bridge) -> list[int]:
    # Friction holds DX at every base the bridge stands on, however the support is written.
    held = {model.nodes[n].ID * 6 for n in bridge.support_nodes()}
    free = []
    for node in model.nodes.values():
        for k, dof in enumerate(_DOFS):
            idx = node.ID * 6 + k
            if not getattr(node, f"support_{dof}") and idx not in held:
                free.append(idx)
    return free


def global_buckling(
    model, bridge: Bridge, n_modes: int = 3, dense_max_dof: int = DENSE_MAX_DOF,
    extra: list[StiffnessTerm] | tuple = (),
) -> BucklingResult:
    """Solve the buckling eigenproblem on an already analysed Pynite model.

    ``extra`` stiffness (e.g. the deck diaphragm, :func:`diaphragm_terms`) is added to K
    only: it carries no force in the static analysis, so it has no geometric stiffness.
    """
    free = _free_dofs(model, bridge)
    # K and K_g are assembled over every DOF before the free ones are sliced out, so the
    # full size decides whether a dense matrix fits in memory.
    sparse = 6 * len(model.nodes) > dense_max_dof
    K = model.Ke(COMBO, sparse=sparse, check_stability=False)
    G = model.Kg(COMBO, sparse=sparse, first_step=False)
    if extra:
        X = _extra_matrix(model, extra, K.shape[0])
        K = K.tocsr() + X if sparse else np.asarray(K) + X.toarray()
    if sparse:
        K, G = K.tocsr(), G.tocsr()
        K11 = K[free][:, free].tocsc()
        G11 = G[free][:, free].tocsc()
    else:
        K, G = np.asarray(K), np.asarray(G)
        K11 = K[np.ix_(free, free)]
        G11 = G[np.ix_(free, free)]
    if not free:
        return BucklingResult(math.inf, note="No free degrees of freedom: no buckling mode.")
    if sparse and len(free) <= max(n_modes + 2, 50):  # ARPACK needs k < N; tiny is cheap
        K11, G11, sparse = K11.toarray(), G11.toarray(), False
    K11 = 0.5 * (K11 + K11.T)
    G11 = 0.5 * (G11 + G11.T)
    if (G11.count_nonzero() if sparse else np.count_nonzero(G11)) == 0:
        return BucklingResult(math.inf, note="No axial forces: no buckling mode.")
    failed = BucklingResult(
        math.nan, note="The eigen-solver failed, so global buckling (instability) was NOT "
                       "evaluated. The predicted load ignores it and may be too high.")
    try:
        if sparse:
            try:
                k = max(1, min(n_modes, len(free) - 2))
                mu, vecs = spla.eigsh(-G11, k=k, M=K11, which="LA")
            except (RuntimeError, spla.ArpackError):
                if len(free) > DENSE_FALLBACK_MAX_DOF:
                    return failed
                mu, vecs = sla.eigh(-G11.toarray(), K11.toarray())
        else:
            mu, vecs = sla.eigh(-G11, K11)
    except (np.linalg.LinAlgError, ValueError):
        return failed
    tol = 1e-12 * max(1.0, float(np.max(np.abs(mu))))
    pos = np.where(mu > tol)[0]
    if pos.size == 0:
        return BucklingResult(math.inf, note="No compression-driven buckling mode.")
    order = pos[np.argsort(-mu[pos])][:n_modes]
    lambdas = [float(1.0 / mu[k]) for k in order]

    names = {n.id for n in bridge.nodes}
    elements = {}  # member id -> [(dof indices, element stiffness)] of its active elements
    for mem in bridge.members:
        pm = model.members[mem.id]
        if not pm.active[COMBO]:  # a slack tension-only member is not in K or K_g
            elements[mem.id] = []
            continue
        subs = (pm.sub_members or {pm.name: pm}).values()  # the member's elements
        elements[mem.id] = [(np.r_[s.i_node.ID * 6:s.i_node.ID * 6 + 6,
                                   s.j_node.ID * 6:s.j_node.ID * 6 + 6], np.asarray(s.Ke()))
                            for s in subs]
    modes, energies, keys = [], [], []
    for k in order:
        phi = np.zeros(K.shape[0])
        phi[free] = vecs[:, k]
        tr = phi.reshape(-1, 6)[:, :3]
        phi /= float(np.max(np.linalg.norm(tr, axis=1))) or 1.0
        modes.append({nid: tuple(float(v) for v in phi[n.ID * 6:n.ID * 6 + 3])
                      for nid, n in model.nodes.items() if nid in names})
        energy = {mid: sum(float(phi[d] @ ke @ phi[d]) for d, ke in els)
                  for mid, els in elements.items()}
        emax = max(energy.values(), default=0.0) or 1.0
        energies.append({mid: e / emax for mid, e in energy.items()})
        keys.append(max(energy, key=energy.get) if energy else None)
    descriptions = [describe_mode(m, key) for m, key in zip(modes, keys, strict=True)]
    return BucklingResult(lambdas[0], lambdas, modes[0], energies[0], keys[0],
                          modes_mm=modes, descriptions=descriptions)
