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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from bridgesim.model import COMBO
from bridgesim.schema import Bridge

_DOFS = ("DX", "DY", "DZ", "RX", "RY", "RZ")

#: Above this many free degrees of freedom the sparse (ARPACK) solver is used instead of a
#: dense one, so memory stays proportional to the number of members.
DENSE_MAX_DOF = 1500
#: If the sparse solver fails to converge, retry densely up to this size (~0.5 GB).
DENSE_FALLBACK_MAX_DOF = 6000


@dataclass
class BucklingResult:
    lambda_cr: float  # critical load factor on P_ref (inf if no compression-driven mode)
    lambdas: list[float] = field(default_factory=list)  # first few positive factors
    mode_mm: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    member_energy: dict[str, float] = field(default_factory=dict)  # normalised, max = 1
    key_member: str | None = None
    note: str = ""


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
    model, bridge: Bridge, n_modes: int = 3, dense_max_dof: int = DENSE_MAX_DOF
) -> BucklingResult:
    """Solve the buckling eigenproblem on an already analysed Pynite model."""
    free = _free_dofs(model, bridge)
    # K and K_g are assembled over every DOF before the free ones are sliced out, so the
    # full size decides whether a dense matrix fits in memory.
    sparse = 6 * len(model.nodes) > dense_max_dof
    K = model.Ke(COMBO, sparse=sparse, check_stability=False)
    G = model.Kg(COMBO, sparse=sparse, first_step=False)
    if sparse:
        K, G = K.tocsr(), G.tocsr()
        K11 = K[free][:, free].tocsc()
        G11 = G[free][:, free].tocsc()
    else:
        K, G = np.asarray(K), np.asarray(G)
        K11 = K[np.ix_(free, free)]
        G11 = G[np.ix_(free, free)]
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
    order = pos[np.argsort(-mu[pos])]
    lambdas = [float(1.0 / mu[k]) for k in order[:n_modes]]

    phi = np.zeros(K.shape[0])
    phi[free] = vecs[:, order[0]]
    tr = phi.reshape(-1, 6)[:, :3]
    scale = float(np.max(np.linalg.norm(tr, axis=1))) or 1.0
    phi /= scale
    names = {n.id for n in bridge.nodes}
    mode = {nid: tuple(float(v) for v in phi[n.ID * 6:n.ID * 6 + 3])
            for nid, n in model.nodes.items() if nid in names}

    energy = {}
    for mem in bridge.members:
        pm = model.members[mem.id]
        e = 0.0
        for sub in (pm.sub_members or {pm.name: pm}).values():  # the member's elements
            i, j = sub.i_node.ID * 6, sub.j_node.ID * 6
            v = phi[np.r_[i:i + 6, j:j + 6]]
            e += float(v @ np.asarray(sub.Ke()) @ v)
        energy[mem.id] = e
    emax = max(energy.values(), default=0.0) or 1.0
    energy = {k: e / emax for k, e in energy.items()}
    key = max(energy, key=energy.get) if energy else None
    return BucklingResult(lambdas[0], lambdas, mode, energy, key)
