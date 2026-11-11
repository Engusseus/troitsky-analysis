"""The stick field: how a part built of sticks varies stick to stick (round 28).

A member is a chain of sticks (one stick length each) in every ply, side by side across
its width; a plate a mat of them. Each stick's stiffness and strength is lognormal about
the material's, the sticks of one body partly alike (``scatter.ply_corr`` rho: one batch,
often one tree). The scatter of one stick (log-variance s^2 = ln(1 + CoV^2)) is split into
a part the body's sticks share, rho s^2 (``idealize.stick_factors``, drawn from the
body's stream), and one of its own, (1 - rho) s^2, drawn here per physical **stick cell**:
along the member in stick lengths, its boundaries at each ply's butt joints (where the
model places them: the brick pattern, ``butt_stagger_mm``, the seeded offset of the butt
zones, ``butt_at``), across the section in plies (rows through the thickness) and stick
widths (columns). Each cell's draw comes from a stream named after the body and the
cell's indices, never from the mesh, so a seed is the same bridge at every fidelity.

A beam fibre stands for the sticks its share of the section covers: its strength factor
(f_t, f_c and G_f alike) at an element is, over the sticks its area overlaps, the mean
across the section (parallel sticks share the load) of the weakest stick along the
element's length (the weakest section in the element governs). An element's modulus is
the mean of its sticks' (a member's stiffness averages them). So a long member in tension
is as strong as its weakest section, below the mean stick on average, and a bundle of
plies averages its sticks' scatter: with rho = 0.3 a 10-ply member's strength varies by
0.61 of a stick's, as round 27 had it, but the weakest of its sections now sets it.

Plates (layered shells) carry the same field per element: one cell per stick length along
the plate's main grain and stick width across, each with the mean of its n plies (local
log-variance (1 - rho) s^2 / n), on the element's fibre strengths (``shell.element_data``'s
``kt`` and ``kc``) and its modulus.
"""

from __future__ import annotations

import math
import zlib

import numpy as np


def field_on(sp) -> bool:
    """Whether the stick field is on (``scatter.stick_field``, 1 by default; 0: one
    factor per body, as round 27)."""
    return float(sp.values.get("scatter.stick_field", 1.0)) >= 0.5


def log_sd(cov: float, share: float, amp: float) -> float:
    """The log standard deviation of a lognormal factor carrying ``share`` of the
    log-variance ln(1 + cov^2) of a stick, times the amplitude (scatter.amplitude)."""
    return math.sqrt(max(share, 0.0) * math.log(1.0 + cov * cov)) * amp


def lognormal(s: float, z: float) -> float:
    """exp(s z - s^2 / 2): the lognormal factor of mean 1 and log-sd s at z."""
    return math.exp(s * z - 0.5 * s * s)


def cell_z(seed: int, key: str, idx: tuple[int, ...], rho_es: float) -> tuple[float, float]:
    """The standard normal values (stiffness, strength) of one stick cell, from the stream
    named after ``key`` (the body, or a frame's arm) and the cell's indices; the two
    correlated by ``rho_es`` (scatter.E_strength_corr: a stiffer stick is stronger)."""
    r = np.random.default_rng([int(seed), zlib.crc32(f"stick:{key}".encode()),
                               *(int(i) for i in idx)])
    zE, zI = r.standard_normal(2)
    return float(zE), float(rho_es * zE + math.sqrt(max(0.0, 1.0 - rho_es**2)) * zI)


def ply_joints(n: int, L: float, Ls: float, stagger: float, offset: float,
               at: list[tuple[float, list[int]]] | None = None) -> list[np.ndarray]:
    """Where each of the ``n`` plies of a member ``L`` long is butted (its sticks ``Ls``
    long meet): ply k at offset + k x stagger, every stick length, or where ``at`` (the
    designer's butt_at: fractions of the length and the plies cut there) puts ply k's
    joints; none in a member no longer than a stick (one stick per ply). The positions
    the butt zones use (idealize.butt_zones), before they are merged into zones."""
    if L <= 1.02 * Ls or Ls <= 0:
        return [np.zeros(0) for _ in range(n)]
    listed: dict[int, list[float]] = {}
    for f, ks in at or []:
        for k in ks:
            if 0 <= k < n:
                listed.setdefault(k, []).append(f * L)
    out = []
    for k in range(n):
        if k in listed:
            xs = np.array(sorted(listed[k]))
        else:
            xs = np.arange((offset + k * stagger) % Ls, L, Ls)
        out.append(xs[(xs > 1e-6) & (xs < L - 1e-6)])
    return out


def overlap(a0: float, a1: float, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Length of [a0, a1] inside each [b0, b1]."""
    return np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0, None)


def fibre_weights(ny: int, nz: int, n_cols: int, n_rows: int) -> np.ndarray:
    """(ny nz, rows, cols): the share of each fibre's area (a ny x nz grid over the section,
    fibre iy nz + iz, iy across the width e2, iz through the thickness e3) that each
    stick cell (row through the thickness, column across the width) covers."""
    fy = np.arange(ny + 1) / ny
    fz = np.arange(nz + 1) / nz
    cy = np.arange(n_cols + 1) / n_cols
    cz = np.arange(n_rows + 1) / n_rows
    wy = np.array([overlap(fy[i], fy[i + 1], cy[:-1], cy[1:]) * ny for i in range(ny)])
    wz = np.array([overlap(fz[i], fz[i + 1], cz[:-1], cz[1:]) * nz for i in range(nz)])
    W = wy[:, None, None, :] * wz[None, :, :, None]  # (ny, nz, rows, cols)
    return W.reshape(ny * nz, n_rows, n_cols)


class MemberField:
    """The stick cells of one member (or one arm of a frame): ``n_rows`` plies through its
    thickness, ``n_cols`` sticks across its width, each ply cut into segments at its butt
    joints (``joints``, positions from the member's start). Cell (row, col, segment):
    stiffness and strength factors lognormal of mean 1 and log-sd ``sE`` and ``sS``
    (the stick's own share of the scatter), drawn from the stream of ``key`` and the
    cell's indices."""

    def __init__(self, seed: int, key: str, n_rows: int, n_cols: int,
                 joints: list[np.ndarray], L: float, sE: float, sS: float, rho_es: float):
        self.seed, self.key = int(seed), key
        self.n_rows, self.n_cols = max(1, n_rows), max(1, n_cols)
        self.joints = [np.asarray(j, float) for j in joints] or [np.zeros(0)]
        self.L = float(L)
        self.sE, self.sS, self.rho = sE, sS, rho_es
        self._cell: dict[tuple[int, int, int], tuple[float, float]] = {}

    def cell(self, r: int, c: int, k: int) -> tuple[float, float]:
        key = (r, c, k)
        if key not in self._cell:
            zE, zS = cell_z(self.seed, self.key, key, self.rho)
            self._cell[key] = (lognormal(self.sE, zE), lognormal(self.sS, zS))
        return self._cell[key]

    def section(self, sa: float, sb: float) -> tuple[np.ndarray, np.ndarray]:
        """Over the element from ``sa`` to ``sb`` along the member: each cell column's
        (rows, cols) weakest strength factor along the element and its length-weighted
        mean stiffness factor."""
        smin = np.empty((self.n_rows, self.n_cols))
        emean = np.empty((self.n_rows, self.n_cols))
        for r in range(self.n_rows):
            jp = self.joints[min(r, len(self.joints) - 1)]
            lo = np.r_[-np.inf, jp]
            hi = np.r_[jp, np.inf]
            ov = overlap(sa, sb, lo, hi)
            ks = np.where(ov > 1e-9)[0]
            if not len(ks):  # a zero-length element: the segment it sits in
                ks = np.array([int(np.searchsorted(jp, sa))])
                ov = np.ones(len(lo))
            w = ov[ks] / ov[ks].sum()
            for c in range(self.n_cols):
                vals = [self.cell(r, c, int(k)) for k in ks]
                smin[r, c] = min(v[1] for v in vals)
                emean[r, c] = float(sum(wi * v[0] for wi, v in zip(w, vals, strict=True)))
        return smin, emean

    def element(self, sa: float, sb: float, W: np.ndarray) -> tuple[np.ndarray, float]:
        """The fibre strength factors (nf,) of the element from ``sa`` to ``sb`` (W:
        fibre_weights for its section: the mean over the sticks each fibre covers of the
        weakest along the element) and its stiffness factor (the mean of its cells)."""
        smin, emean = self.section(sa, sb)
        return np.einsum("frc,rc->f", W, smin), float(emean.mean())


def plate_cells(seed: int, key: str, pts2: np.ndarray, g2: np.ndarray, a_min: float,
                c_min: float, Ls: float, ws: float, stagger: float, offset: float,
                long_along: bool, sE: float, sS: float, rho_es: float,
                cache: dict) -> tuple[float, float]:
    """A plate element's (strength, stiffness) factors from the stick cells under its
    sample points ``pts2`` (plate coordinates): rows of sticks ``ws`` wide across the
    main grain ``g2`` (from ``c_min``), each cut every stick length ``Ls`` along it (from
    ``a_min``, row j's joints at offset + j x stagger; one cell along a plate no longer
    than a stick, ``long_along`` False). The strength is the mean over the rows the
    element covers of the weakest cell along it, the stiffness the mean of the points'
    cells."""
    a2 = np.array([-g2[1], g2[0]])
    al = pts2 @ g2 - a_min
    ac = pts2 @ a2 - c_min
    rows = np.floor(np.maximum(ac, 0.0) / ws).astype(int)
    if long_along:
        o = np.mod(offset + rows * stagger, Ls)
        segs = np.floor((np.maximum(al, 0.0) - o) / Ls).astype(int) + 1
    else:
        segs = np.zeros(len(al), int)
    sv, ev = [], []
    for j, k in zip(rows, segs, strict=True):
        idx = (0, int(j), int(k))
        if idx not in cache:
            zE, zS = cell_z(seed, key, idx, rho_es)
            cache[idx] = (lognormal(sE, zE), lognormal(sS, zS))
        ev.append(cache[idx][0])
        sv.append(cache[idx][1])
    sv, ev = np.array(sv), np.array(ev)
    s = 0.0
    for j in np.unique(rows):
        sel = rows == j
        s += sel.sum() * sv[sel].min()
    return float(s / len(rows)), float(ev.mean())


def triangle_samples(xy3: np.ndarray) -> np.ndarray:
    """Seven points over a triangle (its centroid and two thirds of the way from it to each
    corner and to each edge's midpoint): where a plate element reads the stick cells."""
    c = xy3.mean(0)
    mids = 0.5 * (xy3 + np.roll(xy3, -1, axis=0))
    return np.vstack([c, c + 2.0 / 3.0 * (xy3 - c), c + 2.0 / 3.0 * (mids - c)])
