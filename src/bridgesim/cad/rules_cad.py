"""Rulebook §8 measurements taken directly from CAD solids.

The same measurement keys as :mod:`bridgesim.measure` (which works on centre-line
models) are computed from the real body surfaces, so the 2027 rules file and the rule
engine (:func:`bridgesim.rules.evaluate_values`) apply unchanged:

* lengths: the clear span between the inner ends of the parts resting on the table,
  the span between the middles of the two support footprints, the deck length (flat
  deck-level surface), the total length;
* heights: deck top above the table, total height, clearance above the deck for the
  150 x 200 mm cart, nothing below the table;
* widths: deck width (flat deck-level surface at mid-span) and total width;
* the 1000 x 150 mm clear-span box under the bridge, the 100 x 100 mm clear opening above
  mid-span, mass, and the inclination of the parts that rest on the table.

Surfaces are sampled densely (every triangle is subdivided to <= 8 mm), so a big flat
face crossing a test box is caught even if none of its corners is inside.
"""

from __future__ import annotations

import math

import numpy as np

from bridgesim.cad import CadBody
from bridgesim.measure import Measurements


def surface_points(bodies: list[CadBody], spacing: float = 8.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Points on every body surface and the index of the body each belongs to."""
    pts, owner = [], []
    for bi, b in enumerate(bodies):
        V, T = b.mesh.vertices, b.mesh.triangles
        a, bb, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        emax = np.max(np.stack([np.linalg.norm(bb - a, axis=1), np.linalg.norm(c - bb, axis=1),
                                np.linalg.norm(a - c, axis=1)]), axis=0)
        P = [V]
        for n in np.unique(np.clip(np.ceil(emax / spacing).astype(int), 1, 60)):
            sel = np.where(np.ceil(emax / spacing).clip(1, 60).astype(int) == n)[0]
            if n == 1 or not len(sel):
                continue
            ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
            w = np.array([(i / n, j / n, 1 - (i + j) / n) for i, j in ij])
            Q = (w[None, :, 0, None] * a[sel, None] + w[None, :, 1, None] * bb[sel, None]
                 + w[None, :, 2, None] * c[sel, None]).reshape(-1, 3)
            P.append(Q)
        P = np.vstack(P)
        pts.append(P)
        owner.append(np.full(len(P), bi))
    return np.vstack(pts), np.concatenate(owner)


def _intervals_free(lo: float, hi: float, blocked: list[tuple[float, float]],
                    centre: float) -> tuple[float, float]:
    """Largest free interval inside [lo, hi] that contains ``centre``."""
    a, b = lo, hi
    for s, e in blocked:
        if e < centre:
            a = max(a, e)
        elif s > centre:
            b = min(b, s)
        else:
            return centre, centre
    return a, b


def measure_cad(bodies: list[CadBody], density_kg_m3: float, glue_fraction: float,
                constants: dict[str, float], deck_y: float | None = None,
                deck_z: float | None = None) -> Measurements:
    c = constants
    P, owner = surface_points(bodies)
    table = float(P[:, 1].min())
    out = Measurements()
    v, d = out.values, out.details
    culprits: dict[str, str] = d.setdefault("culprits", {})
    names = [b.name for b in bodies]

    # ---- supports
    sup = P[:, 1] - table < 0.5
    xs = P[sup, 0]
    xmid0 = 0.5 * (P[:, 0].min() + P[:, 0].max())
    left, right = xs[xs < xmid0], xs[xs >= xmid0]
    if not len(left) or not len(right):
        raise ValueError("the bridge must rest on the table at both ends")
    v["clear_span_mm"] = float(right.min() - left.max())
    v["span_cc_mm"] = float(0.5 * (right.min() + right.max()) - 0.5 * (left.min() + left.max()))
    sup_bodies = sorted({names[i] for i in np.unique(owner[sup])})
    d["supporting_members"] = sup_bodies
    culprits["clear_span_mm"] = "inner ends of the parts resting on the table: " + \
        ", ".join(sup_bodies[:6])
    mid_x = 0.5 * (0.5 * (right.min() + right.max()) + 0.5 * (left.min() + left.max()))

    # ---- deck surface: the flat, upward faces at deck level
    up = []
    for bi, b in enumerate(bodies):
        for f in b.faces:
            if f.kind == "plane" and f.normal[1] > 0.99:
                Q = np.vstack(f.loops)
                up.append((float(Q[:, 1].mean()), bi, Q))
    if deck_y is None:
        # the deck is the biggest horizontal surface (summed per level)
        level: dict[float, float] = {}
        for y, _bi, Q in up:
            key = round(y, 1)
            level[key] = level.get(key, 0.0) + float(np.ptp(Q[:, 0]) * np.ptp(Q[:, 2]))
        deck_y = max(level, key=level.get)
    deck = [u for u in up if abs(u[0] - deck_y) < 0.6]
    dq = np.vstack([u[2] for u in deck])
    v["deck_length_mm"] = float(dq[:, 0].max() - dq[:, 0].min())
    v["deck_height_mm"] = float(deck_y - table)
    if deck_z is None:
        deck_z = float(0.5 * (dq[:, 2].min() + dq[:, 2].max()))
    # deck width at mid-span: union of deck-level faces crossing x = mid
    zs = []
    for _y, _bi, Q in deck:
        if Q[:, 0].min() - 1 <= mid_x <= Q[:, 0].max() + 1:
            zs.append((float(Q[:, 2].min()), float(Q[:, 2].max())))
    zs.sort()
    best = (deck_z, deck_z)
    if zs:
        cur = list(zs[0])
        segs = []
        for a, b in zs[1:]:
            if a <= cur[1] + 1.0:
                cur[1] = max(cur[1], b)
            else:
                segs.append(tuple(cur))
                cur = [a, b]
        segs.append(tuple(cur))
        best = max(segs, key=lambda s_: s_[1] - s_[0])
    v["deck_width_mm"] = float(best[1] - best[0])
    d["deck_bodies"] = sorted({names[u[1]] for u in deck})

    # ---- overall
    lo, hi = P.min(0), P.max(0)
    v["total_length_mm"] = float(hi[0] - lo[0])
    v["total_height_mm"] = float(hi[1] - lo[1])
    v["total_width_mm"] = float(hi[2] - lo[2])
    v["below_table_mm"] = 0.0
    v["above_table_ok"] = 1.0
    top = int(np.argmax(P[:, 1]))
    culprits["total_height_mm"] = f"highest part: {names[owner[top]]}"

    # ---- cart envelope above the deck (§8.2.2.2): 150 wide, 200 high, along the deck
    cw, ch = c["cart_width_mm"], c["cart_height_mm"]
    xlo, xhi = dq[:, 0].min(), dq[:, 0].max()
    above = (P[:, 1] > deck_y + 0.5) & (P[:, 0] > xlo) & (P[:, 0] < xhi)
    in_band = above & (np.abs(P[:, 2] - deck_z) < cw / 2)
    clear_h = float((P[in_band, 1] - deck_y).min()) if in_band.any() else math.inf
    low = above & (P[:, 1] < deck_y + ch)
    blocked = sorted((float(z), float(z)) for z in P[low, 2])
    za, zb = _intervals_free(-1e9, 1e9, blocked, deck_z)
    v["clearance_above_deck_mm"] = clear_h if math.isfinite(clear_h) else 9999.0
    v["cart_clear_width_mm"] = float(min(zb, 1e4) - max(za, -1e4))
    ok = clear_h >= ch and (zb - za) >= cw
    v["cart_envelope_ok"] = float(ok)
    if not ok:
        hit = np.where(in_band & (P[:, 1] < deck_y + ch))[0]
        culprits["cart_envelope_ok"] = "in the cart's way: " + ", ".join(
            sorted({names[owner[i]] for i in hit})[:6])

    # ---- clear-span box under the bridge (§8.6): 1000 long x 150 high
    under = P[:, 1] < table + c["clear_box_height_mm"] - 0.01  # touching the box top is ok
    bx = sorted((float(x), float(x)) for x in P[under & ~sup, 0])
    bx += sorted((float(x), float(x)) for x in P[sup, 0])
    xa, xb = _intervals_free(-1e9, 1e9, sorted(bx), mid_x)
    v["clear_box_free_length_mm"] = float(xb - xa)

    # ---- clear opening above mid-span (§8.9)
    co = c["clear_opening_mm"]
    above_deck = P[:, 1] > deck_y + 0.5
    dx = np.abs(P[:, 0] - mid_x)
    dz = np.abs(P[:, 2] - deck_z)
    inside = above_deck & (dx < co / 2) & (dz < co / 2)
    margin = float(np.min(np.maximum(dx, dz)[above_deck]) - co / 2) if above_deck.any() \
        else 999.0
    v["clear_opening_ok"] = float(not inside.any())
    v["clear_opening_margin_mm"] = margin
    if inside.any():
        culprits["clear_opening_ok"] = "parts over the loading point: " + ", ".join(
            sorted({names[owner[i]] for i in np.where(inside)[0]})[:6])

    # ---- mass
    vol = sum(b.mesh.volume for b in bodies)
    v["mass_kg"] = vol * 1e-9 * density_kg_m3 * (1 + glue_fraction)

    # ---- inclination of parts on the table (§8.1)
    inc = 0.0
    worst = ""
    for bi in np.unique(owner[sup]):
        V = bodies[bi].mesh.vertices
        cen = V.mean(0)
        _, vecs = np.linalg.eigh(np.cov((V - cen).T))
        ax = vecs[:, 2]
        span = np.ptp((V - cen) @ ax)
        dims = sorted(np.ptp((V - cen) @ vecs, axis=0))
        if span < 3 * dims[1]:  # not a slender leg: no inclination to speak of
            continue
        ang = math.degrees(math.acos(min(1.0, abs(ax[1]))))
        if ang > 60:  # lying along the table, not a leg
            continue
        if ang > inc:
            inc, worst = ang, names[bi]
    v["pier_max_inclination_deg"] = inc
    v["piers_vertical_ok"] = float(inc <= c["pier_max_inclination_deg"])
    if inc > c["pier_max_inclination_deg"]:
        culprits["piers_vertical_ok"] = f"inclined leg: {worst} ({inc:.1f} deg)"
    d["mid_x"] = mid_x
    d["deck_y"] = deck_y
    d["deck_z"] = deck_z
    d["table_y"] = table
    return out
