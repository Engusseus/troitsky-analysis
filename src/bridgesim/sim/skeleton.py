"""Medial axis of a plate outline: the centre lines of a plate made of narrow arms (an X,
a frame, a curved rib), so each arm can be modelled as a member (a beam with the arm's
local width and the plate's thickness) instead of a strip of shell elements one element
wide, whose constant-strain membranes are several times too stiff in in-plane bending.

The axis comes from the Voronoi diagram of points spaced densely along the outline: its
edges inside the plate that separate two points far apart along the outline (the two
sides of an arm) are the medial axis. Short branches running into convex corners are
pruned, and the rest is split into chains between junctions and free ends."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point, Polygon


@dataclass
class Chain:
    """One arm's centre line: points (k x 2, in the plate's own coordinates) and the
    arm's local width at each (twice the distance to the outline). Chains meet at
    exactly equal end points."""

    pts: np.ndarray
    width: np.ndarray
    spur: bool = False  # an arm added out to a glued part off the medial axis

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())


def _rings(poly: Polygon) -> list[np.ndarray]:
    return [np.asarray(poly.exterior.coords)[:-1]] + [
        np.asarray(r.coords)[:-1] for r in poly.interiors]


def _sample(ring: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ~ds along a closed ring and their arc position."""
    pts, arc = [], []
    s0 = 0.0
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        seg = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(seg / ds)))
        for j in range(k):
            t = j / k
            pts.append(a + t * (b - a))
            arc.append(s0 + t * seg)
        s0 += seg
    return np.array(pts), np.array(arc)


def medial_chains(poly: Polygon, ds: float | None = None,
                  spur: float = 1.0) -> list[Chain]:
    """The medial axis of ``poly`` as chains between junctions and free ends. ``ds``: the
    outline sampling (default: a tenth of the mean arm width 2 A / P); ``spur``: branches
    to a free end shorter than ``spur`` x the local width there are pruned (corners)."""
    poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return []
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    arm = 2 * poly.area / max(poly.length, 1e-9)
    ds = ds or max(arm / 10, 0.2)
    P, ring_id, arc, ring_len = [], [], [], []
    for r_i, ring in enumerate(_rings(poly)):
        p, s = _sample(ring, ds)
        P.append(p)
        arc.append(s)
        ring_id.append(np.full(len(p), r_i))
        ring_len.append(float(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0),
                                             axis=1).sum()))
    P = np.vstack(P)
    arc = np.concatenate(arc)
    ring_id = np.concatenate(ring_id)
    if len(P) < 4:
        return []
    vor = Voronoi(P)
    V = vor.vertices
    inside = shapely.contains_xy(poly, V[:, 0], V[:, 1]) if len(V) else np.zeros(0, bool)
    bnd = poly.boundary
    keep = []
    for (i, j), (a, b) in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        if a < 0 or b < 0 or not (inside[a] and inside[b]):
            continue
        if ring_id[i] == ring_id[j]:
            L = ring_len[ring_id[i]]
            d_arc = abs(arc[i] - arc[j])
            d_arc = min(d_arc, L - d_arc)
            chord = float(np.linalg.norm(P[i] - P[j]))
            # two points on the same side, or either side of a convex corner: not the axis
            if d_arc <= 1.6 * chord + 2 * ds:
                continue
        keep.append((a, b))
    if not keep:
        return []
    # graph of the kept Voronoi edges
    adj: dict[int, set[int]] = {}
    for a, b in keep:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    radius = {v: float(bnd.distance(Point(V[v]))) for v in adj}

    def chains_of(adj):
        """Split the graph into chains between nodes of degree != 2 (cycles too)."""
        seen: set[tuple[int, int]] = set()
        out = []
        ends = [v for v, nb in adj.items() if len(nb) != 2]
        for s in ends:
            for nb in adj[s]:
                if (s, nb) in seen:
                    continue
                path = [s, nb]
                seen.add((s, nb))
                seen.add((nb, s))
                while len(adj[path[-1]]) == 2:
                    nxt = next(x for x in adj[path[-1]] if x != path[-2])
                    if (path[-1], nxt) in seen:
                        break
                    seen.add((path[-1], nxt))
                    seen.add((nxt, path[-1]))
                    path.append(nxt)
                out.append(path)
        # closed loops without any junction (a ring-shaped plate)
        for v in adj:
            for nb in adj[v]:
                if (v, nb) not in seen:
                    path = [v, nb]
                    seen.add((v, nb))
                    seen.add((nb, v))
                    while path[-1] != v:
                        nxt = next((x for x in adj[path[-1]] if (path[-1], x) not in seen),
                                   None)
                        if nxt is None:
                            break
                        seen.add((path[-1], nxt))
                        seen.add((nxt, path[-1]))
                        path.append(nxt)
                    out.append(path)
        return out

    def plen(path):
        return float(np.linalg.norm(np.diff(V[path], axis=0), axis=1).sum())

    # prune short branches to free ends (into corners), repeatedly
    for _ in range(20):
        changed = False
        for path in chains_of(adj):
            a, b = path[0], path[-1]
            if a == b:
                continue
            leaf = [x for x in (a, b) if len(adj[x]) == 1]
            if not leaf or (len(leaf) == 2 and len(adj) <= len(path)):
                continue  # a lone arm: keep it
            root = b if leaf[0] == a else a
            if plen(path) < spur * 2 * radius[root]:
                for x, y in zip(path[:-1], path[1:], strict=True):
                    adj[x].discard(y)
                    adj[y].discard(x)
                for x in path:
                    if not adj.get(x):
                        adj.pop(x, None)
                changed = True
                break
        if not changed:
            break
    out = []
    for path in chains_of(adj):
        pts = V[path]
        if plen(path) <= 1e-6:
            continue
        w = np.array([2 * radius[v] for v in path])
        out.append(Chain(pts, w))
    return out


def resample(ch: Chain, h: float, bnd=None) -> Chain:
    """The chain simplified (straight runs merged) and cut into pieces of about ``h``,
    keeping its end points exactly; widths interpolated along it (or measured from the
    outline ``bnd`` when given)."""
    line = LineString(ch.pts)
    simple = line.simplify(0.05 * float(np.median(ch.width)), preserve_topology=False)
    L = simple.length
    n = max(1, int(math.ceil(L / max(h, 1e-9) - 1e-9)))
    s = np.linspace(0, L, n + 1)
    pts = np.array([simple.interpolate(x).coords[0] for x in s])
    pts[0], pts[-1] = ch.pts[0], ch.pts[-1]
    if bnd is not None:
        w = np.array([2 * bnd.distance(Point(p)) for p in pts])
    else:
        s_old = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ch.pts, axis=0), axis=1))]
        w = np.interp(s * s_old[-1] / max(L, 1e-9), s_old, ch.width)
    return Chain(pts, w, ch.spur)


__all__ = ["Chain", "medial_chains", "resample"]
