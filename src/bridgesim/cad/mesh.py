"""Triangle meshes of solid bodies, and their mass properties.

Planar faces are triangulated with a constrained Delaunay triangulation of their boundary
loops (outer loop plus holes, :func:`shapely.constrained_delaunay_triangles`); cylindrical
faces are triangulated in their unrolled (arc length, height) parameter plane. Every
triangle is oriented with its normal pointing out of the material, and the body's volume,
centroid and inertia come from the closed mesh by the divergence theorem, so a body that
does not close (or turns inside out) is detected instead of silently giving wrong mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon

from bridgesim.cad.parasolid import Body, Face


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _signed_area(p2: np.ndarray) -> float:
    x, y = p2[:, 0], p2[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def face_polygon(face: Face) -> tuple[Polygon, np.ndarray, np.ndarray, np.ndarray]:
    """(2D polygon, origin, e1, e2) of a planar face: 3D point = o + u e1 + v e2."""
    n = face.normal
    e1, e2 = plane_basis(n)
    o = face.loops[0][0]
    rings = [np.c_[(lp - o) @ e1, (lp - o) @ e2] for lp in face.loops]
    k = int(np.argmax([abs(_signed_area(r)) for r in rings]))
    holes = [r for i, r in enumerate(rings) if i != k]
    poly = Polygon(rings[k], holes).buffer(0)
    return poly, o, e1, e2


def _triangulate_2d(poly) -> list[np.ndarray]:
    if poly.is_empty:
        return []
    tris = shapely.constrained_delaunay_triangles(poly)
    out = []
    for g in getattr(tris, "geoms", [tris]):
        c = np.asarray(g.exterior.coords)[:3]
        if len(c) == 3 and abs(_signed_area(c)) > 1e-9:
            out.append(c)
    return out


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) mm
    triangles: np.ndarray  # (m, 3) int
    tri_face: np.ndarray  # (m,) index of the face each triangle belongs to
    volume: float = 0.0  # mm^3
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # mm^5 (x density)
    capped_loops: int = 0  # open boundary loops closed by a fan (unreadable faces)

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def _orient(tri: np.ndarray, out_normal: np.ndarray) -> np.ndarray:
    nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return tri if float(np.dot(nrm, out_normal)) >= 0 else tri[[0, 2, 1]]


def _cyl_frame(face: Face):
    ax = face.axis
    e1, e2 = plane_basis(ax)
    return ax, e1, e2


def _cylinder_tris(face: Face) -> list[np.ndarray]:
    ax, e1, e2 = _cyl_frame(face)
    o, r = face.origin, face.radius
    if r <= 0:
        return []
    params = []
    for lp in face.loops:
        d = lp - o
        h = d @ ax
        th = np.unwrap(np.arctan2(d @ e2, d @ e1))
        params.append((th, h))
    spans = [th.max() - th.min() for th, _ in params]
    tris: list[np.ndarray] = []

    def to3(th, h):
        return o + h * ax + r * (math.cos(th) * e1 + math.sin(th) * e2)

    if all(s > 1.9 * math.pi for s in spans):  # band between full rings
        h_all = np.concatenate([h for _, h in params])
        h0, h1 = float(h_all.min()), float(h_all.max())
        n = 32
        for k in range(n):
            a, b = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            p00, p01, p10, p11 = to3(a, h0), to3(b, h0), to3(a, h1), to3(b, h1)
            tris += [np.array([p00, p01, p11]), np.array([p00, p11, p10])]
    else:
        th0 = params[0][0][0]
        rings = []
        for th, h in params:
            th = th - 2 * math.pi * round((th[0] - th0) / (2 * math.pi))
            rings.append(np.c_[r * th, h])
        k = int(np.argmax([abs(_signed_area(q)) for q in rings]))
        poly = Polygon(rings[k], [q for i, q in enumerate(rings) if i != k]).buffer(0)
        for t in _triangulate_2d(poly):
            tris.append(np.array([to3(u / r, v) for u, v in t]))
    out = []
    for t in tris:
        c = t.mean(axis=0) - o
        radial = c - (c @ ax) * ax
        nr = np.linalg.norm(radial)
        if nr > 0:
            out.append(_orient(t, face.outward * radial / nr))
    return out


def _fan_other(face: Face) -> list[np.ndarray]:
    """Fallback for an unsupported surface with one loop: best-fit plane through it.
    Faces with several loops (e.g. the wall of a hole on an unknown surface) are left
    open here and closed by :func:`_cap_open_loops`."""
    if len(face.loops) != 1:
        return []
    lp = face.loops[0]
    c = lp.mean(axis=0)
    n = np.zeros(3)
    for a, b in zip(lp, np.roll(lp, -1, axis=0), strict=True):  # Newell normal
        n += np.cross(a - c, b - c)
    if np.linalg.norm(n) == 0:
        return []
    n /= np.linalg.norm(n)
    f = Face("plane", [lp], normal=n)
    return _plane_tris(f, n)


def _plane_tris(face: Face, n: np.ndarray) -> list[np.ndarray]:
    poly, o, e1, e2 = face_polygon(face)
    out = []
    for t in _triangulate_2d(poly):
        p = o + np.outer(t[:, 0], e1) + np.outer(t[:, 1], e2)
        out.append(_orient(p, n))
    return out


def mass_properties(V: np.ndarray, T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid and inertia tensor about the centroid (unit density) of a closed,
    outward-oriented triangle mesh."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = vol6.sum() / 6.0
    if abs(vol) < 1e-12:
        return 0.0, V.mean(axis=0), np.zeros((3, 3))
    cen = (vol6[:, None] * (a + b + c)).sum(axis=0) / (24.0 * vol)
    # second moments of each tetrahedron (origin, a, b, c)
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = np.sum(vol6 / 120.0 * (
                2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j]
                + a[:, j] * c[:, i] + b[:, i] * c[:, j] + b[:, j] * c[:, i]))
    S -= vol * np.outer(cen, cen)
    inertia = np.trace(S) * np.eye(3) - S
    return float(vol), cen, inertia


def _cap_open_loops(V: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, int]:
    """Close every boundary loop (edges used by one triangle) with a fan around its
    centroid. Returns the new triangles and the number of loops capped."""
    e = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
    key = {}
    for a, b in e:
        key[(a, b)] = key.get((a, b), 0) + 1
    nxt = {}
    for (a, b), _c in key.items():
        if (b, a) not in key:  # boundary edge a->b; the cap runs b->a
            nxt[b] = a
    caps = []
    n_loops = 0
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start and cur in nxt and cur not in seen and len(loop) < 10000:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        if cur != start or len(loop) < 3:
            continue
        n_loops += 1
        c = len(V) + len(caps)
        caps.append(V[loop].mean(axis=0))
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            T = np.vstack([T, [a, b, c]])
    if caps:
        V = np.vstack([V, np.array(caps)])
    return (V, T), n_loops


def tessellate(body: Body, weld_mm: float = 1e-4) -> Mesh:
    tris: list[np.ndarray] = []
    owner: list[int] = []
    for fi, face in enumerate(body.faces):
        if face.kind == "plane":
            ft = _plane_tris(face, face.normal)
        elif face.kind == "cylinder":
            ft = _cylinder_tris(face)
        else:
            ft = _fan_other(face)
        tris += ft
        owner += [fi] * len(ft)
    if not tris:
        raise ValueError(f"body {body.name or body.index} has no triangles")
    P = np.concatenate(tris)
    key = np.round(P / weld_mm).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = P[first]
    T = inv.reshape(-1, 3)
    good = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T = T[good]
    tri_face = np.array(owner)[good]
    (V, T), n_caps = _cap_open_loops(V, T)
    tri_face = np.r_[tri_face, np.full(len(T) - len(tri_face), -1)]
    vol, cen, inertia = mass_properties(V, T)
    return Mesh(V, T, tri_face, vol, cen, inertia, capped_loops=n_caps)
