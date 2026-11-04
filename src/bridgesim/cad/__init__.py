"""CAD import: SolidWorks parts (``.SLDPRT``), raw Parasolid (``.x_b``) and STL.

:func:`load_cad` returns one :class:`CadBody` per solid body, with its name (as named in
SolidWorks), its exact boundary faces and a closed triangle mesh, all in millimetres in
the SolidWorks coordinate system (X along the bridge, Y up, Z across).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bridgesim.cad.mesh import Mesh, mass_properties, tessellate
from bridgesim.cad.parasolid import Body, Face, read_bodies
from bridgesim.cad.sldprt import read_part

SUPPORTED = (".sldprt", ".x_b", ".stl")


@dataclass
class CadBody:
    name: str
    mesh: Mesh
    faces: list[Face]
    index: int
    named_in_cad: bool = True


def _mirror_names(bodies: list[CadBody]) -> None:
    """Name unnamed bodies after a named body with the same volume and shape (SolidWorks
    mirror features copy bodies without copying their names)."""
    named = [b for b in bodies if b.named_in_cad]

    def sig(b: CadBody):
        lo, hi = b.mesh.bbox
        return b.mesh.volume, sorted(hi - lo)

    used: dict[str, int] = {}
    for b in bodies:
        if b.named_in_cad:
            continue
        v, dims = sig(b)
        best = None
        for c in named:
            vc, dc = sig(c)
            if abs(vc - v) <= 0.005 * max(v, 1) and np.allclose(sorted(dc), sorted(dims),
                                                                  atol=0.2):
                best = c
                break
        base = f"{best.name} (mirror copy)" if best else f"Body {b.index}"
        used[base] = used.get(base, 0) + 1
        b.name = base if used[base] == 1 else f"{base} #{used[base]}"


def _unique_names(bodies: list[CadBody]) -> None:
    seen: dict[str, int] = {}
    for b in bodies:
        seen[b.name] = seen.get(b.name, 0) + 1
    count: dict[str, int] = {}
    for b in bodies:
        if seen[b.name] > 1:
            count[b.name] = count.get(b.name, 0) + 1
            b.name = f"{b.name} #{count[b.name]}"


def bodies_from_parasolid(stream: bytes) -> list[CadBody]:
    raw: list[Body] = read_bodies(stream)
    out = []
    for b in raw:
        m = tessellate(b)
        if m.volume <= 0:
            raise ValueError(f"body {b.name or b.index} has no volume (inside-out or open)")
        out.append(CadBody(b.name, m, b.faces, b.index, named_in_cad=bool(b.name)))
    out.sort(key=lambda c: (-c.mesh.volume, c.name))
    _mirror_names(out)
    _unique_names(out)
    return out


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:1024]:
        pts = []
        for line in data.decode("ascii", "ignore").splitlines():
            s = line.strip().split()
            if s and s[0] == "vertex":
                pts.append([float(s[1]), float(s[2]), float(s[3])])
        P = np.array(pts, float)
    else:
        n = struct.unpack("<I", data[80:84])[0]
        arr = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
            [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        P = arr["v"].reshape(-1, 3).astype(float)
    key = np.round(P / 1e-3).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[first], inv.reshape(-1, 3)


def bodies_from_stl(path: Path, units_mm: float = 1.0) -> list[CadBody]:
    """Split an STL into bodies (connected triangle sets). Faces are rebuilt as planar
    groups of coplanar triangles so contacts can be found like in a SolidWorks part.
    Export from SolidWorks with 'Save all bodies to one file', units mm."""
    V, T = _read_stl(path)
    V = V * units_mm
    # connected components through shared vertices
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in T:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    comp = np.array([find(t[0]) for t in T])
    out = []
    for k, cid in enumerate(np.unique(comp)):
        tris = T[comp == cid]
        used, inv = np.unique(tris, return_inverse=True)
        Vb = V[used]
        Tb = inv.reshape(-1, 3)
        vol, cen, inertia = mass_properties(Vb, Tb)
        if vol < 0:
            Tb = Tb[:, [0, 2, 1]]
            vol, cen, inertia = mass_properties(Vb, Tb)
        faces = _planar_faces(Vb, Tb)
        mesh = Mesh(Vb, Tb, np.zeros(len(Tb), int), vol, cen, inertia)
        out.append(CadBody(f"Body {k + 1}", mesh, faces, k, named_in_cad=False))
    out.sort(key=lambda c: -c.mesh.volume)
    return out


def _planar_faces(V: np.ndarray, T: np.ndarray) -> list[Face]:
    """Group coplanar, edge-connected triangles into planar faces with boundary loops."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.where(nn > 0, nn, 1)[:, None]
    d = np.einsum("ij,ij->i", n, a)
    key = np.c_[np.round(n * 1e3), np.round(d * 10)]
    faces = []
    for k in np.unique(key, axis=0):
        sel = np.where((key == k).all(axis=1))[0]
        tri = T[sel]
        e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
        es = {tuple(x) for x in e}
        bnd = [(p, q) for p, q in e if (q, p) not in es]
        nxt = {p: q for p, q in bnd}
        loops, seen = [], set()
        for s in list(nxt):
            if s in seen:
                continue
            lp, cur = [s], nxt[s]
            seen.add(s)
            while cur != s and cur in nxt and cur not in seen:
                lp.append(cur)
                seen.add(cur)
                cur = nxt[cur]
            if len(lp) >= 3:
                loops.append(V[lp])
        if loops:
            faces.append(Face("plane", loops, normal=n[sel[0]], origin=V[tri[0, 0]]))
    return faces


def load_cad(path: str | Path, stl_units_mm: float = 1.0) -> list[CadBody]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".sldprt":
        stream, _ = read_part(p)
        return bodies_from_parasolid(stream)
    if ext == ".x_b":
        return bodies_from_parasolid(p.read_bytes())
    if ext == ".stl":
        return bodies_from_stl(p, stl_units_mm)
    raise ValueError(f"unsupported CAD file {p.name}: use one of {', '.join(SUPPORTED)} "
                     "(SolidWorks: save the part as .SLDPRT, or export STL with all bodies "
                     "in one file). Assemblies (.SLDASM) are not read: save the assembly as "
                     "a multi-body part first.")


__all__ = ["CadBody", "load_cad", "SUPPORTED"]
