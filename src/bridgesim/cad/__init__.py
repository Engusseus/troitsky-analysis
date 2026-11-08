"""CAD import: SolidWorks parts (``.SLDPRT``), raw Parasolid (``.x_b``) and STL.

:func:`load_cad` returns one :class:`CadBody` per solid body, with its name (as named in
SolidWorks), its exact boundary faces and a closed triangle mesh, all in millimetres in
the SolidWorks coordinate system (X along the bridge, Y up, Z across).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bridgesim.cad.mesh import Mesh, mass_properties, tessellate
from bridgesim.cad.parasolid import Body, Face, _body_name, _names, read_bodies
from bridgesim.cad.sldprt import read_part

SUPPORTED = (".sldprt", ".x_b", ".stl")


@dataclass
class CadBody:
    name: str
    mesh: Mesh
    faces: list[Face]
    index: int
    named_in_cad: bool = True
    # bodies the file names that could not be read (the same on every body of a file)
    file_unread: tuple[str, ...] = ()


_SIDE_SWAPS = {2: (("Near", "Far"),), 0: (("L", "R"), ("Left", "Right"))}


def _swap_side(name: str, axes: tuple[int, ...]) -> str:
    """``name`` with its side words swapped across the mirror planes of ``axes`` (z: Near
    and Far; x: L and R, Left and Right), as whole words."""
    for ax in axes:
        for a, b in _SIDE_SWAPS.get(ax, ()):
            name = re.sub(rf"\b({a}|{b})\b", lambda m, a=a, b=b: b if m.group(1) == a
                          else a, name)
    return name


def _mirror_names(bodies: list[CadBody]) -> None:
    """Name unnamed bodies after a named body with the same volume and shape (SolidWorks
    mirror features copy bodies without copying their names): the one whose mirror image,
    across the bridge's middle plane along its span or across it, lies where the copy
    is, with its side words swapped to the copy's side (the adversarial review of round
    25: 'Pier Leg L Far (mirror copy)' stood on the near side)."""
    named = [b for b in bodies if b.named_in_cad]
    if bodies:
        lo = np.min([b.mesh.bbox[0] for b in bodies], axis=0)
        hi = np.max([b.mesh.bbox[1] for b in bodies], axis=0)
        mid = 0.5 * (lo + hi)
    else:
        mid = np.zeros(3)

    def sig(b: CadBody):
        lo, hi = b.mesh.bbox
        return b.mesh.volume, sorted(hi - lo)

    def centre(b: CadBody) -> np.ndarray:
        return 0.5 * (b.mesh.bbox[0] + b.mesh.bbox[1])

    mirrors = ((), (2,), (0,), (0, 2))
    used: dict[str, int] = {}
    for b in bodies:
        if b.named_in_cad:
            continue
        v, dims = sig(b)
        cb = centre(b)
        best, best_d, best_ax = None, np.inf, ()
        for c in named:
            vc, dc = sig(c)
            if not (abs(vc - v) <= 0.005 * max(v, 1) and np.allclose(sorted(dc), sorted(dims),
                                                                     atol=0.2)):
                continue
            for axes in mirrors:
                pc = centre(c).copy()
                for ax in axes:
                    pc[ax] = 2 * mid[ax] - pc[ax]
                d = float(np.linalg.norm(pc - cb))
                if d < best_d - 1e-6:
                    best, best_d, best_ax = c, d, axes
        if best is None:
            base = f"Body {b.index}"
        else:
            base = f"{_swap_side(best.name, best_ax)} (mirror copy)"
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
    # a body the file names but whose faces could not be put together is left out:
    # say which (its name may instead be one a read body lost: a mirror copy)
    named = {k: _body_name(v) for k, v in _names(stream).items()}
    unread: tuple[str, ...] = ()
    if len(named) > len(raw):
        got = {b.index for b in raw}
        unread = tuple(sorted(n for k, n in named.items() if k not in got and n))
    out = []
    for b in raw:
        m = tessellate(b)
        if m.volume <= 0:
            raise ValueError(f"body {b.name or b.index} has no volume (inside-out or open)")
        out.append(CadBody(b.name, m, b.faces, b.index, named_in_cad=bool(b.name),
                           file_unread=unread and (f"{len(named) - len(raw)} of "
                                                   f"{len(named)}",) + unread))
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


def _named_solids(path: Path) -> list[tuple[str, np.ndarray]]:
    """The named solids of an ASCII STL that has several (``solid <name>`` ...
    ``endsolid``): one body each, with its name. Empty for anything else."""
    data = path.read_bytes()
    if data[:5].lower() != b"solid" or b"facet" not in data[:1024]:
        return []
    out, name, pts = [], None, []
    for line in data.decode("ascii", "ignore").splitlines():
        s = line.strip().split(maxsplit=1)
        if not s:
            continue
        if s[0] == "solid":
            name, pts = (s[1].strip() if len(s) > 1 else ""), []
        elif s[0] == "vertex":
            pts.append([float(v) for v in s[1].split()[:3]])
        elif s[0] == "endsolid" and name is not None:
            out.append((name, np.array(pts, float)))
            name = None
    return out if len(out) > 1 and all(n for n, _ in out) else []


def _body_from_triangles(name: str, P: np.ndarray, k: int, named: bool) -> CadBody:
    key = np.round(P / 1e-3).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Vb, Tb = P[first], inv.reshape(-1, 3)
    vol, cen, inertia = mass_properties(Vb, Tb)
    if vol < 0:
        Tb = Tb[:, [0, 2, 1]]
        vol, cen, inertia = mass_properties(Vb, Tb)
    mesh = Mesh(Vb, Tb, np.zeros(len(Tb), int), vol, cen, inertia)
    return CadBody(name, mesh, _planar_faces(Vb, Tb), k, named_in_cad=named)


def bodies_from_stl(path: Path, units_mm: float = 1.0) -> list[CadBody]:
    """Split an STL into bodies (connected triangle sets). Faces are rebuilt as planar
    groups of coplanar triangles so contacts can be found like in a SolidWorks part.
    Export from SolidWorks with 'Save all bodies to one file', units mm. An ASCII STL
    with one named ``solid`` per body keeps the names (and touching bodies apart)."""
    solids = _named_solids(path)
    if solids:
        out = [_body_from_triangles(n, P * units_mm, k, True)
               for k, (n, P) in enumerate(solids)]
        out.sort(key=lambda c: -c.mesh.volume)
        return out
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


def _edge_components(tri: np.ndarray) -> list[np.ndarray]:
    """Indices into ``tri`` of each set of triangles joined edge to edge."""
    parent = list(range(len(tri)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[tuple[int, int], int] = {}
    for i, t in enumerate(tri):
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (int(min(p, q)), int(max(p, q)))
            j = owner.setdefault(e, i)
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(len(tri)):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(g) for g in groups.values()]


def _planar_faces(V: np.ndarray, T: np.ndarray) -> list[Face]:
    """Group coplanar, edge-connected triangles into planar faces with boundary loops:
    one face per connected region (two separate coplanar regions, the tops of a U-shaped
    block's two posts, are two faces, not one whose second region is read as a hole and
    dropped: the adversarial review of round 24)."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.where(nn > 0, nn, 1)[:, None]
    d = np.einsum("ij,ij->i", n, a)
    key = np.c_[np.round(n * 1e3), np.round(d * 10)]
    faces = []
    groups = []
    for k in np.unique(key, axis=0):
        sel0 = np.where((key == k).all(axis=1))[0]
        groups += [sel0[g] for g in _edge_components(T[sel0])]
    for sel in groups:
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


AXES = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
UNITS_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


def orientation_matrix(up: str = "y", span: str = "x") -> np.ndarray:
    """Rotation taking the file's axes to bridgesim's (X along the span, Y up, Z across):
    ``up`` and ``span`` name the file axes ('x', 'y', 'z', optionally '-z' etc.)."""
    if up.lstrip("-") == span.lstrip("-"):
        raise ValueError("--up-axis and --span-axis must be different axes")
    if up not in AXES or span not in AXES:
        raise ValueError(f"axes must be one of {', '.join(AXES)}")
    u = np.zeros(3)
    u[AXES[up]] = -1.0 if up.startswith("-") else 1.0
    sx = np.zeros(3)
    sx[AXES[span]] = -1.0 if span.startswith("-") else 1.0
    z = np.cross(sx, u)
    return np.stack([sx, u, z])  # rows: new X, Y, Z in file coordinates


def transform_bodies(bodies: list[CadBody], M: np.ndarray, scale: float = 1.0) -> None:
    """Rotate (rows of M = new axes) and scale bodies in place, faces included."""
    from bridgesim.cad.mesh import mass_properties

    for b in bodies:
        b.mesh.vertices = scale * b.mesh.vertices @ M.T
        vol, cen, inertia = mass_properties(b.mesh.vertices, b.mesh.triangles)
        b.mesh.volume, b.mesh.centroid, b.mesh.inertia = vol, cen, inertia
        for f in b.faces:
            f.loops = [scale * lp @ M.T for lp in f.loops]
            if f.normal is not None:
                f.normal = M @ f.normal
            if f.axis is not None:
                f.axis = M @ f.axis
            if f.origin is not None:
                f.origin = scale * (M @ f.origin)
            f.radius *= scale


def orientation_check(bodies: list[CadBody]) -> list[str]:
    """Problems with the orientation or units of a model meant to be in mm with X along
    the span and Y up (the SolidWorks default with the bridge drawn on the Top Plane)."""
    V = np.vstack([b.mesh.vertices for b in bodies])
    ext = np.ptp(V, axis=0)
    out = []
    if ext.max() < 20:
        out.append(f"the model is only {ext.max():.3g} units long: it looks like metres or "
                   "inches, not mm (use --units m or --units in)")
    elif ext.max() > 20000:
        out.append(f"the model is {ext.max():.0f} mm long: wrong units? (use --units)")
    if int(np.argmax(ext)) != 0:
        ax = "xyz"[int(np.argmax(ext))]
        out.append(f"the longest dimension is along {ax.upper()}, not X: is the span along "
                   f"{ax.upper()}? (use --span-axis {ax})")
    lo = V[:, 1].min()
    near = V[:, 1] < lo + 0.5
    if near.mean() < 1e-4 or np.ptp(V[near, 0]) < 0.5 * ext[0]:
        out.append("nothing rests on the lowest Y plane at both ends of the span: is Y up? "
                   "(use --up-axis z if the bridge was drawn standing on the Front Plane)")
    else:
        mid = 0.5 * (V[:, 0].min() + V[:, 0].max())
        band = (mid - 0.15 * ext[0], mid + 0.15 * ext[0])
        on_floor = False
        for b in bodies:  # triangles lying on the lowest plane that reach mid-span
            P = b.mesh.vertices[b.mesh.triangles]
            low = (P[:, :, 1] < lo + 0.5).all(axis=1)
            if low.any():
                x0, x1 = P[low, :, 0].min(axis=1), P[low, :, 0].max(axis=1)
                if ((x0 < band[1]) & (x1 > band[0])).any():
                    on_floor = True
                    break
        if on_floor:
            out.append("the bridge rests on the lowest Y plane at mid-span, not only at its "
                       "ends: it is probably lying on its side (use --up-axis z if it was "
                       "drawn standing on the Front Plane, or --up-axis -z / x)")
    return out


def load_cad(path: str | Path, stl_units_mm: float = 1.0, units: str | None = None,
             up: str = "y", span: str = "x") -> list[CadBody]:
    """Read a CAD file. ``units`` (mm, cm, m, in) scales the model to mm (SolidWorks parts
    are always read in mm); ``up``/``span`` say which file axes are up and along the span
    (bridgesim uses X along the span, Y up)."""
    p = Path(path)
    ext = p.suffix.lower()
    scale = UNITS_MM[units] if units else 1.0
    if units and units not in UNITS_MM:
        raise ValueError(f"units must be one of {', '.join(UNITS_MM)}")
    if ext == ".sldprt":
        stream, _ = read_part(p)
        bodies = bodies_from_parasolid(stream)
        scale = 1.0  # the Parasolid model is in metres and read as mm already
    elif ext == ".x_b":
        bodies = bodies_from_parasolid(p.read_bytes())
        scale = 1.0
    elif ext == ".stl":
        bodies = bodies_from_stl(p, stl_units_mm)
    else:
        bodies = None
    if bodies is not None:
        if (up, span) != ("y", "x") or scale != 1.0:
            transform_bodies(bodies, orientation_matrix(up, span), scale)
        return bodies
    raise ValueError(f"unsupported CAD file {p.name}: use one of {', '.join(SUPPORTED)} "
                     "(SolidWorks: save the part as .SLDPRT, or export STL with all bodies "
                     "in one file). Assemblies (.SLDASM) are not read: save the assembly as "
                     "a multi-body part first.")


__all__ = ["CadBody", "load_cad", "SUPPORTED", "orientation_check", "orientation_matrix"]
