"""Read solid bodies from a Parasolid neutral-binary transmit stream (``.x_b`` content).

A Parasolid stream is a list of *nodes*. Each fixed-length node is written as its type
(big-endian uint16), its index (uint16) and its fields; pointers are node indices
(uint16, 1 = null) and reals are big-endian float64 in metres. The field order follows
the published Parasolid XT format reference:

========  ====  ================================================================
node      type  fields read here
========  ====  ================================================================
SHELL     13    node_id, attributes, body, next, face, ...
FACE      14    node_id, attributes, tolerance, next, previous, loop, shell, surface,
                sense
LOOP      15    node_id, attributes, fin, face, next
EDGE      16    node_id, attributes, tolerance, fin, previous, next, curve, ...
FIN       17    attributes, loop, forward, backward, vertex, other, edge, curve,
                next_at_vertex, sense
VERTEX    18    node_id, attributes, fin, previous, next, point, tolerance, owner
POINT     29    node_id, attributes, owner, next, previous, position
LINE      30    node_id, attributes, owner, next, previous, geometric_owner, sense,
                position, direction
CIRCLE    31    ... sense, centre, normal, x_axis, radius
PLANE     50    ... sense, position, normal, x_axis
CYLINDER  51    ... sense, position, axis, radius, x_axis
ATTRIBUTE 81    (variable) node_id, definition, owner, ..., value pointers
CHAR_VAL. 84    (variable) characters
========  ====  ================================================================

Instead of walking the stream (which would need the full embedded schema), every node of
these types is found by its type code and accepted only if its pointers lead to nodes of
the right types (a vertex must point to a point, a fin to a vertex, a loop to a fin, a
face to a loop, a shell to a face). Faces are grouped into bodies through their shells,
and SolidWorks body names are read from the ``SW_INTERNAL_BODY_NAME``-style string
attributes owned by each body.

A fin runs from the vertex of the fin before it to its own vertex, along its edge's curve
(reversed when the fin's sense is ``-``). Circles are sampled, so holes come out as
polygons with ``ARC_SEGMENTS`` sides per full turn.
"""

from __future__ import annotations

import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

ARC_SEGMENTS = 32
NULL = 1  # a null pointer is written as index 1 in the binary form
M_TO_MM = 1000.0


class ParasolidError(ValueError):
    pass


@dataclass
class Face:
    kind: str  # "plane", "cylinder" or "other"
    loops: list[np.ndarray]  # each (k, 3) mm, in fin order
    normal: np.ndarray | None = None  # outward unit normal for planes
    axis: np.ndarray | None = None  # cylinder axis (unit)
    origin: np.ndarray | None = None  # point on plane / on cylinder axis (mm)
    radius: float = 0.0  # cylinder radius (mm)
    outward: float = 1.0  # +1: material inside the cylinder (a pin); -1: a hole wall


@dataclass
class Body:
    index: int
    name: str
    faces: list[Face] = field(default_factory=list)


def _scan(x: bytes, type_code: int, fmt: str) -> dict[int, list[tuple]]:
    tb = struct.pack(">H", type_code)
    n = struct.calcsize(fmt)
    out: dict[int, list[tuple]] = defaultdict(list)
    i = x.find(tb)
    while i >= 0:
        if i + 2 + n <= len(x):
            v = struct.unpack(fmt, x[i + 2 : i + 2 + n])
            out[v[0]].append(v)
        i = x.find(tb, i + 1)
    return out


def _pick(cands: dict[int, list[tuple]], ok) -> dict[int, tuple]:
    res = {}
    for k, lst in cands.items():
        for v in lst:
            if ok(v):
                res[k] = v
                break
    return res


def _finite(vals, lim=100.0) -> bool:
    return all(math.isfinite(a) and abs(a) < lim for a in vals)


def _sense(c: bytes) -> bool:
    return c in (b"+", b"-")


def _names(x: bytes) -> dict[int, str]:
    """Body index -> name, from string attributes whose owner is a body."""
    strings: dict[int, str] = {}
    for m in re.finditer(b"\x00\x54", x):
        i = m.start()
        if i + 8 > len(x):
            break
        n, idx = struct.unpack(">IH", x[i + 2 : i + 8])
        if 0 < n < 512:
            s = x[i + 8 : i + 8 + n]
            if len(s) == n and all(32 <= c < 127 for c in s):
                strings[idx] = s.decode("ascii")
    owned: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for m in re.finditer(b"\x00\x51", x):
        i = m.start()
        if i + 24 > len(x):
            break
        n, _idx, _nid, defn, owner = struct.unpack(">IHIHH", x[i + 2 : i + 16])
        if not 0 < n < 16 or i + 24 + 2 * n > len(x):
            continue
        for f in struct.unpack(f">{n}H", x[i + 24 : i + 24 + 2 * n]):
            if f in strings:
                owned[owner].append((defn, strings[f]))
    return {o: v for o, v in owned.items()}


_GUID = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


def _body_name(attrs: list[tuple[int, str]]) -> str:
    cands = [s for _d, s in attrs if not _GUID.match(s)]
    return max(cands, key=len) if cands else ""


def read_bodies(x: bytes) -> list[Body]:
    """Solid bodies of a Parasolid neutral-binary partition, in model units of mm."""
    if not x.startswith(b"PS"):
        raise ParasolidError("not a Parasolid neutral-binary stream")
    P = _pick(_scan(x, 29, ">HIHHHH3d"), lambda v: _finite(v[6:9], 50.0))
    V = _pick(_scan(x, 18, ">HIHHHHHdH"), lambda v: v[6] in P)
    F = _pick(_scan(x, 17, ">HHHHHHHHHHc"),
              lambda v: (v[5] in V or v[5] == NULL) and _sense(v[10]))
    L = _pick(_scan(x, 15, ">HIHHHH"), lambda v: v[3] in F)
    FA = _pick(_scan(x, 14, ">HIHdHHHHHc"), lambda v: v[6] in L and _sense(v[9]))
    E = _pick(_scan(x, 16, ">HIHdHHHHHHH"), lambda v: v[4] in F)
    S = _pick(_scan(x, 13, ">HIHHHHHHHH"), lambda v: v[5] in FA)
    PL = _pick(_scan(x, 50, ">HIHHHHHc9d"), lambda v: _sense(v[7]) and _finite(v[8:17]))
    CY = _pick(_scan(x, 51, ">HIHHHHHc10d"), lambda v: _sense(v[7]) and _finite(v[8:18]))
    CI = _pick(_scan(x, 31, ">HIHHHHHc10d"), lambda v: _sense(v[7]) and _finite(v[8:18]))
    if not FA or not S:
        raise ParasolidError("no faces or shells found: unsupported Parasolid version?")

    def pt(vertex_idx: int) -> np.ndarray:
        return np.array(P[V[vertex_idx][6]][6:9]) * M_TO_MM

    def fin_points(f: tuple, start: np.ndarray | None, end: np.ndarray | None) -> list:
        """Points along fin ``f`` from start to end, excluding ``start``."""
        edge = E.get(f[7])
        curve = edge[7] if edge else NULL
        fwd = f[10] == b"+"
        if curve in CI:
            c = CI[curve]
            centre = np.array(c[8:11]) * M_TO_MM
            nrm = np.array(c[11:14])
            xa = np.array(c[14:17])
            r = c[17] * M_TO_MM
            if c[7] == b"-":
                nrm = -nrm
            ya = np.cross(nrm, xa)
            sgn = 1.0 if fwd else -1.0
            if start is None or end is None:  # ring edge: full turn
                a0, sweep = 0.0, 2 * math.pi * sgn
            else:
                a0 = math.atan2(np.dot(start - centre, ya), np.dot(start - centre, xa))
                a1 = math.atan2(np.dot(end - centre, ya), np.dot(end - centre, xa))
                sweep = (a1 - a0) * sgn % (2 * math.pi)
                if sweep < 1e-9:
                    sweep = 2 * math.pi
                sweep *= sgn
            n = max(2, int(math.ceil(abs(sweep) / (2 * math.pi) * ARC_SEGMENTS)))
            angs = a0 + sweep * np.arange(1, n + 1) / n
            pts = [centre + r * (math.cos(a) * xa + math.sin(a) * ya) for a in angs]
            if end is not None:
                pts[-1] = end
            return pts
        return [end] if end is not None else []

    bodies: dict[int, Body] = {}
    names = _names(x)
    shell_body = {k: v[3] for k, v in S.items()}
    for _fk, fa in FA.items():
        shell = fa[7]
        if shell not in shell_body:
            continue
        b = shell_body[shell]
        body = bodies.setdefault(b, Body(index=b, name=_body_name(names.get(b, []))))
        loops: list[np.ndarray] = []
        lk = fa[6]
        seen_loops = set()
        while lk in L and lk not in seen_loops:
            seen_loops.add(lk)
            lp = L[lk]
            fins = []
            f0 = lp[3]
            fk2 = f0
            while fk2 in F and len(fins) < 10000:
                fins.append(F[fk2])
                fk2 = F[fk2][3]
                if fk2 == f0:
                    break
            pts: list[np.ndarray] = []
            for j, f in enumerate(fins):
                prev = fins[j - 1]
                end = pt(f[5]) if f[5] in V else None
                start = pt(prev[5]) if prev[5] in V else None
                pts.extend(fin_points(f, start, end))
            if len(pts) >= 3:
                arr = np.array(pts, dtype=float)
                keep = np.ones(len(arr), bool)
                d = np.linalg.norm(arr - np.roll(arr, 1, axis=0), axis=1)
                keep[d < 1e-6] = False
                arr = arr[keep]
                if len(arr) >= 3:
                    loops.append(arr)
            lk = lp[5]
        if not loops:
            continue
        surf = fa[8]
        outward = fa[9] == b"+"
        if surf in PL:
            p = PL[surf]
            n = np.array(p[11:14])
            if p[7] == b"-":
                n = -n
            n = n if outward else -n
            body.faces.append(Face("plane", loops, normal=n / np.linalg.norm(n),
                                   origin=np.array(p[8:11]) * M_TO_MM))
        elif surf in CY:
            c = CY[surf]
            sgn = (1.0 if c[7] == b"+" else -1.0) * (1.0 if outward else -1.0)
            ax = np.array(c[11:14])
            body.faces.append(Face("cylinder", loops, axis=ax / np.linalg.norm(ax),
                                   origin=np.array(c[8:11]) * M_TO_MM,
                                   radius=c[14] * M_TO_MM, outward=sgn))
        else:
            body.faces.append(Face("other", loops))
    out = [b for b in bodies.values() if b.faces]
    if not out:
        raise ParasolidError("no solid bodies could be rebuilt")
    return out
