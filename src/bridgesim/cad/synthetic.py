"""Exact box-shaped solid bodies, built in code (for tests, examples and quick studies).

A box made here goes through exactly the same pipeline as a body read from SolidWorks:
it has planar faces with outward normals and boundary loops, a closed triangle mesh and
mass properties.
"""

from __future__ import annotations

import numpy as np

from bridgesim.cad import CadBody
from bridgesim.cad.mesh import tessellate
from bridgesim.cad.parasolid import Body, Face


def box(name: str, lo, hi) -> CadBody:
    """Axis-aligned box from corner ``lo`` to corner ``hi`` (mm)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    P = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    quads = [((0, 3, 2, 1), (0, 0, -1)), ((4, 5, 6, 7), (0, 0, 1)),
             ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
             ((0, 4, 7, 3), (-1, 0, 0)), ((1, 2, 6, 5), (1, 0, 0))]
    faces = [Face("plane", [P[list(q)]], normal=np.array(n, float), origin=P[q[0]])
             for q, n in quads]
    body = Body(index=0, name=name, faces=faces)
    return CadBody(name, tessellate(body), faces, 0, named_in_cad=True)


def prism(name: str, outline_xz, y0: float, y1: float) -> CadBody:
    """A flat extrusion: the simple polygon ``outline_xz`` ((x, z) points, mm) in the
    horizontal plane, from height ``y0`` up to ``y1`` (a plate of any outline: an X, a
    frame, a gusset)."""
    Q = np.asarray(outline_xz, float)
    a2 = 0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))
    if a2 < 0:  # make it counter-clockwise in (x, z)
        Q = Q[::-1]
    lo = np.c_[Q[:, 0], np.full(len(Q), y0), Q[:, 1]]
    hi = np.c_[Q[:, 0], np.full(len(Q), y1), Q[:, 1]]
    faces = [Face("plane", [lo], normal=np.array([0.0, -1.0, 0.0]), origin=lo[0]),
             Face("plane", [hi[::-1]], normal=np.array([0.0, 1.0, 0.0]), origin=hi[0])]
    for i in range(len(Q)):
        j = (i + 1) % len(Q)
        d = Q[j] - Q[i]
        n = np.array([d[1], 0.0, -d[0]]) / np.linalg.norm(d)  # outward (counter-clockwise)
        faces.append(Face("plane", [np.array([lo[i], lo[j], hi[j], hi[i]])], normal=n,
                          origin=lo[i]))
    body = Body(index=0, name=name, faces=faces)
    return CadBody(name, tessellate(body), faces, 0, named_in_cad=True)


def simple_beam_bridge(span: float = 1000.0, deck_w: float = 150.0) -> list[CadBody]:
    """A minimal bridge: two piers, two stringers glued on top of them and a deck plate
    glued on the stringers. Small enough to simulate in seconds."""
    L = span + 100
    h = 150.0
    z = deck_w / 2
    return [
        box("Pier L (block)", (-L / 2, 0, -z), (-L / 2 + 50, h, z)),
        box("Pier R (block)", (L / 2 - 50, 0, -z), (L / 2, h, z)),
        box("Stringer Near (3-ply UD)", (-L / 2, h, z - 20), (L / 2, h + 19, z - 14)),
        box("Stringer Far (3-ply UD)", (-L / 2, h, -z + 14), (L / 2, h + 19, -z + 20)),
        box("Deck (2-ply 0-90)", (-L / 2, h + 19, -z), (L / 2, h + 23.1, z)),
    ]


def write_stl(bodies: list[CadBody], path) -> None:
    """Write bodies to one binary STL (mm), as SolidWorks 'save all bodies' does."""
    import struct
    from pathlib import Path

    tris = []
    for b in bodies:
        V, T = b.mesh.vertices, b.mesh.triangles
        tris.append(V[T])
    P = np.concatenate(tris).astype(np.float32)
    with Path(path).open("wb") as fh:
        fh.write(b"bridgesim synthetic".ljust(80, b" "))
        fh.write(struct.pack("<I", len(P)))
        for t in P:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n = n / (np.linalg.norm(n) or 1)
            fh.write(struct.pack("<12fH", *n, *t.ravel(), 0))
