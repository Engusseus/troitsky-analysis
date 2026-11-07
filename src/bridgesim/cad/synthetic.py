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


def upright(name: str, outline_xy, z0: float, z1: float) -> CadBody:
    """A flat extrusion standing upright: the simple polygon ``outline_xy`` ((x, y)
    points, mm) in a vertical plane, from z0 to z1 (a web, a diagonal, an L-shaped leg)."""
    flat = prism(name, [(x, -y) for x, y in outline_xy], z0, z1)
    Rm = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # (x, y, z) -> (x, -z, y)
    faces = [Face(f.kind, [lp @ Rm.T for lp in f.loops], normal=Rm @ f.normal,
                  origin=Rm @ f.origin) for f in flat.faces]
    body = Body(index=0, name=name, faces=faces)
    return CadBody(name, tessellate(body), faces, 0, named_in_cad=True)


def _strip(p, q, w):
    """The outline of a straight strip of width w from p to q (x, y)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = (q - p) / np.linalg.norm(q - p)
    n = np.array([-d[1], d[0]]) * w / 2
    return [tuple(p - n), tuple(q - n), tuple(q + n), tuple(p + n)]


def layered_truss(span: float = 1000.0, height: float = 140.0, panels: int = 4
                  ) -> list[CadBody]:
    """A light Pratt truss bridge (~0.4 kg of sticks), as teams build them from sticks:
    in each of the two trusses (near and far) the verticals, the chords and the
    diagonals lie in three layers, out from the middle, glued face to face where they
    cross; floor beams resting across the bottom chords (mid-panel) carry a deck under
    the loading plate, and struts across the top chords brace them. Support blocks under the
    chords' ends stand on the platform. A slender, soft bridge, unlike Bridge_v6. The
    trusses stand 104 mm apart, so the 90 mm loading plate fits between them."""
    out: list[CadBody] = []
    w = 52.0  # half the clear width between the trusses (the deck's)
    x0, x1 = -span / 2 - 40, span / 2 + 40  # the chords' ends over the support blocks
    y0 = 60.0  # top of the support blocks
    yt = y0 + height  # top of the top chords
    xs = np.linspace(-span / 2 + 100, span / 2 - 100, panels + 1)  # verticals
    out.append(box("Support L (block)", (x0, 0, -w - 28), (x0 + 60, y0, w + 28)))
    out.append(box("Support R (block)", (x1 - 60, 0, -w - 28), (x1, y0, w + 28)))
    for side, s in (("Near", 1.0), ("Far", -1.0)):
        # three layers per truss, out from the middle: verticals, chords, diagonals,
        # each web glued face to face to the chords where it crosses them
        zv = sorted((w * s, (w + 6) * s))
        zo = sorted(((w + 6) * s, (w + 12) * s))
        zd = sorted(((w + 12) * s, (w + 18) * s))
        out.append(box(f"Bottom Chord (3-ply UD) - {side}", (x0, y0, zo[0]),
                       (x1, y0 + 20, zo[1])))
        out.append(box(f"Top Chord (3-ply UD) - {side}", (xs[0] - 25, yt - 20, zo[0]),
                       (xs[-1] + 25, yt, zo[1])))
        for k, xv in enumerate(xs):
            out.append(box(f"Vertical {k} (3-ply UD) - {side}", (xv - 5, y0, zv[0]),
                           (xv + 5, yt, zv[1])))
        ends = [((x0 + 30, y0 + 10), (xs[0], yt - 10)),
                ((x1 - 30, y0 + 10), (xs[-1], yt - 10))]
        mid = len(xs) // 2
        inner = [((xs[k], yt - 10), (xs[k + 1], y0 + 10)) for k in range(mid)] + \
            [((xs[k + 1], yt - 10), (xs[k], y0 + 10)) for k in range(mid, len(xs) - 1)]
        for k, (p, q) in enumerate(ends + inner):
            p, q = np.asarray(p, float), np.asarray(q, float)
            # each end runs to a chord's centreline, set 10 mm along the chord toward
            # the diagonal's other end, so the two diagonals meeting at a node lie side
            # by side on the chord (each lapped over ~20 mm of it) instead of crossing
            dx = np.array([np.sign(q[0] - p[0]) * 10.0, 0.0])
            out.append(upright(f"Diagonal {k} (3-ply UD) - {side}",
                               _strip(p + dx, q - dx, 10.0), zd[0], zd[1]))
    # floor beams across the two bottom chords, resting on them (glued, bearing) at the
    # middle of each panel, clear of the verticals; the deck on the two middle ones,
    # between the verticals, under the loading plate
    xf = 0.5 * (xs[:-1] + xs[1:])
    for k, xm in enumerate(xf):
        out.append(box(f"Floor Beam {k} (3-ply UD)", (xm - 5, y0 + 20, -w - 12),
                       (xm + 5, y0 + 26, w + 12)))
    mid = len(xf) // 2
    out.append(box("Deck (2-ply 0-90)", (xf[mid - 1] - 10, y0 + 26, -w),
                   (xf[mid] + 10, y0 + 30, w)))
    mid = len(xs) // 2
    for k in (0, mid, len(xs) - 1):
        out.append(box(f"Top Strut {k} (3-ply UD)", (xs[k] - 5, yt, -w - 12),
                       (xs[k] + 5, yt + 6, w + 12)))
    return out


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


def write_stl(bodies: list[CadBody], path, names: bool = False) -> None:
    """Write bodies to one binary STL (mm), as SolidWorks 'save all bodies' does; with
    ``names``, an ASCII STL with one named solid per body (names kept on reading)."""
    import struct
    from pathlib import Path

    if names:
        with Path(path).open("w", encoding="ascii") as fh:
            for b in bodies:
                fh.write(f"solid {b.name}\n")
                for t in b.mesh.vertices[b.mesh.triangles]:
                    n = np.cross(t[1] - t[0], t[2] - t[0])
                    n = n / (np.linalg.norm(n) or 1)
                    fh.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n"
                             "    outer loop\n")
                    for v in t:
                        fh.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    fh.write("    endloop\n  endfacet\n")
                fh.write(f"endsolid {b.name}\n")
        return

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
