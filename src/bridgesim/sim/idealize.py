"""Turn CAD bodies into the structural model of the crush simulation.

Every solid body is looked at the way an engineer would idealise it:

* **bar** (a member: long, with a compact cross-section): a chain of corotational fibre
  beam elements on its centre line, with its real cross-section and laminate;
* **shell** (a plate: gusset, web, deck, diaphragm): triangular shell elements on its
  mid-surface, meshed from the body's real outline (holes included);
* **block** (a chunky piece: cleat, saddle, toggle): a few rigid cells of its wood
  (block.cell_mm), each with its mass and rotational inertia, joined by the wood's own
  stiffness and strength across the planes between them (a rigid-body-spring model), so
  it deforms and can split; one rigid body with block.cell_mm 0 or kind ``rigid``;
* **cable** (floss): a tension-only cable between the bodies it is tied to.

Bodies are joined where they really touch: every pair of coplanar faces that overlap
becomes a glued *patch* of the overlap's exact shape, modelled by a grid of cohesive glue
springs attached to each body through rigid arms at their real position (so eccentric
joints, and joints loaded in peel, behave as built). Anything the builder cannot decide
from the file (an unknown layup, a body that touches nothing, a floss end not tied to
anything) is recorded as a *question* with the assumption made, so a person or an AI
agent can answer it in an overrides file and run again.
"""

from __future__ import annotations

import dataclasses
import difflib
import fnmatch
import math
import re
import zlib
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, Point, Polygon

from bridgesim.cad import CadBody
from bridgesim.cad.mesh import face_polygon, plane_basis
from bridgesim.schema import Material
from bridgesim.sim import shell as shellmod
from bridgesim.sim.materials import (
    LaminateProps,
    Layup,
    Ply,
    SimParams,
    inplane_split_strength,
    laminate,
    parse_layup,
    parse_override_layup,
    rect_torsion,
)
from bridgesim.sim.materials import fibre_spread as _exact
from bridgesim.sim.model import G_MM_S2, TONNE_PER_KG, BodyInfo, Model, Patch

CONTACT_TOL_MM = 0.3
#: faces drawn this close but not touching (a gap, or one face inside the other part) are
#: glued as if they touched, and asked about (near_miss): a member drawn 0.4 mm thinner
#: than its gussets' gap lost its eight 27,770 mm^2 of laps and read 37-49 % low, with no
#: question (the adversarial review of round 26); closer than this (a glue line fills
#: nothing at 1 mm: glue.gap_fill_mm, gap_strength)
NEAR_GLUE_MM = 0.8
#: up to this far apart they are asked about, not glued
NEAR_ASK_MM = 3.0
MIN_PATCH_MM2 = 2.0


@dataclass
class Overrides:
    """What a designer (or an AI agent) tells the simulation that the CAD file does not.

    ``bodies`` maps a body name or glob pattern (``"Floss*"``) to fields: ``kind``
    (bar/shell/frame/block/rigid/cable/ignore: ``block`` a compact block of the wood's
    cells, ``rigid`` one rigid body, as blocks were until round 27: a steel stand-in, a
    load hanger), ``layup`` (any text bridgesim can parse, e.g.
    ``"6-ply UD"``), ``grain`` (``long``/``span``/``arm``), ``glue_factor`` (multiplies the
    strength of every glue line on that body, e.g. 0.5 for a poorly glued part) and
    ``strength_factor``; ``butt_stagger_mm`` is the distance between the butt joints of
    adjacent plies of a laminated member longer than one stick (0: no butt joints, e.g.
    built from full-length strips). ``section_scale`` is a what-if: it scales a member's
    cross-section (both sides) or a plate's thickness, and its mass, without changing the
    CAD or its glued areas. ``params`` overrides values of ``materials/simulation.yaml``
    (e.g. ``floss.strands: 6``). ``glue`` maps ``"Body A | Body B"`` to a factor.
    """

    bodies: dict[str, dict] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    glue: dict[str, float] = field(default_factory=dict)
    solid: list[str] = field(default_factory=list)  # glue keys set to "solid": one piece
    acknowledge: list[str] = field(default_factory=list)  # question ids (or patterns)
    # params keys whose values were measured (calibrate writes them); any other value set
    # under params stays a placeholder (assumed), still drawn by montecarlo around it
    measured: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)  # top-level keys we do not read
    used: set = field(default_factory=set)  # patterns and glue keys that matched

    KINDS = ("bar", "shell", "frame", "block", "rigid", "cable", "ignore")
    FIELDS = ("kind", "layup", "grain", "glue_factor", "strength_factor", "butt_stagger_mm",
              "butt_gap_mm", "butt_at", "section_scale", "axis")

    def for_body(self, name: str) -> dict:
        out: dict = {}
        for pat, val in self.bodies.items():
            if pat == name or fnmatch.fnmatchcase(name, pat):
                out.update(val)
                self.used.add(pat)
        return out

    def problems(self, body_names: list[str]) -> list[str]:
        """Everything in the overrides that had no effect (a typo, an unknown kind, a
        layup bridgesim cannot read, a joint that does not exist)."""
        out = [f"overrides: top-level key {k!r} is not read (bodies, glue, params, "
               "measured and acknowledge are; 'cad' and 'cli' in a question's answer_with "
               "are things to do, not overrides)" for k in self.unknown]
        out += [f"overrides: measured {k!r} is not set under params" for k in self.measured
                if k not in self.params]
        for pat, val in self.bodies.items():
            if pat not in self.used:
                stem = pat.rstrip("*").lower()
                close = difflib.get_close_matches(
                    stem, [n.lower()[: len(stem)] for n in body_names], n=1, cutoff=0.7)
                close = sorted({n for n in body_names if close and
                                n.lower().startswith(close[0])})[:3]
                out.append(f"overrides: body pattern {pat!r} matches no body"
                           + (f" (did you mean {close}?)" if close else ""))
            if not isinstance(val, dict):
                out.append(f"overrides: body {pat!r} must map to fields, got {val!r}")
                continue
            for k, v in val.items():
                if k not in self.FIELDS:
                    out.append(f"overrides: unknown field {k!r} for {pat!r} "
                               f"(known: {', '.join(self.FIELDS)})")
                elif k == "butt_at" and butt_at_of(v) is None:
                    out.append(f"overrides: butt_at {v!r} for {pat!r} must be a list of "
                               "[fraction of the member's length (0-1), [ply indices from "
                               "0]], e.g. [[0.5, [0, 4]]]: the outer plies of a 5-ply member "
                               "butted at mid-length")
                elif k == "kind" and v not in self.KINDS:
                    out.append(f"overrides: kind {v!r} for {pat!r} is not one of "
                               f"{', '.join(self.KINDS)}")
                elif k == "layup" and parse_override_layup(str(v)) is None:
                    out.append(f"overrides: layup {v!r} for {pat!r} cannot be read (write "
                               "e.g. '10-ply UD', '2-ply 0-90', '4-ply +-45', 'block')")
                elif k == "grain" and grain_vectors_of(v) is None and v not in (
                        "long", "span", "arm"):
                    out.append(f"overrides: grain {v!r} for {pat!r} must be long, span, "
                               "arm (a frame's plies laid along each of its arms), or a "
                               "block's fibres: x, y, z, a direction [x, y, z], or one per "
                               "layer [[x, y, z], ...]")
                elif k == "axis" and (grain_vectors_of(v) is None
                                      or len(grain_vectors_of(v)) != 1):
                    out.append(f"overrides: axis {v!r} for {pat!r} must be x, y, z or a "
                               "direction [x, y, z]: the direction a member spans")
                elif k in ("section_scale", "glue_factor", "strength_factor") and not (
                        isinstance(v, (int, float)) and 0 <= v <= 10
                        and (k != "section_scale" or v > 0)):
                    out.append(f"overrides: {k} {v!r} for {pat!r} must be a number "
                               + ("above 0" if k == "section_scale" else ">= 0")
                               + " (at most 10)")
        for key in self.glue:
            if key not in self.used:
                out.append(f"overrides: glue joint {key!r} not found (write 'Body A | Body "
                           "B' with two bodies that are glued to each other)")
        return out

    @classmethod
    def from_dict(cls, d: dict | None) -> Overrides:
        d = d or {}
        ph = list(_placeholders(d))
        if ph:
            raise ValueError("overrides: still the placeholders of a question's answer_with "
                             "(fill them in with the designer's answer): "
                             + "; ".join(f"{k} = {v!r}" for k, v in ph))
        ack = d.get("acknowledge", []) or []

        def num(section, k, v):
            try:
                return float(v)
            except (TypeError, ValueError):
                raise ValueError(f"overrides: {section}.{k} must be a number, got {v!r}") \
                    from None

        g_raw = d.get("glue", {}) or {}
        solid = [k for k, v in g_raw.items() if str(v).strip().lower() in SOLID_WORDS]
        return cls(bodies=dict(d.get("bodies", {}) or {}),
                   params={k: num("params", k, v)
                           for k, v in (d.get("params", {}) or {}).items()},
                   glue={k: num("glue", k, v) for k, v in g_raw.items() if k not in solid},
                   solid=solid,
                   acknowledge=[str(a) for a in (ack if isinstance(ack, list) else [ack])],
                   measured=[str(x) for x in (d.get("measured") or [])],
                   unknown=[str(k) for k in d if k not in ("bodies", "params", "glue",
                                                           "acknowledge", "measured")])

    def acknowledged(self, qid: str) -> bool:
        """Whether a question was acknowledged ('the design is as drawn', 'confirmed')."""
        return any(qid == a or fnmatch.fnmatchcase(qid, a) for a in self.acknowledge)


SOLID_WORDS = ("solid", "one piece")  # a glue override: the two bodies are one piece


def _joint_matches(key: str, a: str, b: str) -> bool:
    """A glue override key 'A | B' (exact names or glob patterns, either order) against
    the joint between bodies a and b."""
    parts = [x.strip() for x in key.split("|")]
    if len(parts) != 2:
        return False
    p, q = parts
    return (fnmatch.fnmatchcase(a, p) and fnmatch.fnmatchcase(b, q)) or \
        (fnmatch.fnmatchcase(a, q) and fnmatch.fnmatchcase(b, p))


_PLACEHOLDER = re.compile(r"<[^<>]+>")


def _placeholders(d, path: str = ""):
    """(key, value) of every string in an overrides document that still holds a '<...>'
    placeholder of a question's answer_with."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _placeholders(v, f"{path}.{k}" if path else str(k))
            if isinstance(k, str) and _PLACEHOLDER.search(k):
                yield (path or "(top)", k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _placeholders(v, f"{path}[{i}]")
    elif isinstance(d, str) and _PLACEHOLDER.search(d):
        yield (path, d)


@dataclass
class Shape:
    kind: str
    layup: Layup
    origin: np.ndarray  # point on the big face
    n: np.ndarray  # big face outward normal
    e1: np.ndarray
    e2: np.ndarray
    poly: Polygon | None  # profile in (e1, e2)
    T: float  # thickness along n
    u: np.ndarray  # long axis (3D)
    w: np.ndarray  # short in-plane axis (3D)
    Lp: float
    Wp: float
    fill: float
    centroid: np.ndarray
    why: str = ""
    frame: list | None = None  # centre lines of the arms of a plate modelled as a frame


def _outward(n: np.ndarray, c: np.ndarray) -> float:
    """How far the normal n of a body centred at c (from the bridge's centre) faces away
    from the bridge's centre (its largest component's sign, for a body centred on the
    centre's plane): a gusset's outer face, whatever the CAD origin (the adversarial
    review of round 23: moving a bridge 300 mm changed which ply its gussets glue on). A
    plate lying flat (its normal within 45 degrees of y, up) faces up, whatever its
    height: a deck's first-named ply is its bottom one, on what it rests on (the
    adversarial review of round 24: whether a deck sat above or below the bridge's
    mid-height decided it, and a taller truss turned the floor beams' laps on its 0 ply
    into laps on its 90 ply)."""
    if abs(float(n[1])) > math.sqrt(0.5) * float(np.linalg.norm(n)):
        return float(n[1])
    d = float(np.dot(n, c))
    if abs(d) > 1e-6 * max(float(np.linalg.norm(c)), 1.0):
        return d
    k = int(np.argmax(np.abs(n)))
    return 1e-9 * float(np.sign(n[k]))


def _mrr(poly: Polygon) -> tuple[np.ndarray, np.ndarray, float, float]:
    r = shapely.minimum_rotated_rectangle(poly)
    c = np.asarray(r.exterior.coords)[:4]
    a, b = c[1] - c[0], c[2] - c[1]
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if la >= lb:
        return a / la, b / max(lb, 1e-9), la, lb
    return b / lb, a / max(la, 1e-9), lb, la


#: a solid extrusion this many times longer than its section is a member (a Timoshenko
#: beam, which bends and shears like a deep beam down to about this length); shorter, it
#: is a compact rigid block
BAR_MIN_ASPECT = 2.0


def analyse_shape(b: CadBody, layup: Layup, forced: str | None,
                  center: np.ndarray | None = None) -> Shape:
    """How a body is modelled (bar, shell, block, ...) and its geometry; ``center``: the
    bridge's centre, which a plate's normal faces away from (so its plies are listed from
    the face toward the bridge's middle, whatever the CAD origin)."""
    m = b.mesh
    cm = m.centroid - (np.zeros(3) if center is None else np.asarray(center, float))
    planes = []
    for _i, f in enumerate(b.faces):
        if f.kind != "plane":
            continue
        try:
            poly, o, e1, e2 = face_polygon(f)
        except Exception:  # noqa: BLE001
            continue
        planes.append((poly, o, e1, e2, f))
    if layup.kind in ("toothpick", "floss") or not planes:
        return _pca_shape(b, layup, forced)
    # The extrusion caps: a pair of opposite parallel faces of equal area whose area
    # times their distance is the body's volume. Prefer the biggest such pair.
    best = None
    for poly, o, e1, e2, f in planes:
        n = f.normal
        for poly2, o2, _a, _b, g in planes:
            if float(np.dot(g.normal, n)) > -0.99:
                continue
            if abs(poly2.area - poly.area) > 0.03 * poly.area:
                continue
            T = abs(float(np.dot(o2 - o, n)))
            if T <= 0:
                continue
            err = abs(poly.area * T - m.volume) / m.volume
            if err > 0.12:
                continue
            # the bigger face; of two the same size (the two faces of a plate, equal but
            # for round-off), the one facing away from the CAD origin, so that mirrored
            # parts list their plies from the same side (outward), not as round-off
            # decides (the adversarial review of round 20: mirrored gussets stacked
            # their unsymmetric 0/90 plies in opposite orders)
            if best is None or poly.area > best[0].area * (1 + 1e-6) or (
                    poly.area >= best[0].area * (1 - 1e-6)
                    and _outward(n, cm) > _outward(best[4].normal, cm)):
                best = (poly, o, e1, e2, f, T)
    extr = best is not None
    if best is None:  # fall back to the biggest face and its farthest opposite face
        poly, o, e1, e2, f = max(planes, key=lambda q: q[0].area)
        ds = [abs(float(np.dot(q[1] - o, f.normal))) for q in planes
              if float(np.dot(q[4].normal, f.normal)) < -0.99]
        if not ds:
            return _pca_shape(b, layup, forced)
        best = (poly, o, e1, e2, f, max(ds))
    poly, o, e1, e2, f, T = best
    n = f.normal
    u2, w2, Lp, Wp = _mrr(poly)
    u = u2[0] * e1 + u2[1] * e2
    w = w2[0] * e1 + w2[1] * e2
    fill = poly.area / max(Lp * Wp, 1e-9)
    sh = Shape("block", layup, o, n, e1, e2, poly, T, u, w, Lp, Wp, fill, m.centroid)
    if layup.kind == "block":
        sh.kind, sh.why = "block", "named a block"
    elif not extr:
        sh.kind, sh.why = "block", "not a simple extrusion"
    elif Lp / max(Wp, T) >= BAR_MIN_ASPECT and Wp <= 4 * T and fill >= 0.75:
        sh.kind, sh.why = "bar", (f"slender ({Lp:.0f} x {Wp:.1f} x {T:.1f} mm)"
                                  if Lp / max(Wp, T) >= 3 else
                                  f"deep member ({Lp:.0f} x {Wp:.1f} x {T:.1f} mm, a "
                                  "shear-flexible beam)")
    elif Lp / max(Wp, T) >= BAR_MIN_ASPECT and Wp <= 4 * T and fill >= 0.4:
        # a wedge or tapered strip (a lip, a filler cut along a diagonal): still a member;
        # its section follows the profile along its length
        sh.kind, sh.why = "bar", (f"tapered member ({Lp:.0f} x {Wp:.1f} x {T:.1f} mm, "
                                  f"{fill:.0%} of its bounding box)")
    elif Wp >= 3 * T:
        sh.kind, sh.why = "shell", f"thin plate ({T:.1f} mm thick)"
    else:
        sh.kind, sh.why = "block", f"compact ({Lp:.0f} x {Wp:.1f} x {T:.1f} mm)"
    if forced in ("bar", "shell", "block", "cable", "frame"):
        if forced == "bar" and not extr:
            return _pca_shape(b, layup, "bar")
        if forced == "frame" and not extr:
            return sh  # a frame is cut from a plate: needs an extrusion
        sh.kind, sh.why = forced, "set in overrides"
    return sh


def _pca_shape(b: CadBody, layup: Layup, forced: str | None) -> Shape:
    V = b.mesh.vertices
    c = b.mesh.centroid
    _, vecs = np.linalg.eigh(np.cov((V - c).T))
    u = vecs[:, 2]
    w = vecs[:, 1]
    n = vecs[:, 0]
    pu, pw, pn = (V - c) @ u, (V - c) @ w, (V - c) @ n
    Lp, Wp, T = np.ptp(pu), np.ptp(pw), np.ptp(pn)
    # short toothpicks (floss toggles, pins) are rigid pieces; long ones are members
    kind = forced or ("cable" if layup.kind == "floss" else
                      "bar" if Lp / max(Wp, T) >= 5 and Lp >= 60 else "block")
    sh = Shape(kind, layup, c, n, u, w, None, float(T), u, w, float(Lp), float(Wp),
               b.mesh.volume / max(Lp * Wp * T, 1e-9), c,
               why=f"{layup.kind} {Lp:.0f} x {Wp:.1f} x {T:.1f} mm")
    return sh


# ------------------------------------------------------------------------------ contacts


def find_patches(bodies: list[CadBody], outlines: list | None = None,
                 near: list | None = None
                 ) -> list[tuple[int, int, Polygon, np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray]]:
    """(body a, body b, overlap polygon in a's face basis, origin, e1, e2, normal a->b).
    ``outlines``: if given, the two faces' own outlines in that basis are appended to it,
    one (a's, b's) pair per patch (how far a gusset reaches past a member lapped on it).
    ``near``: if given, faces that face each other over an overlap but are drawn apart
    (a gap, + mm) or one inside the other part (-, mm) by more than CONTACT_TOL_MM are
    listed in it as (a, b, area, offset, index of its patch or None, a's normal); up to
    NEAR_GLUE_MM they are patches too."""
    reach = NEAR_ASK_MM if near is not None else CONTACT_TOL_MM
    lo = np.array([b.mesh.bbox[0] for b in bodies]) - reach
    hi = np.array([b.mesh.bbox[1] for b in bodies]) + reach
    faces = []
    for bi, b in enumerate(bodies):
        for f in b.faces:
            if f.kind != "plane":
                continue
            try:
                poly, o, e1, e2 = face_polygon(f)
            except Exception:  # noqa: BLE001
                continue
            if poly.area > 0.5:
                faces.append((bi, f.normal, o, e1, e2, poly, f))
    out = []
    for ia in range(len(bodies)):
        for ib in range(ia + 1, len(bodies)):
            if (lo[ia] > hi[ib]).any() or (lo[ib] > hi[ia]).any():
                continue
            fa = [x for x in faces if x[0] == ia]
            fb = [x for x in faces if x[0] == ib]
            for _a, na, oa, e1, e2, pa, _f in fa:
                for _b, nb, ob, _e1b, _e2b, _pb, fbf in fb:
                    if float(np.dot(na, nb)) > -0.995:
                        continue
                    off = float(np.dot(ob - oa, na))
                    if abs(off) > reach:
                        continue
                    ring = [np.c_[(lp - oa) @ e1, (lp - oa) @ e2] for lp in fbf.loops]
                    k = int(np.argmax([Polygon(r).area for r in ring]))
                    pbp = Polygon(ring[k], [r for i, r in enumerate(ring) if i != k]
                                  ).buffer(0)
                    inter = pa.intersection(pbp)
                    if inter.area < MIN_PATCH_MM2:
                        continue
                    if abs(off) > CONTACT_TOL_MM:
                        glued = abs(off) < NEAR_GLUE_MM
                        near.append((ia, ib, float(inter.area), off,
                                     len(out) if glued else None, na))
                        if not glued:
                            continue
                    out.append((ia, ib, inter, oa, e1, e2, na))
                    if outlines is not None:
                        outlines.append((pa, pbp))
    return out


def _sample_polygon(poly, spacing: float, max_pts: int = 25,
                    min_per_side: int = 1, exact_moment: bool = False) -> np.ndarray:
    """Points spread over a (multi)polygon on a grid aligned with its minimum rotated
    rectangle, so long thin strips (a deck glued along a stringer) get a row of points
    along their length instead of one point. ``min_per_side``: at least this many
    across each side of 3 mm or more (a glued patch sampled at one point, or in one
    row, would be a hinge about it). ``exact_moment``: the rows spread so that equal
    shares of the area at the points have the rectangle's own second moment about each
    axis (two rows at +-L / (2 sqrt 3), the Gauss points; n rows: their midpoints x
    sqrt(n^2 / (n^2 - 1))), so a glued patch resists turning with its full stiffness
    whatever the number of points (midpoints give 75 % with two, 89 % with three)."""
    if poly.is_empty:
        return np.zeros((0, 2))
    base = poly if isinstance(poly, Polygon) else poly.convex_hull
    try:
        u, w, Lu, Lw = _mrr(base)
    except Exception:  # noqa: BLE001
        c = np.array(poly.representative_point().coords[0])
        return c[None, :]
    c0 = np.asarray(shapely.minimum_rotated_rectangle(base).centroid.coords[0])
    sp_ = spacing
    for _ in range(8):
        nu = max(min_per_side if Lu >= 3.0 else 1, int(round(Lu / sp_)))
        nw = max(min_per_side if Lw >= 3.0 else 1, int(round(Lw / sp_)))
        if nu * nw > max_pts:
            sp_ *= 1.25
            continue
        a = (np.arange(nu) + 0.5) / nu * Lu - Lu / 2
        b = (np.arange(nw) + 0.5) / nw * Lw - Lw / 2
        if exact_moment:
            a = a * (math.sqrt(nu**2 / (nu**2 - 1.0)) if nu > 1 else 1.0)
            b = b * (math.sqrt(nw**2 / (nw**2 - 1.0)) if nw > 1 else 1.0)
        A, B = np.meshgrid(a, b, indexing="ij")
        pts = c0 + np.outer(A.ravel(), u) + np.outer(B.ravel(), w)
        pts = pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]
        if (len(pts) >= min(3, nu * nw) or sp_ < 0.5) and len(pts):
            return pts
        sp_ /= 1.5
    c = np.array(poly.representative_point().coords[0])
    return c[None, :]


# ------------------------------------------------------------------------------- meshing


def mesh_polygon(poly: Polygon, h: float, seeds: np.ndarray | None = None,
                 fixed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a polygon (holes allowed) with elements of about size h. Holes
    smaller than about h (floss holes, pin holes) are filled: they do not change the
    plate's stiffness or strength noticeably and would force tiny elements. ``fixed``:
    points that become nodes (those inside the polygon), the mesh's own points near them
    dropped: another plate's nodes over the face glued to this one, so the two meshes
    meet node on node."""
    if isinstance(poly, Polygon):
        keep = [r for r in poly.interiors if Polygon(r).area > (0.8 * h) ** 2]
        poly = Polygon(poly.exterior, keep)
    poly = poly.simplify(0.1 * h, preserve_topology=True)
    pts = []
    rings = [poly.exterior] + list(poly.interiors) if isinstance(poly, Polygon) else []
    if not rings:
        geoms = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon)]
        poly = max(geoms, key=lambda g: g.area)
        rings = [poly.exterior] + list(poly.interiors)
    for r in rings:
        c = np.asarray(r.coords)
        for a, b in zip(c[:-1], c[1:], strict=True):
            L = float(np.linalg.norm(b - a))
            k = max(1, int(math.ceil(L / h)))
            for t in np.arange(k) / k:
                pts.append(a + t * (b - a))
    bnd = np.array(pts)
    minx, miny, maxx, maxy = poly.bounds
    dy = h * math.sqrt(3) / 2
    inner = []
    for j, y in enumerate(np.arange(miny + dy / 2, maxy, dy)):
        off = (h / 2) * (j % 2)
        for x in np.arange(minx + off + h / 4, maxx, h):
            inner.append((x, y))
    inner = np.array(inner) if inner else np.zeros((0, 2))
    if len(inner):
        inner = inner[shapely.contains_xy(poly, inner[:, 0], inner[:, 1])]
        if len(inner):
            dist = shapely.distance(poly.boundary, shapely.points(inner))
            inner = inner[dist > 0.45 * h]
    allp = np.vstack([bnd, inner]) if len(inner) else bnd
    if seeds is not None and len(seeds):
        tree = cKDTree(allp)
        dd, _ = tree.query(seeds)
        inside = shapely.contains_xy(poly.buffer(1e-6), seeds[:, 0], seeds[:, 1])
        allp = np.vstack([allp, seeds[(dd > 0.55 * h) & inside]])
    allp = np.unique(np.round(allp, 6), axis=0)
    n_fix = 0
    keep_pt = np.zeros(len(allp), bool)  # points never merged away
    if fixed is not None and len(fixed):
        fx = np.unique(np.round(np.asarray(fixed, float), 6), axis=0)
        fx = fx[shapely.contains_xy(poly.buffer(1e-6), fx[:, 0], fx[:, 1])]
        if len(fx):
            # this plate's own outline nodes stay (the partner may have them too, or
            # needs them where its overlap ends); its inner points near a fixed one
            # give way to it
            on_bnd = shapely.distance(poly.boundary, shapely.points(allp)) < 1e-6
            dd, _ = cKDTree(fx).query(allp)
            sel = (dd > 0.5 * h) | on_bnd
            allp, on_bnd = allp[sel], on_bnd[sel]
            dup = cKDTree(fx).query(allp)[0] < 1e-6  # an outline node it has already
            allp, on_bnd = allp[~dup], on_bnd[~dup]
            allp = np.vstack([fx, allp])
            n_fix = len(fx)
            keep_pt = np.r_[np.ones(n_fix, bool), on_bnd]
    # merge points closer than 0.3 h (short CAD edges, seeds next to the boundary): they
    # would make sliver elements and a tiny stable time step (fixed points, another
    # mesh's nodes, and the outline's nodes beside them are kept unless nearly on top
    # of each other)
    tree = cKDTree(allp)
    drop = np.zeros(len(allp), bool)
    for i, j in sorted(tree.query_pairs(0.3 * h)):
        if drop[i] or drop[j]:
            continue
        if keep_pt[i] and keep_pt[j]:
            if np.linalg.norm(allp[i] - allp[j]) < 0.05 * h:
                drop[j if j >= n_fix else i] = True
            continue
        drop[j if not keep_pt[j] else i] = True
    allp = allp[~drop]
    tri = Delaunay(allp).simplices
    cen = allp[tri].mean(axis=1)
    keep = shapely.contains_xy(poly, cen[:, 0], cen[:, 1])
    tri = tri[keep]
    a, b, c = allp[tri[:, 0]], allp[tri[:, 1]], allp[tri[:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    tri = tri[np.abs(area) > 1e-3 * h * h]
    area = area[np.abs(area) > 1e-3 * h * h]
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
    # drop slivers (caps of nearly straight boundary chains): quality 4 sqrt(3) A / sum e^2
    P_ = allp[tri]
    e2 = sum(((P_[:, (k + 1) % 3] - P_[:, k]) ** 2).sum(1) for k in range(3))
    qual = 4 * math.sqrt(3) * np.abs(area) / e2
    tri = tri[qual > 0.15]
    used, inv = np.unique(tri, return_inverse=True)
    return allp[used], inv.reshape(-1, 3)


# --------------------------------------------------------------------------- the builder


class _Builder:
    def __init__(self) -> None:
        self.X: list[np.ndarray] = []
        self.mass: list[float] = []
        self.rot: list[float] = []
        self.node_body: list[int] = []
        self.att_node: list[int] = []
        self.att_r0: list[np.ndarray] = []
        self.imperf = None  # a global imperfection field applied to every point
        self.X_cad: list[np.ndarray] = []  # each node where the CAD puts it (no lean)

    def _at(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return p if self.imperf is None else self.imperf(p)

    def node(self, p, body: int) -> int:
        self.X_cad.append(np.asarray(p, float))
        self.X.append(self._at(p))
        self.mass.append(0.0)
        self.rot.append(0.0)
        self.node_body.append(body)
        return len(self.X) - 1

    def attach(self, node: int, p) -> int:
        self.att_node.append(node)
        self.att_r0.append(self._at(p) - self.X[node])
        return len(self.att_node) - 1


def grain_vectors_of(v) -> list[list[float]] | None:
    """The fibre directions a ``grain`` override gives a block, one unit vector per layer:
    'x', 'y' or 'z' (one axis), [x, y, z] (one direction), or [[x, y, z], ...] (each
    layer's: a heel seat with 6 of 13 layers along its post and 7 along its chord); None
    if it is none of these."""
    if isinstance(v, str):
        ax = {"x": 0, "y": 1, "z": 2}.get(v.strip().lower())
        if ax is None:
            return None
        return [[1.0 if k == ax else 0.0 for k in range(3)]]
    try:
        a = np.asarray(v, float)
    except (TypeError, ValueError):
        return None
    if a.shape == (3,):
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 3 or not len(a):
        return None
    nrm = np.linalg.norm(a, axis=1)
    if (nrm < 1e-9).any() or not np.isfinite(a).all():
        return None
    return (a / nrm[:, None]).tolist()


def grain_vectors(sh) -> list[np.ndarray]:
    """Grain direction of each ply of a bar or a plate (unit vectors); none for a block,
    unless its fibres are given (grain: an axis, a direction, or one per layer)."""
    if getattr(sh.layup, "grain_vecs", None):
        return [np.asarray(g, float) for g in sh.layup.grain_vecs]
    if sh.kind == "bar" or (sh.kind == "block" and sh.layup.kind == "laminate"
                            and sh.layup.source in ("name", "override")):
        u, w = sh.u, sh.w  # plies lie in the (u, w) plane, stacked through the thickness
    elif sh.kind in ("shell", "frame"):
        u = sh.u
        if sh.layup.grain == "span":
            gx = np.array([1.0, 0, 0]) - sh.n * sh.n[0]
            if np.linalg.norm(gx) > 0.3:
                u = gx / np.linalg.norm(gx)
        w = np.cross(sh.n, u)
    else:
        return []
    out = []
    for a in sh.layup.angles or [0.0]:
        g = math.cos(math.radians(a)) * u + math.sin(math.radians(a)) * w
        out.append(g / np.linalg.norm(g))
    return out


def hankinson(f_par: float, f_perp: float, cos_t: float) -> float:
    """Strength of wood loaded at an angle to the grain (Hankinson's formula; cos_t is
    the cosine between the load and the grain)."""
    c2 = min(1.0, cos_t * cos_t)
    return f_par * f_perp / (f_par * (1.0 - c2) + f_perp * c2)


def end_grain_share(cos_t: float, end_factor: float = 0.5) -> float:
    """How much a glued face is end grain, 0 (side grain) to 1 (end grain), for a ply
    whose fibres meet the face's normal at cos_t: the glue's strength on the face is
    1 - share x (1 - end_factor), interpolated like Hankinson's formula, g = e / (e (1 -
    c^2) + c^2), so a face cut obliquely across the fibres is partly end grain (a
    diagonal's foot cut at 55 degrees to its fibres: 0.78 of the way), not side grain up
    to 60 degrees and end grain past it (glued at 0.6 of side grain with end grain at
    0.5)."""
    c2 = min(1.0, cos_t * cos_t)
    if end_factor >= 0.999:
        return c2
    g = end_factor / (end_factor * (1.0 - c2) + c2)
    return (1.0 - g) / (1.0 - end_factor)


def grain_bearing(sh, n: np.ndarray, f_c: float, f_c90: float, end_factor: float = 0.5
                  ) -> tuple[float, float]:
    """Crushing strength of a body pressed along n (MPa), averaged over its plies, and
    how much of a face across n is end grain (end_grain_share, averaged)."""
    gs = grain_vectors(sh)
    if not gs:
        return f_c90, 0.0
    cs = [abs(float(g @ n)) for g in gs]
    return (float(np.mean([hankinson(f_c, f_c90, c) for c in cs])),
            float(np.mean([end_grain_share(c, end_factor) for c in cs])))


# the words that name a copy's side, standing on their own in a body's name ("End Post
# L Near", "gus_A_R Far Inner", "Diagonal - Left"): copies differ only in these
SIDE_WORDS = re.compile(r"(?<![A-Za-z0-9])(L|R|Left|Right|Near|Far)(?![A-Za-z0-9])")
SIDE_GLOB = {"L": "[LR]", "R": "[LR]", "Left": "[LR]*t", "Right": "[LR]*t",
             "Near": "[NF]*r", "Far": "[NF]*r"}


def _base(name: str) -> str:
    return re.sub(r"( - .*| \(mirror copy\).*| #\d+)$", "", name).strip()


def _group(name: str) -> str:
    """Name without its side / copy words: 'Top Chord (..) - Far' -> 'Top Chord (..)',
    'Knee-Diagonal L Near (13-ply UD)' -> 'Knee-Diagonal (13-ply UD)', 'gus_A_R Far
    Inner' -> 'gus_A Inner': the same for every copy of a part."""
    g = SIDE_WORDS.sub("", _base(name))
    g = re.sub(r"_+(?=[\s_]|$)", "", g)  # 'gus_A_ ' -> 'gus_A '
    return re.sub(r"\s+", " ", g).strip()


def copies_glob(name: str) -> str:
    """A glob that matches a part and its copies on the other sides ('End Post L Near
    (13-ply UD)' -> 'End Post [LR] [NF]*r (13-ply UD)', 'Top Chord - Far' -> 'Top
    Chord*'), and nothing else ('Body 4' does not match 'Body 41')."""
    import glob as _glob

    base = _base(name)
    parts = SIDE_WORDS.split(base)
    pat = "".join(_glob.escape(p) if i % 2 == 0 else SIDE_GLOB[p]
                  for i, p in enumerate(parts))
    if base != name:
        pat += "*"  # the suffix the copies carry
    return pat


def stick_factors(seed: int | None, name: str, n_plies: int, sp: SimParams
                  ) -> tuple[float, float]:
    """The seeded stiffness and strength factors of a body (lognormal, mean 1): a
    member of n plies averages its sticks, which are partly alike (``scatter.ply_corr``
    rho), so both vary by CoV x sqrt((1 + (n - 1) rho) / n); and a stiffer body is
    stronger, the two correlated by ``scatter.E_strength_corr`` (in log terms). Each
    draw comes from a stream named after the body; ``scatter.amplitude`` scales both."""
    if seed is None:
        return 1.0, 1.0
    amp = float(sp.values.get("scatter.amplitude", 1.0))
    r_p = float(sp.values.get("scatter.ply_corr", 0.0))
    n = max(1, int(n_plies))
    k_lam = math.sqrt((1 + (n - 1) * r_p) / n)
    rho = float(sp.values.get("scatter.E_strength_corr", 0.0))

    def z(key):
        return float(np.random.default_rng(
            [int(seed), zlib.crc32(key.encode("utf-8"))]).standard_normal())

    def ln(cov, zz):
        s_ = math.sqrt(math.log(1 + (cov * k_lam) ** 2)) * amp
        return math.exp(s_ * zz - 0.5 * s_ * s_)

    zE, zS = z(f"E:{name}"), z(f"S:{name}")
    return (ln(sp["scatter.E_cov"], zE),
            ln(sp["scatter.strength_cov"], rho * zE + math.sqrt(max(0.0, 1 - rho**2)) * zS))


def interlaminar_strength(angles, f_v: float, f_rolling: float, bond: float) -> float:
    """Shear strength of the glue lines between the plies of a laminate (plies at
    ``angles`` degrees to the member's axis): the lower of the bond line's own strength
    and what the weakest ply takes there, the interlaminar shear running along its grain
    (f_v) and across it (rolling shear) in proportion to the angle (a quadratic
    interaction): f_v for a UD stack, the rolling shear for a ply across the member."""
    worst = f_v
    for a in angles:
        c, s = abs(math.cos(math.radians(a))), abs(math.sin(math.radians(a)))
        worst = min(worst, 1.0 / math.hypot(c / f_v, s / max(f_rolling, 1e-9)))
    return min(bond, worst)


def cross_ply_sheet_shear(angles, t_over_a: float = 0.2,
                          tor_over_fv: float = math.inf) -> tuple[float, float]:
    """In-plane shear of a 0/90 sheet laid from sticks whose edges are not glued: its
    shear stiffness and strength factors (Bogensperger, Moosbrugger & Silly 2010, WCTE,
    cross-laminated timber with unglued narrow faces: G* / G0 = 1 / (1 + 6 F (t/a)^2),
    F = 0.5345 (t/a)^-0.7947 for 3 layers, 0.4253 (t/a)^-0.7941 for 5 or more, t/a a
    stick's thickness over its width; the shear in the boards' cross-sections doubles,
    so the nominal shear strength is halved). The glued crossings carry the shear by
    torsion, tau_T = 3 tau t* / a with t* = n t / (n - 1) the sheet's thickness per glued
    interface (Boggian, Andreolli & Tomasi 2019, Front. Built Environ. 5:58), which
    caps the nominal shear at ``tor_over_fv`` (a crossing's torsional strength over
    f_v) x a (n - 1) / (3 n t): 0.83 x the crossing's strength for 2 plies of 2 x 10 mm
    sticks, below half f_v once the crossings are weaker than ~5 MPa. (1, 1) for a
    sheet that is not 0/90."""
    fam = {round(((float(a) % 180) + 180) % 180 / 90.0) % 2 for a in angles}
    near = all(min(abs(float(a) % 90), 90 - abs(float(a) % 90)) < 10 for a in angles)
    if len(fam) < 2 or not near:
        return 1.0, 1.0
    F = (0.5345 * t_over_a ** -0.7947 if len(angles) <= 3
         else 0.4253 * t_over_a ** -0.7941)
    n = len(angles)
    tor = tor_over_fv * (n - 1) / (3.0 * n * t_over_a) if n > 1 else math.inf
    return 1.0 / (1.0 + 6.0 * F * t_over_a**2), min(0.5, tor)


def lap_size_factor(area_mm2: float, ref_mm2: float = 200.0, m: float = 10.0,
                    cap_mm2: float = 3000.0) -> float:
    """The average strength of a hand-made glued lap of ``area_mm2`` over that of the
    coupon tau_g is measured on (``ref_mm2``, the 20 x 10 mm lap): a Weibull size effect
    of modulus ``m``, (A / A_ref)^(-1/m), for a lap larger than the coupon (more starved
    spots and uneven clamping over a larger area: bonded timber laps follow a Weibull
    size effect, Tannert, Vallee & Hehl 2012, Wood Sci. Technol. 46:503), no further than
    ``cap_mm2``: pressed glued crossings of 1600 to 29,000 mm^2 showed none (Flaig &
    Blass 2013, CIB-W18 46-12-1), so the law is not carried past the laps it is meant
    for. The fracture part of the size effect (shear lag along a long lap) is in the glue
    law itself, so m is the statistical part only: ~10 (6-20)."""
    if area_mm2 <= ref_mm2 or m <= 0:
        return 1.0
    return float((min(area_mm2, max(cap_mm2, ref_mm2)) / ref_mm2) ** (-1.0 / m))


T90_REF_VOLUME_MM3 = 4800.0  # Kruger et al. 2025's coupons: an 80 x 30 mm field, 2 mm
T90_DEPTH_MM = 2.0  # how deep into the wood a glue line's peel stresses it (a ply)
T90_FACTOR_MAX = 2.5


def t90_size_factor(area_mm2: float, m: float = 8.0) -> float:
    """The tension strength across the grain of the small volume of wood a glued joint
    pulls on, over that of the coupons wood.f_t90_MPa comes from: a weakest-link size
    effect (V_ref / V)^(1/m), V = the joint's area x T90_DEPTH_MM, V_ref = 4800 mm^3,
    at least 1 (a big joint's peel stays at its edges) and at most T90_FACTOR_MAX. Birch
    veneer pulled across the grain scatters with CoV 25 / 12.6 / 10.8 % at 1 / 2 / 3 mm
    (m ~ 5 / 9.5 / 11, Kruger et al. 2025, doi:10.5281/zenodo.17201843); Douglas fir
    ~5 (Barrett 1974, Wood Fiber 6:126, the basis of Eurocode 5's (V0/V)^0.2). A
    vertical's end grain glued on a chord's side (47 mm^2) held 2.6 MPa in a published
    truss test (Moraes et al. 2019): 1.7 x f_t90, where m = 8 gives 1.64. Caps
    the peel of a glue line and the end-grain joints (wood.t90_weibull_m)."""
    if m <= 0 or area_mm2 <= 0:
        return 1.0
    v = area_mm2 * T90_DEPTH_MM
    return float(min(max((T90_REF_VOLUME_MM3 / v) ** (1.0 / m), 1.0), T90_FACTOR_MAX))


# shear strength of 1.4 mm birch veneer loaded at 0, 15, ..., 90 degrees to its grain over
# that along it, on 40 x 40 mm (Wang, Wang, Debertolis, Crocetti, Walinder & Blomqvist
# 2024, Eur. J. Wood Wood Prod. 82:1407, doi:10.1007/s00107-024-02097-9: 6.4 / 5.7 / 4.4 /
# 4.0 / 4.2 / 3.8 / 4.6 MPa; a dip at 45-75 degrees Hankinson's form does not follow)
VENEER_ANGLE_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
VENEER_SHEAR_RATIO = (1.0, 0.89, 0.69, 0.625, 0.656, 0.594, 0.719)


def face_shear_strength(f_v: float, f_rolling: float, cos_t: float) -> float:
    """Shear strength of the ply a glue line lies on, sheared at an angle to its grain
    (cos_t between the force and the grain, in the glue plane): the lower of birch
    veneer's measured ratio at that angle x f_v (VENEER_SHEAR_RATIO: 0.62 at 45 degrees,
    0.72 at 90, with a dip at 45-75 degrees) and Hankinson's form between f_v along the
    grain and the rolling shear f_rolling across it (4.0 of 8 at 90 degrees)."""
    c = min(abs(cos_t), 1.0)
    deg = math.degrees(math.acos(c))
    r = float(np.interp(deg, VENEER_ANGLE_DEG, VENEER_SHEAR_RATIO))
    return min(f_v * r, hankinson(f_v, f_rolling, c))


def panel_factor(h_f: float, b_w: float, sides: int) -> float:
    """What a panel (a gusset) passes in rolling shear under a part glued on its face
    (a member: the web) as a share of its rolling shear strength: (4 sides h_f / b_w)^0.8,
    at most 1, h_f the panel's thickness, b_w the glued width across the force, sides
    the number of sides on which the panel reaches past the web (EN 1995-1-1 9.1.2 eq.
    9.14 for glued thin-flanged beams: 8 h_f inside a panel, 4 h_f at its edge, the
    U-section; APA PDS Supplement 3 3.6.4: half the rolling shear at the edge stringers).
    A panel that reaches past the web on neither side is sheared evenly over the glued
    face, as a lap of two sticks is (1)."""
    if sides <= 0 or b_w <= 0 or h_f <= 0:
        return 1.0
    return float(min(1.0, (4.0 * sides * h_f / b_w) ** 0.8))


def gap_strength(gap: float, fill: float = 0.15) -> float:
    """Strength of a white-glue line filling a gap ``gap`` mm wide, as a share of a thin
    one: full up to ``fill`` (glue.gap_fill_mm), then falling as the thick line shrinks
    and voids, 0.9 at fill + 0.1 mm, 0.6 at fill + 0.35 and nothing from fill + 0.85
    (with fill 0.15: 0.25, 0.5 and 1.0 mm; shimmed block-shear tests, Kurt 2006, Wood Res.
    51(1):59, -10 % at 0.25 and -39 % at 0.48 mm; a 0.4 mm PVAc line carried 1.38 MPa for
    70 days, River & Gillespie 1981, USDA FPL RP-400; with PVAc a loose mortise and tenon
    lost little, Tankut 2007)."""
    x = gap - fill
    if x <= 0:
        return 1.0
    return float(np.interp(x, [0.0, 0.1, 0.35, 0.85], [1.0, 0.9, 0.6, 0.0]))


def butt_path_capacity(n: int, s: float, Ls: float, P_ply: float, tau_w: float) -> float:
    """Tension capacity of ``n`` plies (or stick rows) of sticks ``Ls`` long whose butt
    joints are staggered by ``s`` from one ply to the next (ply i's joints at i s, every
    ``Ls``): the weakest path through the plies. A path crosses each ply either at one of
    its joints (free) or through its fibres (``P_ply``), and between neighbouring plies
    shears the glue line over the distance between their crossings (``tau_w`` per mm).
    Joints repeat every stick, so in a brick pattern (s = Ls / 2) every other ply is cut at
    the same section: the net section keeps n // 2 plies, not n - 1. Found by dynamic
    programming over the joint positions (where a path's optimum crossings lie)."""
    if n < 1:
        return 0.0
    pos = np.unique(np.round(np.mod(np.arange(n) * s, Ls), 6))
    joint = np.isclose(np.mod(np.arange(n)[:, None] * s, Ls), pos[None, :], atol=1e-6) | \
        np.isclose(np.mod(np.arange(n)[:, None] * s, Ls) - Ls, pos[None, :], atol=1e-6)
    d = np.abs(pos[:, None] - pos[None, :])
    d = np.minimum(d, Ls - d)  # the nearest joint of the next ply, either way
    cost = np.where(joint[0], 0.0, P_ply)
    for i in range(1, n):
        cost = np.where(joint[i], 0.0, P_ply) + (cost[:, None] + tau_w * d).min(0)
    return float(cost.min())


def _laminated_member(name: str, sh, lam, L: float, mat: Material, o: dict,
                      questions: list, f_rolling: float = 2.0,
                      bond: float = 6.0, butt_eff: float = 1.0) -> tuple[float, float]:
    """Tension-strength factor and shear strength of a member laminated from sticks.

    A ply longer than one stick has butt joints; its load goes round each joint through
    the glue lines to the next plies. The tension capacity is the weakest path through
    the plies (:func:`butt_path_capacity`): straight through the joints that line up
    (every other ply in a brick pattern), or zigzagging from joint to joint and shearing
    the glue lines between them at the laminate's interlaminar strength (the pressed bond
    lines and the plies next to them, as where a member splits), times ``butt_eff``: in
    plies as thin as a stick the glue line cracks from a butt joint's tips well before
    the member fails and the fracture runs into the plies, below the statics (Jung &
    Murphy 1983; wood.butt_tension_efficiency). Bending shear between
    plies runs in the same glue lines: :func:`interlaminar_strength` (the bond line,
    pressed while it cured, ``bond``; the plies along and across their grain,
    ``f_rolling``)."""
    n = len(lam.angles)
    f_v = mat.f_v_MPa.value
    if sh.layup.kind != "laminate" or n < 2:
        return 1.0, f_v
    gf = float(o.get("glue_factor", 1.0))
    fv = interlaminar_strength(lam.angles, f_v, f_rolling, bond * max(gf, 1e-3))
    Ls = mat.stick.length_mm
    if L <= 1.02 * Ls or o.get("butt_stagger_mm") == 0:
        return 1.0, fv
    s = float(o.get("butt_stagger_mm") or Ls / 2)  # brick pattern: half a stick
    w = sh.Wp  # glue-line width (plies stacked through the thickness)
    P_ply = lam.f_t * sh.Wp * sh.T / n
    P = butt_path_capacity(n, s, Ls, P_ply, fv * w)
    kt = min(1.0, P / (n * P_ply))
    if kt < 1.0:
        kt *= butt_eff
    g = _group(name)
    if o.get("butt_stagger_mm") is None and not any(
            q["id"] == f"butt_joints:{g}" for q in questions):
        questions.append(_butt_question(g, name, L, Ls, s, kt))
    return kt, fv


def bearing_depth(ell: float, b: float, h: float, ca: float = 1.0, cb: float = 0.33
                  ) -> float:
    """The depth of wood that would deform as much as a part ``h`` deep does under a
    face ``ell`` long and ``b`` wide pressed into it, the stress spreading at slope
    ``ca`` along ``ell`` and ``cb`` along ``b`` (45 degrees along the grain, ~1/3 across
    it): ell b times the integral over the depth of 1 / ((ell + 2 ca z)(b + 2 cb z))
    (Blass & Gorlacher 2004, with spread across the grain added)."""
    if h <= 0:
        return 0.0
    z = np.linspace(0.0, h, 201)
    f = ell * b / ((ell + 2.0 * ca * z) * (b + 2.0 * cb * z))
    return float(np.trapezoid(f, z))


# ------------------------------------------------------------------- compact blocks

BLOCK_MAX_CELLS = 3  # cells along each of a block's axes at most
BLOCK_MIN_CELLS = 1.5  # a block shorter than this many cells along an axis is not cut
BLOCK_MERGE_SHARE = 0.25  # a cell filled less than this (a corner a chamfer cuts off) joins
#                           its neighbour across the largest face
BLOCK_SUBSAMPLES = 6  # sample points per cell and axis for a block that is not a box


def _inside_mesh(P: np.ndarray, V: np.ndarray, T: np.ndarray,
                 d: np.ndarray | None = None) -> np.ndarray:
    """Whether each point of P lies inside the closed triangle mesh (V, T): the parity of
    the crossings of a ray from it (Moller-Trumbore, vectorised over the triangles, in
    chunks), along ``d`` (a direction tilted off the mesh's faces and edges)."""
    P = np.asarray(P, float).reshape(-1, 3)
    if d is None:
        d = np.array([1.0, 3.17e-4, 7.31e-4])
    d = np.asarray(d, float) / np.linalg.norm(d)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out = np.zeros(len(P), bool)
    step = max(1, 200_000 // max(len(T), 1))
    for s0 in range(0, len(P), step):
        s = P[s0:s0 + step, None, :] - a[None, :, :]
        u = (s * h[None]).sum(2) * inv[None]
        q = np.cross(s, e1[None, :, :])
        v = (q @ d) * inv[None]
        t = (q * e2[None]).sum(2) * inv[None]
        hit = ok[None] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        out[s0:s0 + step] = hit.sum(1) % 2 == 1
    return out


@dataclass
class BlockLattice:
    """A compact block cut into cells (a rigid-body-spring model of its wood): its
    bounding box in its own axes cut by planes normal to them, each lattice cell (or a
    group of them, where a cell is mostly outside the solid) a rigid cell with its own
    mass and rotational inertia, neighbouring cells joined across each lattice face."""

    centre: np.ndarray  # (3,) the solid's centroid: the origin of the block's frame
    axes: np.ndarray  # (3, 3) rows: the block's axes (its long axis, its width, across)
    lo: np.ndarray  # (3,) low corner of its bounding box, in its frame
    size: np.ndarray  # (3,) lattice cell size along each axis (mm)
    k: np.ndarray  # (3,) cells along each axis
    group: np.ndarray  # (k0, k1, k2): the cell of each lattice cell (-1: outside the solid)
    X: np.ndarray  # (G, 3) each cell's centroid (CAD coordinates)
    mass: np.ndarray  # (G,) t
    rot: np.ndarray  # (G,) mean principal moment of inertia about its centroid, t mm^2
    # one per lattice face between two different cells: (axis, lower cell, upper cell,
    # centre (3,), points (P, 3) on the face, area mm^2)
    faces: list = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return len(self.mass)

    def cell_at(self, p) -> int:
        """The cell holding the point p (CAD coordinates): the lattice cell it falls in,
        or, if that one is outside the solid, the cell whose centroid is nearest."""
        q = self.axes @ (np.asarray(p, float) - self.centre)
        i = np.clip(np.floor((q - self.lo) / self.size).astype(int), 0, self.k - 1)
        g = int(self.group[i[0], i[1], i[2]])
        if g >= 0:
            return g
        return int(np.argmin(np.linalg.norm(self.X - np.asarray(p, float), axis=1)))


def block_axes(sh) -> np.ndarray:
    """A block's axes (rows, right-handed): its outline's long axis, its width and the
    direction across both (the extrusion's normal for an extruded block)."""
    u = np.asarray(sh.u, float)
    u = u / np.linalg.norm(u)
    w = np.asarray(sh.w, float)
    w = w - float(w @ u) * u
    w = w / np.linalg.norm(w)
    return np.array([u, w, np.cross(u, w)])


def block_lattice(b: CadBody, sh, cell_mm: float, mass_t: float, J_t: float,
                  points_per_side: int = 3) -> BlockLattice:
    """Cut a compact block into cells of about ``cell_mm`` (1 to BLOCK_MAX_CELLS along each
    of its axes; an axis shorter than BLOCK_MIN_CELLS cells is not cut) and give each cell
    the mass, centroid and rotational inertia of the wood in it. A box is cut exactly; any
    other solid is sampled on a grid (BLOCK_SUBSAMPLES^3 points a cell, each standing for
    its little box), a cell filled less than BLOCK_MERGE_SHARE joining the neighbour it
    shares the most face with, and the cells' masses and positions are then scaled and
    shifted so that the total mass ``mass_t``, the centre of gravity and the mean
    principal moment ``J_t`` (the round-27 rigid block's) are the solid's exactly. Each
    lattice face between two different cells gets ``points_per_side``^2 points spread so
    that their equal shares of its area have its second moment (the glue grids'
    exact_moment), inside the solid."""
    A = block_axes(sh)
    c0 = np.asarray(b.mesh.centroid, float)
    Lv = (b.mesh.vertices - c0) @ A.T
    lo, hi = Lv.min(0), Lv.max(0)
    ext = np.maximum(hi - lo, 1e-9)
    k = np.array([1 if e < BLOCK_MIN_CELLS * cell_mm else
                  min(BLOCK_MAX_CELLS, int(e / cell_mm + 0.5)) for e in ext])
    size = ext / k
    vbox = float(np.prod(ext))
    is_box = abs(float(b.mesh.volume) - vbox) <= 1e-6 * vbox
    idx = np.array(np.meshgrid(*[np.arange(n) for n in k], indexing="ij")).reshape(3, -1).T
    ncell = len(idx)
    vol = np.zeros(ncell)
    cen = np.zeros((ncell, 3))
    tr = np.zeros(ncell)  # trace of the inertia tensor about the cell's centroid / rho
    ns = BLOCK_SUBSAMPLES
    inside3 = None
    if is_box:
        vol[:] = float(np.prod(size))
        cen[:] = lo + (idx + 0.5) * size
        tr[:] = vol * float((size**2).sum()) / 6.0
    else:
        ax_pts = [lo[a] + (np.arange(k[a] * ns) + 0.5) * size[a] / ns for a in range(3)]
        G3 = np.array(np.meshgrid(*ax_pts, indexing="ij"))  # (3, n0, n1, n2), local
        P = G3.reshape(3, -1).T
        tilt = A.T @ np.array([1.0, 3.17e-4, 7.31e-4])
        inside3 = _inside_mesh(c0 + P @ A, b.mesh.vertices, b.mesh.triangles,
                               tilt).reshape(G3.shape[1:])
        dv = float(np.prod(size / ns))
        h2 = float(((size / ns) ** 2).sum())
        for r_, (i, j, l) in enumerate(idx):
            sl = (slice(i * ns, (i + 1) * ns), slice(j * ns, (j + 1) * ns),
                  slice(l * ns, (l + 1) * ns))
            ins = inside3[sl]
            n_in = int(ins.sum())
            if not n_in:
                continue
            q = np.stack([G3[a][sl][ins] for a in range(3)], axis=1)
            vol[r_] = n_in * dv
            cen[r_] = q.mean(0)
            tr[r_] = 2.0 * dv * float(((q - cen[r_]) ** 2).sum()) + n_in * dv * h2 / 6.0

    def lin(i, j, l):
        return (i * k[1] + j) * k[2] + l

    # each lattice face between two filled lattice cells and how much of it is solid
    def face_share(a: int, c1, c2) -> float:
        if is_box:
            return 1.0
        sl1 = [slice(c1[t] * ns, (c1[t] + 1) * ns) for t in range(3)]
        sl2 = [slice(c2[t] * ns, (c2[t] + 1) * ns) for t in range(3)]
        sl1[a] = slice((c1[a] + 1) * ns - 1, (c1[a] + 1) * ns)
        sl2[a] = slice(c2[a] * ns, c2[a] * ns + 1)
        both = inside3[tuple(sl1)].squeeze(a) & inside3[tuple(sl2)].squeeze(a)
        return float(both.mean())

    adj = []  # (axis, lin lower, lin upper, solid share of the face)
    for a in range(3):
        for i, j, l in idx:
            c1 = np.array([i, j, l])
            if c1[a] + 1 >= k[a]:
                continue
            c2 = c1.copy()
            c2[a] += 1
            r1, r2 = lin(*c1), lin(*c2)
            if vol[r1] > 0 and vol[r2] > 0:
                adj.append((a, r1, r2, face_share(a, c1, c2)))
    # cells mostly outside the solid join a neighbour (a sliver would be light and stiff)
    grp = np.where(vol > 0, np.arange(ncell), -1)
    full = float(np.prod(size))

    def root(r):
        while grp[r] != r:
            r = grp[r]
        return r

    for r in sorted(np.where(vol > 0)[0], key=lambda r_: (vol[r_], r_)):
        rr = root(r)
        if sum(vol[q] for q in range(ncell) if vol[q] > 0 and root(q) == rr) \
                >= BLOCK_MERGE_SHARE * full:
            continue
        best, best_a = None, 0.0
        for a, r1, r2, sh_ in adj:
            if rr in (root(r1), root(r2)) and root(r1) != root(r2):
                area = sh_ * float(np.prod(np.delete(size, a)))
                if area > best_a:
                    best, best_a = (root(r2) if root(r1) == rr else root(r1)), area
        if best is not None:
            grp[rr] = best
    roots = sorted({root(r) for r in range(ncell) if vol[r] > 0})
    gid = {r: g for g, r in enumerate(roots)}
    group = np.full(ncell, -1)
    for r in range(ncell):
        if vol[r] > 0:
            group[r] = gid[root(r)]
    G = len(roots)
    Vg = np.array([vol[group == g].sum() for g in range(G)])
    Cg = np.array([(vol[group == g, None] * cen[group == g]).sum(0) / Vg[g] for g in range(G)])
    Tg = np.array([float((tr[group == g] + 2.0 * vol[group == g]
                          * ((cen[group == g] - Cg[g]) ** 2).sum(1)).sum()) for g in range(G)])
    rho = mass_t / float(Vg.sum())
    m_g = rho * Vg
    X = c0 + Cg @ A
    X += c0 - (m_g[:, None] * X).sum(0) / mass_t  # the solid's own centre of gravity
    j_g = rho * Tg / 3.0
    J_asm = float(j_g.sum() + 2.0 / 3.0 * (m_g * ((X - c0) ** 2).sum(1)).sum())
    j_g = np.maximum(j_g + (J_t - J_asm) * m_g / mass_t, 0.1 * j_g)
    # the faces between different cells, with their points
    faces = []
    pps = max(1, int(points_per_side))
    for a, r1, r2, sh_ in adj:
        g1, g2 = int(group[r1]), int(group[r2])
        if g1 == g2 or sh_ <= 0:
            continue
        b_, c_ = [t for t in range(3) if t != a]
        cl = lo + idx[r1] * size  # the lower lattice cell's low corner, local
        x_a = cl[a] + size[a]
        rect = Polygon([(0, 0), (size[b_], 0), (size[b_], size[c_]), (0, size[c_])])
        pts2 = _sample_polygon(rect, 2.0 * max(size[b_], size[c_]), max_pts=pps * pps,
                               min_per_side=pps, exact_moment=True)
        q = np.zeros((len(pts2), 3))
        q[:, a] = x_a
        q[:, b_] = cl[b_] + pts2[:, 0]
        q[:, c_] = cl[c_] + pts2[:, 1]
        if not is_box:
            off = np.zeros(3)
            off[a] = 0.25 * size[a] / ns
            keep = _inside_mesh(c0 + (q - off) @ A, b.mesh.vertices, b.mesh.triangles) & \
                _inside_mesh(c0 + (q + off) @ A, b.mesh.vertices, b.mesh.triangles)
            q = q[keep]
            if not len(q):
                continue
        ctr = np.zeros(3)
        ctr[a] = x_a
        ctr[b_] = cl[b_] + 0.5 * size[b_]
        ctr[c_] = cl[c_] + 0.5 * size[c_]
        faces.append((a, g1, g2, c0 + ctr @ A, c0 + q @ A,
                      sh_ * float(size[b_] * size[c_])))
    return BlockLattice(c0, A, lo, size, k, group.reshape(tuple(k)), X, m_g, j_g, faces)


@dataclass
class WoodAcross:
    """What the wood of a block holds across a plane through it (a cut between two of its
    cells, or the wood under one of its faces): moduli (MPa), strengths (MPa), fracture
    energies (N/mm), and whether a crack there runs along the grain or breaks fibres."""

    E: float
    G: float
    f_t: float
    f_v: float
    f_c: float
    G_I: float
    G_II: float
    split: str  # "along" (fibres in the plane: a split) or "across" (fibres broken)


def wood_across(ply, grains, n, area_mm2: float, f_rolling: float, G_rolling: float,
                t90_m: float, G_f: float, G_f90: float, G_v: float) -> WoodAcross:
    """The wood of a compact block across a plane of normal n, for each layer's grain g
    (cos t = |g.n|, averaged over the layers; a block whose grain is not known has its
    fibres in the plane, as the code takes a block elsewhere: it bears across its grain,
    f_c90, E_T): normal stiffness Hankinson's form between E_L (end grain) and E_T, as
    wood_normal_stiffness; tension across the plane Hankinson's between f_t and f_t90 x
    the size factor of the small volume it pulls on (t90_size_factor), as tension_at;
    crushing Hankinson's between f_c and f_c90, as the glue lines'. Shear has no direction
    known (the cohesive law is isotropic in the plane), so it is taken both ways in the
    plane, as shear_compliance does when the force's direction is unknown: with the fibres
    in the plane, half along the grain (G_LT, f_v) and half across it (G_rolling, the
    rolling shear: face_shear_strength at 90 degrees), compliance and strength averaged;
    with the fibres crossing the plane, G_LT and f_v (the shear on an end-grain plane is
    the shear along the grain, by the symmetry of the stress). f_v is the average stress
    on the sheared plane of a compact notched block at failure (ASTM D143 block shear,
    which the Wood Handbook's f_v is), not the 1.5 V / A of a beam's section the solid
    override uses. Fracture energy across the fibres (G_f) where they cross the plane,
    across the grain (G_f90) where they lie in it, in shear G_v."""
    n = np.asarray(n, float)
    cs = [abs(float(np.asarray(g, float) @ n)) for g in grains] if len(grains) else [0.0]
    vf = t90_size_factor(area_mm2, t90_m)
    roll = face_shear_strength(ply.f_v, f_rolling, 0.0)
    inv_g, E, ft, fv, fc, cross = [], [], [], [], [], []
    for c in cs:
        c2 = min(1.0, c * c)
        across = 0.5 * (1.0 - c2)  # the share of the in-plane shear across the fibres
        inv_g.append((1.0 - across) / ply.G_LT + across / max(G_rolling, 1e-9))
        E.append(hankinson(ply.E_L, ply.E_T, c))
        ft.append(hankinson(ply.f_t, ply.f_t90 * vf, c))
        fv.append(c2 * ply.f_v + (1.0 - c2) * 0.5 * (ply.f_v + roll))
        fc.append(hankinson(ply.f_c, ply.f_c90, c))
        cross.append(c2)
    x = float(np.mean(cross))
    return WoodAcross(E=float(np.mean(E)), G=1.0 / float(np.mean(inv_g)),
                      f_t=float(np.mean(ft)), f_v=float(np.mean(fv)), f_c=float(np.mean(fc)),
                      G_I=G_f90 + x * (G_f - G_f90), G_II=G_v,
                      split="across" if x > 0.5 else "along")


def butt_open_compression(n: int, s: float, Ls: float, t: float, tol: float = 1.0) -> float:
    """Compression strength of ``n`` plies whose butt joints stand open (their gaps never
    close) over that of plies without joints: at the weakest section the cut plies carry
    nothing and each ply next to a cut one only y(d) = sqrt(1 - (1 - min(d / 50 t, 1))^2)
    of its share, d the distance to its own nearest joint (Freas & Selbo 1954, USDA Tech.
    Bull. 1069, fig. 61: 0.6 at 10 t, 0.8 at 20 t, 0.92 at 30 t; open butt joints carry
    compression only after considerable deformation, p. 93): a 13-ply brick pattern of
    2 mm plies 0.42, joints 40 mm apart ply to ply 0.89. Ply i's joints at i s, every
    ``Ls``."""
    if n < 2 or Ls <= 0 or t <= 0:
        return 1.0
    pos = np.mod(np.arange(n) * s, Ls)

    def dist(a: float, b: float) -> float:
        d = abs(a - b) % Ls
        return min(d, Ls - d)

    worst = 1.0
    for x0 in np.unique(np.round(pos, 6)):
        cut = {i for i in range(n) if dist(pos[i], x0) <= tol}
        tot = 0.0
        for i in range(n):
            if i in cut:
                continue
            if i - 1 in cut or i + 1 in cut:
                x = min(dist(pos[i], x0) / (50.0 * t), 1.0)
                tot += math.sqrt(max(0.0, 1.0 - (1.0 - x) ** 2))
            else:
                tot += 1.0
        worst = min(worst, tot / n)
    return float(worst)


def butt_stiffness_factor(n: int, L: float, Ls: float, stagger: float | None, E: float,
                          t: float, K: float) -> float:
    """Axial stiffness of a laminated member with butt joints over that of one without:
    near each joint the cut ply hands its force to its neighbours over a shear-lag length
    1 / lambda, lambda^2 = 2 K / (E t) (the ply between two pressed glue lines of shear
    stiffness K), and they stretch the more for it: n joints per stick length Ls add
    2 / lambda x n / (n - 1) of strain-free length, k = 1 / (1 + 2 n / ((n - 1) lambda
    Ls)): ~0.88 for 2 mm plies of 10.5 GPa on lines of 200 N/mm^3. Gluing the butts
    raised such laminates' stiffness 13-39 % (Dansoh, Koizumi & Hirai 2003); closing the
    gaps 9 % (Berard et al. 2011). 1 for a member no longer than a stick or with plies
    without joints (stagger 0)."""
    if n < 2 or L <= 1.02 * Ls or stagger == 0 or K <= 0 or E <= 0 or t <= 0:
        return 1.0
    lam = math.sqrt(2.0 * K / (E * t))
    return 1.0 / (1.0 + 2.0 * n / ((n - 1) * lam * Ls))


def butt_at_of(v) -> list[tuple[float, list[int]]] | None:
    """The ``butt_at`` override as [(fraction along the member, [plies]), ...], or None
    if it cannot be read."""
    try:
        out = []
        for f, plies in v:
            f = float(f)
            ks = [int(k) for k in plies]
            if not 0.0 < f < 1.0 or not ks or min(ks) < 0:
                return None
            out.append((f, ks))
        return out or None
    except (TypeError, ValueError):
        return None


def butt_zones(n: int, L: float, Ls: float, stagger: float, offset: float, lz: float,
               at: list[tuple[float, list[int]]] | None = None
               ) -> list[tuple[float, frozenset[int]]]:
    """Where a laminated member's butt joints are, as zones (centre along the member from
    its start, the plies cut there): ply k's joints at offset + k x stagger, every stick
    length ``Ls`` (or the designer's ``at``: fractions of the length and the plies cut
    there), joints of different plies closer than half a zone ``lz`` merged into one;
    none within half a zone of the member's ends (a joint there is the node's)."""
    pts: list[tuple[float, int]] = []
    if at:
        for f, ks in at:
            pts += [(f * L, k) for k in ks if k < n]
    else:
        for k in range(n):
            x = (offset + k * stagger) % Ls
            while x < L:
                pts.append((x, k))
                x += Ls
    pts = sorted(p for p in pts if lz / 2 < p[0] < L - lz / 2)
    zones: list[list] = []
    for x, k in pts:
        if zones and x - zones[-1][0][0] < lz / 2:
            zones[-1][0].append(x)
            zones[-1][1].add(k)
        else:
            zones.append([[x], {k}])
    return [(float(np.mean(xs)), frozenset(ks)) for xs, ks in zones]


def zone_layers(n: int, cut: frozenset[int], nz: int, T: float, open_frac: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """For a fibre section of ``nz`` layers through a laminate's thickness ``T`` (``n``
    plies stacked from -T/2): the share of each layer left where the plies ``cut`` are
    butted (times ``open_frac``: 1 for a joint that stays open) and how far each layer's
    remaining area's centroid lies from the layer's centre (the net section's shift)."""
    keep = np.ones(nz)
    shift = np.zeros(nz)
    for m in range(nz):
        z = -T / 2 + (m + (np.arange(64) + 0.5) / 64) * T / nz
        ply = np.minimum((z + T / 2) / (T / n), n - 1).astype(int)
        w = 1.0 - open_frac * np.isin(ply, list(cut))
        keep[m] = float(w.mean())
        if w.sum() > 1e-9:
            shift[m] = float((w * z).sum() / w.sum() - z.mean())
    return keep, shift


def _butt_question(g: str, name: str, L: float, Ls: float, s: float, kt: float,
                   what: str = "member") -> dict:
    return {
        "member": f"'{g}' ({L:.0f} mm long, {kt:.0%} of a solid {what}'s tension "
                  f"strength with joints {s:.0f} mm apart)",
        "id": f"butt_joints:{g}", "body": name, "topic": "design", "severity": "check",
        "question": f"'{g}' is {L:.0f} mm long, longer than one stick ({Ls:.0f} mm), so "
                    "its plies have butt joints. How far apart are the joints of "
                    "neighbouring plies? This sets its tension strength: "
                    f"{kt:.0%} of a solid {what} with joints {s:.0f} mm apart. Joints "
                    "that line up across the plies (a brick pattern lines up every other "
                    "ply) or lie close together cut more of it. And how do the ply ends "
                    "meet: cut square and butted tight (sanded), or as cut, or rounded "
                    "stick ends? A gap that does not close before the member is damaged "
                    "(more than ~0.5 mm; rounded stick ends leave ~2 mm) leaves the joint "
                    "out in compression too.",
        "assumed": f"brick pattern: each ply's joints {s:.1f} mm (half a stick) from its "
                   "neighbours', so every other ply is cut at the same section; the gap "
                   "at the butted ends seeded from 0 to scatter.bearing_gap_mm",
        "answer_with": {"bodies": {g + "*": {
            "butt_stagger_mm": "<distance between joints of neighbouring plies, mm; 0 if "
                               "the plies have no joints>",
            "butt_gap_mm": "<gap between butted ply ends, mm; 0 if sanded tight>"}}}}


def plate_butt_factor(name: str, sh, mat: Material, o: dict, questions: list,
                      ext_along: float, ext_across: float, angles: list[float],
                      f_v: float, f_rolling: float, bond: float,
                      butt_eff: float = 1.0) -> float:
    """Fibre-tension factor of a plate laid from sticks: the plies of a direction in
    which the plate is longer than one stick have butt joints, staggered row to row and
    ply to ply by ``butt_stagger_mm`` (a brick pattern by default). The rows and plies of
    that direction form a chain of stick strands; :func:`butt_path_capacity` gives their
    weakest path, the glue between strands shearing at the laminate's interlaminar
    strength (rolling shear through a cross ply). The lower factor of the two directions
    applies to every ply (the shell keeps one fibre strength per element), times
    ``butt_eff`` (as for a laminated member: :func:`_laminated_member`)."""
    Ls, wst = mat.stick.length_mm, mat.stick.width_mm
    if o.get("butt_stagger_mm") == 0 or sh.layup.kind != "laminate":
        return 1.0
    s = float(o.get("butt_stagger_mm") or Ls / 2)
    gf = float(o.get("glue_factor", 1.0))
    fv = interlaminar_strength(angles, f_v, f_rolling, bond * max(gf, 1e-3))
    kt, longest = 1.0, 0.0
    for ref, L, across in ((0.0, ext_along, ext_across), (90.0, ext_across, ext_along)):
        fam = [a for a in angles if abs(math.cos(math.radians(a - ref))) > 0.9]
        if not fam or L <= 1.02 * Ls:
            continue
        rows = max(1, int(round(across / wst)))
        n = min(len(fam) * rows, 40)  # the factor has converged by 40 strands
        P_strand = 1.0  # per strand (cancels): the shear path per mm is fv w / P_strand
        P_ply = mat.f_t_MPa.value * wst * sh.T / len(angles)
        k = butt_path_capacity(n, s, Ls, P_strand, fv * wst / P_ply) / n
        if k * butt_eff < kt:
            kt, longest = k * butt_eff, L
    g = _group(name)
    if kt < 1.0 and o.get("butt_stagger_mm") is None and not any(
            q["id"] == f"butt_joints:{g}" for q in questions):
        questions.append(_butt_question(g, name, longest, Ls, s, kt, what="plate"))
    return kt


def _merge_butt_questions(questions: list[dict]) -> list[dict]:
    """One question for all the laminated members longer than a stick (instead of one
    per member type): the designer answers once, one line per member."""
    bj = [q for q in questions if q["id"].startswith("butt_joints:")]
    if len(bj) < 2:
        for q in bj:
            q.pop("member", None)
        return questions
    rest = [q for q in questions if not q["id"].startswith("butt_joints:")]
    members = "; ".join(q["member"] for q in bj)
    answer: dict = {}
    for q in bj:
        answer.update(q["answer_with"]["bodies"])
    merged = {"id": "butt_joints", "body": bj[0]["body"], "topic": "design",
              "severity": "check",
              "question": "These laminated members are longer than one stick, so their plies "
                          "have butt joints: " + members + ". How far apart are the joints of "
                          "neighbouring plies in each? Joints close together let the glue "
                          "between them shear first and weaken the member in tension.",
              "assumed": bj[0]["assumed"], "answer_with": {"bodies": answer}}
    i = questions.index(bj[0])
    out = rest[:]
    out.insert(min(i, len(out)), merged)
    return out


# the change of the wood's properties per percentage point of moisture content, below
# the fibre saturation point (USDA Wood Handbook 2010, eq. 5-3 with table 5-3a, paper
# birch, intersection moisture content 27 %, table 5-13)
MOISTURE_PER_PCT = {"E_MPa": -0.020, "G_MPa": -0.020, "f_c_MPa": -0.057, "f_v_MPa": -0.024,
                    "wood.f_c90_MPa": -0.050}


def moisture_adjusted(mat: Material, sp: SimParams) -> tuple[Material, SimParams]:
    """The wood of the bridge ``wood.moisture_excess_pct`` percentage points wetter than
    the coupons its values come from (laminated members glued days before the test: a
    PVA line brings ~97 g/m^2 of water, ~+8 points in a 2 mm ply, and it takes a week to
    even out, USDA Wood Handbook ch. 10): E, G, f_c, f_v and f_c90 lowered as the Wood
    Handbook gives (MOISTURE_PER_PCT). Each value keeps its source."""
    dm = float(sp.values.get("wood.moisture_excess_pct", 0.0))
    if dm == 0.0:
        return mat, sp
    d = mat.model_dump()
    for k, rate in MOISTURE_PER_PCT.items():
        if k.startswith("wood."):
            continue
        d[k]["value"] = float(d[k]["value"]) * max(1.0 + rate * dm, 0.05)
    k90 = "wood.f_c90_MPa"
    if k90 in sp.values:
        sp = sp.updated({k90: sp[k90] * max(1.0 + MOISTURE_PER_PCT[k90] * dm, 0.05)},
                        source=sp.sources.get(k90, "assumed"))
    return Material.model_validate(d), sp


LAMINATION_COVER = 0.5  # share of the smaller plate two plates glued face to face cover
# to be taken as a lamination (the middle of the ramp from a hand-made lap to one)
LAMINATION_RAMP = (0.3, 0.7)
BEARING_DEPTH_SHARE = 0.5  # depth an end-grain face's normal stress reaches, over
# sqrt(its area) (side grain: bearing_depth)
DEPTH_ELEMENTS = 2.0  # a member's beam elements at most this x its depth at standard
DEPTH_ELEMENTS_THIN = 1.33  # ... and this x for a member shallower than THIN_MEMBER_MM
THIN_MEMBER_MM = 10.0


def element_depth_cap(depth: float, mesh_scale: float) -> float:
    """The longest beam element a member ``depth`` mm deep gets: DEPTH_ELEMENTS x its
    depth (DEPTH_ELEMENTS_THIN below THIN_MEMBER_MM) at standard and at quick, in
    proportion finer at fine. Quick does not coarsen a member's elements past
    standard's (the realism review of round 24: a light truss of 6 mm members read about
    half at quick what it read at fine); a light member's are as long as fine's were at
    round 24 (6 mm members: 8 mm elements)."""
    k = DEPTH_ELEMENTS_THIN if depth < THIN_MEMBER_MM else DEPTH_ELEMENTS
    return k * depth * min(mesh_scale / 0.75, 1.0)


def build_model(bodies: list[CadBody], mat: Material, sp: SimParams,
                ov: Overrides | None = None, seed: int | None = 0,
                plate_offset_mm: tuple[float, float] = (0.0, 0.0),
                mesh_scale: float = 1.0) -> Model:
    """Structural model of the CAD bodies. ``mesh_scale`` > 1 makes elements larger
    (faster, coarser: ``--fidelity quick``), < 1 smaller."""
    ov = ov or Overrides()
    if ov.params:
        try:
            sp = sp.updated({k: v for k, v in ov.params.items()
                             if not k.startswith("material.")})
        except KeyError as exc:
            raise ValueError(str(exc).strip("'\"")) from None
    mat, sp = moisture_adjusted(mat, sp)
    rng = np.random.default_rng(seed) if seed is not None else None

    def rng_for(key: str):
        """A random stream per named thing (body, joint, plate), so that editing one part
        of a design does not reshuffle the imperfections of all the others."""
        if seed is None:
            return None
        return np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])
    ply = Ply.from_material(mat, sp)
    rho = mat.density_kg_m3.value * 1e-9 * TONNE_PER_KG  # t/mm^3
    glue_frac = mat.glue.mass_fraction.value
    questions: list[dict] = []
    notes: list[str] = []

    # one scale on every seeded imperfection and scatter (1: as set; 0.1: a nearly
    # perfect bridge whose asymmetry is still set by the seed, for mesh studies)
    amp = float(sp.values.get("scatter.amplitude", 1.0))

    def lognormal(cov: float, z: float) -> float:
        """The lognormal factor of mean 1 and CoV ``cov`` (times the amplitude) at the
        standard normal value ``z``."""
        s = math.sqrt(math.log(1 + cov * cov)) * amp
        return float(np.exp(s * z - 0.5 * s * s))

    def scatter(cov: float, key: str = "") -> float:
        r = rng_for(key) if key else rng
        if r is None or cov <= 0:
            return 1.0
        return lognormal(cov, float(r.standard_normal()))

    butt_gap_max = float(sp.values.get("scatter.bearing_gap_mm", 0.0))
    butt_bear = (float(sp.values.get("wood.butt_bearing_efficiency", 0.95)),
                 float(sp.values.get("wood.butt_bear_gap_mm", 0.5)),
                 float(sp.values.get("wood.butt_open_gap_mm", 1.0)))

    def butt_gap(name: str, o: dict) -> float:
        """The gap at a member's butted ply ends: the designer's ``butt_gap_mm``, else
        seeded per member from 0 to scatter.bearing_gap_mm (hand-cut); the perfect
        bridge's ends are tight."""
        if o.get("butt_gap_mm") is not None:
            return float(o["butt_gap_mm"])
        r = rng_for(f"buttgap:{name}")
        return 0.0 if r is None else amp * float(r.uniform(0.0, butt_gap_max))

    def butt_compression(name: str, kt: float, o: dict, n: int = 0,
                         L: float = 0.0) -> float:
        """Compression-strength factor of a laminated member or plate with butt joints:
        a joint whose gap closes before the member is damaged bears, and the member keeps
        wood.butt_bearing_efficiency (0.95) of its strength; a gap that does not close
        leaves the ply's force to go round the joint through the glue lines: a member of
        ``n`` plies ``L`` long keeps its open joints' net section with the plies beside
        them weakened (butt_open_compression, USDA TB 1069), a plate its butt-joint
        tension factor ``kt``. In 2 mm
        veneer laminates gaps of 0.25 and 0.5 mm close early after first yield and 1 mm
        gaps never do (Berard et al. 2011, J Wood Sci 57:107); tight butt joints keep
        0.94-0.96 of a jointless laminate's compression strength (Dansoh et al. 2003, J
        Wood Sci 49:405). Between wood.butt_bear_gap_mm (0.5) and wood.butt_open_gap_mm
        (1.0) the factor falls linearly. The gap: ``butt_gap_mm`` (the designer's: 0 if
        the ends are sanded tight, ~2 for rounded stick ends), else seeded per member from
        0 to scatter.bearing_gap_mm (hand-cut); the perfect bridge's ends are tight."""
        if kt >= 1.0:
            return 1.0
        eff, g0, g1 = butt_bear
        Ls = mat.stick.length_mm
        if n >= 2 and L > 1.02 * Ls and o.get("butt_stagger_mm") != 0:
            kt = butt_open_compression(n, float(o.get("butt_stagger_mm") or Ls / 2), Ls,
                                       mat.stick.thickness_mm)
        if o.get("butt_gap_mm") is not None:
            g = float(o["butt_gap_mm"])
        else:
            r = rng_for(f"buttgap:{name}")
            if r is None:
                return max(eff, kt)
            g = amp * float(r.uniform(0.0, butt_gap_max))
        if g <= g0:
            return max(eff, kt)
        if g >= g1:
            return kt
        return max(kt, eff + (kt - eff) * (g - g0) / (g1 - g0))

    # ---------------------------------------------------------------- body analysis
    shapes: list[Shape] = []
    # compact blocks set rigid in the overrides (kind: rigid): one rigid body each
    rigid_set: set[int] = set()
    # a compact block's cells: about this size (block.cell_mm; 0 = one rigid body, as
    # until round 27)
    cell_mm = float(sp.values.get("block.cell_mm", 0.0))
    # the bridge's centre (of its bounding box): plates' normals face away from it
    _lo = np.min([b.mesh.vertices.min(0) for b in bodies], axis=0) if bodies else np.zeros(3)
    _hi = np.max([b.mesh.vertices.max(0) for b in bodies], axis=0) if bodies else np.zeros(3)
    center = 0.5 * (_lo + _hi)
    for b in bodies:
        o = ov.for_body(b.name)
        lay = None
        if o.get("layup"):
            lay = parse_override_layup(str(o["layup"]))
            if lay:
                lay.source = "override"
        if lay is None:
            lay = parse_layup(b.name)
        if lay is None:
            lay = Layup("laminate", [0.0], "long", "default", "")
            lay.plies_assumed = True
            questions.append({
                "id": f"layup:{b.name}", "body": b.name, "topic": "material",
                "severity": "important",
                "question": f"What is '{b.name}' made of (number of stick plies and their "
                            "direction)? Its name does not say.",
                "assumed": "sticks along its long axis (UD), in plies of a stick's "
                           "thickness glued together",
                "answer_with": {"bodies": {b.name: {
                    "layup": "<N>-ply <UD | 0-90 | +-45>, or block"}}},
            })
        if o.get("grain"):
            gv = grain_vectors_of(o["grain"])
            if gv is not None:
                lay.grain_vecs = gv
            else:
                lay.grain = str(o["grain"])
        g = _group(b.name)
        if lay.orientation_assumed and lay.source == "name" and not o.get("layup") and \
                not any(q["id"] == f"orientation:{g}" for q in questions):
            questions.append({
                "id": f"orientation:{g}", "body": b.name, "topic": "material",
                "severity": "check",
                "question": f"'{g}' (all its copies) says how many plies but not their "
                            "direction. Do all its sticks run along it (UD), or is it "
                            "cross-laminated?",
                "assumed": "all sticks along its long direction (UD)",
                "answer_with": {"bodies": {g + "*": {
                    "layup": f"{len(lay.angles)}-ply <UD | 0-90 | +-45>"}}}})
        forced = o.get("kind")
        if forced in ("block", "rigid", "ignore") and questions and \
                questions[-1].get("id") == f"layup:{b.name}":
            questions.pop()  # set a block or left out: the kind answers it (a block's
            #                  grain, if known, is its grain override)
        if forced == "rigid":  # one rigid body, as blocks were until round 27
            rigid_set.add(len(shapes))
            forced = "block"
        sh = analyse_shape(b, lay, forced, center)
        if len(shapes) in rigid_set:
            sh.why = "rigid body, set in overrides"
        if o.get("axis") is not None and sh.kind == "bar":
            gv = grain_vectors_of(o["axis"])
            if gv and len(gv) == 1:
                ax = np.asarray(gv[0], float)
                if abs(float(ax @ sh.w)) > abs(float(ax @ sh.u)) and \
                        abs(float(ax @ sh.w)) >= abs(float(ax @ sh.n)):
                    # it spans across what its outline makes its length (a floor beam
                    # wider than it is long between its trusses, the adversarial review
                    # of round 23): its axis the other way in the plane of its profile
                    sh.u, sh.w, sh.Lp, sh.Wp = sh.w.copy(), sh.u.copy(), sh.Wp, sh.Lp
                    sh.why += f"; spanning along {o['axis']} (overrides)"
        if lay.plies_assumed and lay.kind == "laminate" and sh.kind in ("bar", "shell"):
            # a part of a stick bridge is built of sticks: plies of a stick's thickness
            # glued together (glue lines that can split), never one solid piece of wood
            t_s = float(getattr(mat.stick, "thickness_mm", 2.0) or 2.0)
            lay.with_plies(round(sh.T / t_s))
            if questions and questions[-1].get("id") == f"layup:{b.name}":
                questions[-1]["assumed"] = (f"{lay.label}: sticks along its long axis, "
                                            f"{len(lay.angles)} plies of a stick's "
                                            f"thickness ({t_s:g} mm) glued together")
        if lay.source == "default" and sh.kind == "block" and questions and \
                questions[-1].get("id") == f"layup:{b.name}":
            questions[-1]["question"] += (
                " It is compact, so it is modelled as a block of the wood's cells (it "
                "deforms and can split; its grain, if you know it, sets how stiff and "
                "strong it is: 'grain'); if it is built from plates or members, set its "
                "kind too." if cell_mm > 0 else
                " It is compact, so it is modelled as a rigid block whatever it is made "
                "of; if it is built from plates or members, set its kind too.")
            questions[-1]["answer_with"]["bodies"][b.name]["kind"] = "<block | shell | bar>"
        if forced == "ignore":
            sh.kind = "ignore"
        if forced in ("shell", "bar") and sh.kind == forced:
            lo, hi = b.mesh.bbox
            dims = sorted(hi - lo)
            bad = (forced == "shell" and dims[0] > 0.34 * dims[1]) or \
                (forced == "bar" and dims[0] < 0.25 * dims[1])
            if bad:
                questions.append({
                    "id": f"kind:{b.name}", "body": b.name, "topic": "geometry",
                    "severity": "check",
                    "question": f"'{b.name}' was set to kind {forced}, but it measures "
                                f"{dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm: a "
                                + ("plate that thick is not a thin plate (the shell model "
                                   "only knows thin plates)" if forced == "shell" else
                                   "member that flat is a plate") + ". Is that right?",
                    "assumed": f"modelled as a {forced} as asked",
                    "answer_with": {"bodies": {b.name: {"kind": "<block | shell | bar>"}}}})
        shapes.append(sh)

    # ---------------------------------------------------------------- contacts
    raw_outlines: list[tuple[Polygon, Polygon]] = []
    near_faces: list = []
    raw_patches = find_patches(bodies, raw_outlines, near_faces)

    def unglued(ia_: int, ib_: int) -> bool:
        """The designer says these two are not glued (a glue override of 0: a load
        hanger resting on a bridge, a part that only bears)."""
        return any(_joint_matches(k_, bodies[ia_].name, bodies[ib_].name) and not v_
                   for k_, v_ in ov.glue.items())

    # near pairs the overrides already leave unglued are not near misses: their extra
    # patches (drawn apart) go, and no question (a load hanger 0.5 mm off a node)
    gone = {x[4] for x in near_faces if x[4] is not None and unglued(x[0], x[1])}
    if gone:
        keep_k = [k_ for k_ in range(len(raw_patches)) if k_ not in gone]
        raw_patches = [raw_patches[k_] for k_ in keep_k]
        raw_outlines[:] = [raw_outlines[k_] for k_ in keep_k]
    near_faces = [x for x in near_faces if not unglued(x[0], x[1])]
    # nor a gap another part fills (a chord between a splice and a diagonal laid over
    # it): a third part touching both on faces parallel to theirs
    touch: dict[int, list] = {}
    for r_ in raw_patches:
        touch.setdefault(r_[0], []).append((r_[1], r_[6]))
        touch.setdefault(r_[1], []).append((r_[0], r_[6]))

    def filled(a_: int, b_: int, n_) -> bool:
        ca = {c for c, nn in touch.get(a_, []) if abs(float(np.dot(nn, n_))) > 0.99}
        cb = {c for c, nn in touch.get(b_, []) if abs(float(np.dot(nn, n_))) > 0.99}
        return bool((ca & cb) - {a_, b_})

    near_faces = [x for x in near_faces if not filled(x[0], x[1], x[5])]
    # faces drawn close but apart, or one inside the other: glued as if they touched up
    # to NEAR_GLUE_MM, asked about one pair at a time (an acknowledgement covers that
    # pair only), the largest first
    by_pair: dict[tuple[int, int], list] = {}
    for ia_, ib_, ar_, off_, _k, _n in near_faces:
        by_pair.setdefault((ia_, ib_), []).append((ar_, off_))
    for (ia_, ib_), lst in sorted(by_pair.items(), key=lambda kv: -sum(a for a, _ in kv[1])):
        ar_ = sum(a for a, _ in lst)
        off_ = max((o for _, o in lst), key=abs)
        if ar_ < 20.0:
            continue
        na_, nb_ = bodies[ia_].name, bodies[ib_].name
        glued = abs(off_) < NEAR_GLUE_MM
        how = (f"{off_:.2f} mm apart" if off_ > 0
               else f"one {-off_:.2f} mm inside the other")
        questions.append({
            "id": f"near_miss:{na_} | {nb_}", "topic": "geometry",
            "severity": "important" if glued or ar_ >= 100.0 else "check",
            "body": na_,
            "question": f"'{na_}' and '{nb_}' face each other over {ar_:.0f} mm^2 but are "
                        f"drawn {how} (parts closer than {CONTACT_TOL_MM} mm are glued). "
                        "Are they glued together in the bridge? (A member drawn at a "
                        "slightly different thickness than the gap it sits in, or plates "
                        "offset for the glue line, leave such gaps.)",
            "assumed": ("glued as if they touched" if glued else
                        f"not glued (more than {NEAR_GLUE_MM} mm apart)"),
            "answer_with": ({"glue": {f"{na_} | {nb_}": 0},
                             "cad": "close the gap (or the overlap) in SolidWorks"}
                            if glued else
                            {"cad": "close the gap in SolidWorks if they are glued"})})
    # plates of narrow arms become frames (where their arms carry what is glued to them)
    for bi, sh in enumerate(shapes):
        forced = ov.for_body(bodies[bi].name).get("kind")
        if sh.kind not in ("shell", "frame") or forced == "shell" or sh.poly is None:
            continue
        mine = [(inter, oa, e1, e2) for ia, ib, inter, oa, e1, e2, _na in raw_patches
                if bi in (ia, ib)]
        fr = _frame_chains(sh, force=sh.kind == "frame", patches=mine)
        if fr is not None:
            arm = 2 * sh.poly.area / max(sh.poly.length, 1e-9)
            sh.kind, sh.frame = "frame", fr
            sh.why = (f"plate of narrow arms ({arm:.0f} mm wide, {sh.T:.1f} mm thick): "
                      f"a frame of members along its {len(fr)} arm centre line(s)"
                      if forced != "frame" else "frame, set in overrides")
        elif sh.kind == "frame":
            sh.kind = "shell"
            sh.why += " (no arms found: kept as a plate)"
    touching = {i: set() for i in range(len(bodies))}
    for ia, ib, *_ in raw_patches:
        touching[ia].add(ib)
        touching[ib].add(ia)

    bld = _Builder()
    # global imperfection: the built bridge leans out of its plane (sway), seeded up to
    # scatter.lean_rad (EN 1995-1-1 5.4.4: 0.005 rad), applied to every node and glue
    # point alike so the parts stay together and unstressed
    rl = rng_for("lean")
    lean = 0.0 if rl is None else \
        amp * sp.values.get("scatter.lean_rad", 0.0) * rl.uniform(-1, 1)
    if lean:
        y_base = min(float(b.mesh.vertices[:, 1].min()) for b in bodies)
        bld.imperf = lambda p: p + np.array([0.0, 0.0, lean * (p[1] - y_base)])
    infos: list[BodyInfo] = []
    body_nodes: dict[int, np.ndarray] = {}
    block_cells: dict[int, BlockLattice] = {}  # compact blocks cut into cells
    block_sf: dict[int, float] = {}  # their seeded strength factors
    bar_axis: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    shell_tree: dict[int, cKDTree] = {}
    shell_mesh: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    # plates glued face to face (the plies of a laminated deck drawn as separate
    # bodies, a gusset on a plate) are meshed node on node over their overlap, finest
    # first: the glue between them then joins nodes straight across, and a load passes
    # through the stack instead of bending each thin layer between nodes it is hung on
    # off-centre (a three-layer deck under the plate made Bridge_v10 2x too soft at
    # quick and 12 % at standard)
    stacked: dict[int, set] = {}
    for ia, ib, _inter, _oa, _e1, _e2, na in raw_patches:
        if shapes[ia].kind == "shell" and shapes[ib].kind == "shell" and \
                abs(float(np.dot(shapes[ia].n, na))) > 0.99 and \
                abs(float(np.dot(shapes[ib].n, na))) > 0.99:
            stacked.setdefault(ia, set()).add(ib)
            stacked.setdefault(ib, set()).add(ia)
    pre_shell: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def _partner_nodes(bi: int) -> np.ndarray | None:
        sh = shapes[bi]
        fixed = []
        for ob in sorted(stacked[bi]):
            if ob in pre_shell:
                so = shapes[ob]
                X3 = so.origin + np.outer(pre_shell[ob][0][:, 0], so.e1) \
                    + np.outer(pre_shell[ob][0][:, 1], so.e2)
                fixed.append(np.c_[(X3 - sh.origin) @ sh.e1, (X3 - sh.origin) @ sh.e2])
        return np.vstack(fixed) if fixed else None

    # forward, finest first: each plate takes the nodes of those meshed before it;
    # back again: each takes its partners' final nodes (a shorter ply's ends included),
    # so both meshes have the same nodes over the whole overlap
    order_st = sorted(stacked, key=lambda i: (_shell_h(shapes[i]), i))
    for bi in order_st + order_st[-2::-1]:
        sh = shapes[bi]
        pre_shell[bi] = _mesh_robust(sh.poly, _shell_h(sh) * mesh_scale, bodies[bi].name,
                                     fixed=_partner_nodes(bi))
    frame_tree: dict[int, cKDTree] = {}
    frame_h: dict[int, float] = {}
    beams: dict[str, list] = {k: [] for k in ("i", "j", "L0", "E0", "fy", "fz", "fA", "E",
                                              "ft", "fc", "Gf", "GJ", "A", "fv", "fv3",
                                              "nply", "Wt", "kGA", "body")}
    shells: dict[str, list] = {k: [] for k in ("nodes", "E0", "P", "t", "el", "body")}
    n_snap = 0
    snap_el: list[bool] = []  # per shell element: larger than its crack band
    snap3: list[np.ndarray] = []  # ... per mode: fibre tension, across the grain, shear
    snap_excess = 1.0  # the lowest strength factor an oversized element got
    sheets_unglued: list[str] = []  # 0/90 sheets taken with their stick edges unglued
    snap_modes: set = set()
    G_f = sp["wood.G_f_N_per_mm"]
    il_strength = {"f_rolling": sp.values.get("wood.f_rolling_MPa", 2.0),
                   "bond": sp.values.get("glue.interlaminar_MPa", 6.0),
                   "butt_eff": sp.values.get("wood.butt_tension_efficiency", 1.0)}
    # a glued crossing of two sticks in torsion: its bond line, or the plies rolling
    # across their grain, ~2.4 x their rolling-shear strength (Flaig & Blass 2013:
    # torsion 3.5-4.7 MPa against rolling shear 1.43-1.51 on CLT crossings)
    crossing_tor = min(il_strength["bond"], 2.4 * il_strength["f_rolling"])

    def _frame_body(bi, b, sh, info, o, ssc, Ef, Sf, mass_t):
        """A plate of narrow arms as a frame: each arm a chain of fibre beams along its
        centre line with the arm's local width and the plate's thickness, its plies
        stacked through the thickness, the arms sharing nodes where they meet."""
        from types import SimpleNamespace

        from bridgesim.sim.skeleton import resample

        T = sh.T * ssc
        o_mid = sh.origin - sh.n * T / 2
        bnd = sh.poly.boundary
        arm = 2 * sh.poly.area / max(sh.poly.length, 1e-9)
        h_e = min(30.0, max(arm, 10.0)) * mesh_scale
        chains = [resample(c, h_e, bnd) for c in sh.frame]
        P2s: list[np.ndarray] = []
        ids_all: list[int] = []
        ends: list[tuple[np.ndarray, int]] = []  # the arms' end nodes (junctions)
        tol = 0.3 * arm  # arm ends this close are one junction (the medial axis
        #                  splits a junction into nearby vertices); never the nodes
        #                  along an arm, which are an element apart

        def end_at(q):
            for p_, i_ in ends:
                if float(np.hypot(*(p_ - q))) <= tol:
                    return i_
            return None

        def nid(q, off=None, end=False):
            if end:
                i_ = end_at(q)
                if i_ is not None:
                    return i_
            p3 = o_mid + q[0] * sh.e1 + q[1] * sh.e2
            i_ = bld.node(p3 if off is None else p3 + off, bi)
            P2s.append(np.asarray(q, float))
            ids_all.append(i_)
            if end:
                ends.append((np.asarray(q, float), i_))
            return i_

        def bow(ci, ch):
            """Initial bow of an arm, as for a member (L/400 on average, 0.3-1.7 of
            that, in a random direction across it), keyed on the arm: the same at every
            mesh. Zero at the arm's ends, which stay at its junctions."""
            rb = rng_for(f"bow:{b.name}:{ci}")
            ratio = sp["scatter.bow_L_over"]
            L_ = float(np.linalg.norm(np.diff(ch.pts, axis=0), axis=1).sum())
            if rb is None or ratio <= 0 or L_ <= 0 or ch.spur:
                return [None] * len(ch.pts)
            a_ = amp * L_ / ratio * rb.uniform(0.3, 1.7)
            ang = rb.uniform(0, 2 * math.pi)
            d2 = ch.pts[-1] - ch.pts[0]
            d2 = d2 / max(float(np.linalg.norm(d2)), 1e-9)
            w3 = -d2[1] * sh.e1 + d2[0] * sh.e2
            direc = math.cos(ang) * w3 + math.sin(ang) * sh.n
            s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ch.pts, axis=0), axis=1))]
            return [a_ * math.sin(math.pi * x / L_) * direc for x in s_]

        # grain: a UD frame's sticks run along each arm (it is built arm by arm); a
        # cross-ply or angle-ply frame is a sheet cut to shape, its plies fixed to the
        # plate's axis (the long side of its bounding rectangle) whatever the arm's
        # direction (an arm at 45 degrees to a +-45 plate is 0/90), unless its grain is
        # "arm" (plies laid along each arm); either follows the span when its grain says so
        gref = None
        if sh.layup.grain == "span":
            gx = np.array([1.0, 0, 0]) - sh.n * sh.n[0]
            if np.linalg.norm(gx) > 0.3:
                gref = gx / np.linalg.norm(gx)
        multi = len({round(a) % 180 for a in sh.layup.angles}) > 1
        if gref is None and multi and sh.layup.grain != "arm":
            gref = sh.u
            g_ = _group(b.name)
            if not any(q["id"] == f"frame_plies:{g_}" for q in questions):
                ax = ", ".join(f"{v:+.2f}" for v in sh.u)
                questions.append({
                    "id": f"frame_plies:{g_}", "body": b.name, "topic": "material",
                    "severity": "check",
                    "question": f"'{g_}' is a {sh.layup.label} plate cut to the shape of "
                                "its arms (a frame). Are its plies one sheet laid at "
                                f"{sh.layup.label.split()[-1]} to its long axis "
                                f"(direction {ax}) in every arm, or laid along each arm?",
                    "assumed": "one sheet: the plies keep the plate's long axis in every "
                               "arm (an arm at 45 degrees to a +-45 plate is 0/90)",
                    "answer_with": {"bodies": {g_ + "*": {
                        "grain": "<long (one sheet, as assumed) | arm (laid along each "
                                 "arm)>"}}}})
        lam_cache: dict[float, object] = {}
        gf_b = max(float(o.get("glue_factor", 1.0)), 1e-3)
        f_45 = 0.5 * min(ply.f_t, ply.f_c)

        def lam_at(rel: float):
            key = round(rel, 0)
            if key not in lam_cache:
                angs = [a + key for a in sh.layup.angles]
                lam_ = laminate(ply, angs, T)
                fv3_ = mat.f_v_MPa.value
                if sh.layup.kind == "laminate" and len(angs) >= 2:
                    fv3_ = interlaminar_strength(angs, mat.f_v_MPa.value,
                                                 il_strength["f_rolling"],
                                                 il_strength["bond"] * gf_b)
                lam_cache[key] = (lam_, fv3_, inplane_split_strength(
                    angs, mat.f_v_MPa.value, f_45))
            return lam_cache[key]

        elems = []  # (a, b, width, L0, chain index)
        order = sorted(range(len(chains)), key=lambda c_: chains[c_].spur)
        for ci in order:
            ch = chains[ci]
            last = len(ch.pts) - 1
            if ch.spur and P2s:  # joined to the frame at its nearest node
                k0 = int(np.argmin([float(np.hypot(*(p_ - ch.pts[0]))) for p_ in P2s]))
                ids = [ids_all[k0]] + [nid(q, end=k == last)
                                       for k, q in enumerate(ch.pts[1:], 1)]
            else:
                j0, j1 = end_at(ch.pts[0]), end_at(ch.pts[-1])
                if j0 is not None and j0 == j1 and ch.length < 2 * tol:
                    continue  # a short link inside one junction
                ids = [nid(q, off, end=k in (0, last))
                       for k, (q, off) in enumerate(zip(ch.pts, bow(ci, ch), strict=True))]
            for k in range(len(ids) - 1):
                a_, b_ = ids[k], ids[k + 1]
                if a_ == b_:
                    continue
                L0 = float(np.linalg.norm(bld.X[b_] - bld.X[a_]))
                if L0 <= 1e-6:
                    continue
                We = max(0.5 * (ch.width[k] + ch.width[k + 1]), 0.3 * arm) * ssc
                elems.append((a_, b_, We, L0, ci))
        if not elems:
            raise ValueError(f"model build failed: no members in the frame '{b.name}'")
        wl = sum(e[2] * e[3] for e in elems)
        ch_len = [float(sum(e[3] for e in elems if e[4] == ci)) for ci in range(len(chains))]
        ch_w = [float(np.median([e[2] for e in elems if e[4] == ci] or [arm]))
                for ci in range(len(chains))]
        kt_fv = {}
        lam0 = laminate(ply, sh.layup.angles, T)
        for ci in range(len(chains)):
            proxy = SimpleNamespace(layup=sh.layup, Wp=ch_w[ci], T=T)
            kt_fv[ci] = _laminated_member(b.name, proxy, lam0, ch_len[ci], mat, o, questions,
                                          **il_strength)
        for a_, b_, We, L0, ci in elems:
            e1v = (bld.X[b_] - bld.X[a_]) / L0
            e2v = np.cross(sh.n, e1v)
            e2v /= np.linalg.norm(e2v)
            e3v = np.cross(e1v, e2v)
            Ein = np.stack([e1v, e2v, e3v], axis=1)
            rel = 0.0
            if gref is not None:
                rel = math.degrees(math.atan2(float(gref @ e2v), float(gref @ e1v)))
            lam, fv_lam, fv_in = lam_at(rel)
            kt = kt_fv[ci][0]
            kc = butt_compression(b.name, kt, o)
            ny, nz = (5, 3) if We >= T else (3, 5)
            gy = ((np.arange(ny) + 0.5) / ny - 0.5) * We * _exact(ny)
            gz = ((np.arange(nz) + 0.5) / nz - 0.5) * T * _exact(nz)
            FY, FZ = np.meshgrid(gy, gz, indexing="ij")
            A_e = We * T
            J, Wt2, Wt3 = rect_torsion(We, T)
            beams["i"].append(a_)
            beams["j"].append(b_)
            beams["L0"].append(L0)
            beams["E0"].append(Ein)
            beams["fy"].append(FY.ravel())
            beams["fz"].append(FZ.ravel())
            beams["fA"].append(np.full(ny * nz, A_e / (ny * nz)))
            beams["E"].append(lam.E_x * Ef)
            beams["ft"].append(lam.f_t * Ef * Sf * kt)
            beams["fc"].append(lam.f_c * Ef * Sf * kc)
            beams["Gf"].append(G_f * Sf)
            beams["GJ"].append(lam.G_xy * Ef * J)
            beams["A"].append(A_e)
            beams["fv"].append(fv_in * Sf)
            beams["fv3"].append(fv_lam * Sf)
            beams["nply"].append(float(max(2, len(lam.angles))))
            beams["Wt"].append((Wt2, Wt3))
            beams["kGA"].append(5 / 6 * lam.G_xy * Ef * A_e)
            beams["body"].append(bi)
            info.beams.append(len(beams["i"]) - 1)
            while len(info.chains) <= ci:
                info.chains.append([])
            info.chains[ci].append(len(beams["i"]) - 1)
            me = mass_t * We * L0 / wl  # the plate's mass spread over its arms
            bld.mass[a_] += me / 2
            bld.mass[b_] += me / 2
            r2 = (We**2 + T**2) / 12
            for nn in (a_, b_):
                bld.rot[nn] += me / 2 * (L0**2 / 12 + r2)
        ids = np.array(sorted(set(ids_all)))
        body_nodes[bi] = ids
        frame_tree[bi] = cKDTree(np.array([bld.X[i] for i in ids]))
        frame_h[bi] = h_e
        info.E_eff_MPa, info.f_c_eff_MPa = lam0.E_x, lam0.f_c
        info.f_t_eff_MPa = lam0.f_t * min(v[0] for v in kt_fv.values())
        info.thickness_mm = T
        info.section_mm = (round(float(np.median([e[2] for e in elems])), 2), round(T, 2))
        info.length_mm = max(ch_len)  # the longest arm (buckling length in the screen)
        info.nodes = list(ids)

    mass_change_t = 0.0
    for bi, (b, sh) in enumerate(zip(bodies, shapes, strict=True)):
        o = ov.for_body(b.name)
        vol = b.mesh.volume
        # what-if: a member's section (both sides) or a plate's thickness scaled
        ssc = o.get("section_scale", 1.0)
        ssc = float(ssc) if isinstance(ssc, (int, float)) and ssc > 0 else 1.0
        if ssc != 1.0:
            if sh.kind in ("bar", "shell", "frame"):
                vfac = ssc * ssc if sh.kind in ("bar", "frame") else ssc
                dm = rho * vol * (vfac - 1) * (1 + glue_frac)
                mass_change_t += dm
                what = "section" if sh.kind in ("bar", "frame") else "thickness"
                notes.append(f"what-if: '{b.name}' {what} x{ssc:g} (mass "
                             f"{dm / TONNE_PER_KG * 1000:+.1f} g); the CAD and its glued "
                             "areas are unchanged")
                vol *= vfac
            else:
                notes.append(f"overrides: section_scale has no effect on '{b.name}' "
                             f"(a {sh.kind}; only members and plates can be scaled)")
        mass_t = rho * vol * (1 + glue_frac)
        info = BodyInfo(b.name, sh.kind, sh.layup.label, sh.layup.source,
                        mass_t / TONNE_PER_KG, vol)
        info.centroid_mm = tuple(float(v) for v in b.mesh.centroid)
        info.bbox_mm = (b.mesh.bbox[0].copy(), b.mesh.bbox[1].copy())
        info.axis = tuple(float(v) for v in (sh.n if sh.kind in ("shell", "frame")
                                             else sh.u))
        info.notes.append(f"{sh.kind}: {sh.why}")
        if sh.kind == "ignore":
            infos.append(info)
            continue
        # stick scatter (stick_factors); strengths below are scaled by Ef x Sf
        Ef, St = stick_factors(seed, b.name, len(sh.layup.angles)
                               if sh.layup.kind == "laminate" else 1, sp)
        Sf = St / Ef * float(o.get("strength_factor", 1.0))
        if sh.kind == "bar":
            T, Wp = sh.T * ssc, sh.Wp * ssc
            if sh.layup.kind == "toothpick":
                Wp = T = math.sqrt(4 * vol / max(sh.Lp, 1e-9) / math.pi)
                lam = laminate(ply, [0.0], T)
            else:
                lam = laminate(ply, sh.layup.angles, T)
            c = b.mesh.centroid
            u, w = sh.u, sh.w
            nrm = np.cross(u, w)
            s_v = (b.mesh.vertices - c) @ u
            s0, s1 = float(s_v.min()), float(s_v.max())
            L = s1 - s0
            kt, fv_lam = _laminated_member(b.name, sh, lam, L, mat, o, questions,
                                           **il_strength)
            kc = butt_compression(b.name, kt, o,
                                  len(lam.angles) if sh.layup.kind == "laminate" else 0, L)
            # butt joints cost stiffness too (shear lag round each joint)
            kE = butt_stiffness_factor(
                len(lam.angles) if sh.layup.kind == "laminate" else 1, L,
                mat.stick.length_mm, o.get("butt_stagger_mm"), ply.E_L,
                mat.stick.thickness_mm,
                sp.values.get("glue.lamination_K_s_N_per_mm3", 200.0))
            fv_in = inplane_split_strength(lam.angles, mat.f_v_MPa.value,
                                           0.5 * min(ply.f_t, ply.f_c))
            # butt joints that stay open (their gap does not close: rounded stick ends)
            # at their places: a short element of the net section at each, its cut plies'
            # fibres removed and the rest shifted to their centroid, over the gap and the
            # shear-lag length 2 / lambda where the cut ply hands its force on (the
            # literature review of round 26: USDA TB 1069 fig. 61, ~13-24 mm for 2 mm
            # sticks), in place of the factors over the whole member: a member can then
            # buckle where its outer plies are butted (published test B; the realism
            # reviews of rounds 21-26). Joints that bear keep the factors (a fibre's
            # strength is its element's, so a ply cannot be cut in tension alone)
            n_ply = len(lam.angles) if sh.layup.kind == "laminate" else 1
            Ls = mat.stick.length_mm
            zones: list[tuple[float, frozenset[int]]] = []
            open_frac = 0.0
            lz = 0.0
            if n_ply >= 2 and L > 1.02 * Ls and o.get("butt_stagger_mm") != 0:
                g_b = butt_gap(b.name, o)
                _eff, g0_b, g1_b = butt_bear
                open_frac = float(np.clip((g_b - g0_b) / max(g1_b - g0_b, 1e-9), 0.0, 1.0))
                if open_frac > 0:
                    t_p = T / n_ply
                    K_l = sp.values.get("glue.lamination_K_s_N_per_mm3", 200.0)
                    lam_s = math.sqrt(2.0 * K_l / (ply.E_L * t_p))
                    lz = 2.0 / lam_s + g_b
                    r_off = rng_for(f"buttpos:{b.name}")
                    off_b = Ls / 2 if r_off is None else float(r_off.uniform(0.0, Ls))
                    zones = butt_zones(n_ply, L, Ls,
                                       float(o.get("butt_stagger_mm") or Ls / 2), off_b,
                                       lz, butt_at_of(o.get("butt_at")))
            if zones:
                kE = 1.0  # the shear lag is in the zones now
            # uniform elements; joints attach to the nearest node through a rigid arm. No
            # longer than element_depth_cap: a light bridge's 6 mm members had 22 mm
            # elements at standard and read 13 % below fine (the realism review of round
            # 20); members 20 mm deep and more change little
            h = min(30.0 * mesh_scale, L / 4 * min(mesh_scale, 1.0)) if L >= 60 \
                else max(L / 2, 1e-3)
            h = min(h, element_depth_cap(min(T, Wp), mesh_scale))
            n_el = max(2, int(math.ceil(L / h - 1e-9)))
            st2 = list(np.linspace(s0, s1, n_el + 1))
            if zones:
                edges = [s0 + xc + d * lz / 2 for xc, _c in zones for d in (-1, 1)]
                keep_ = [x for x in st2[1:-1] if all(abs(x - e) > 0.3 * min(h, lz)
                                                    for e in edges)
                         and not any(abs(x - s0 - xc) < lz / 2 for xc, _c in zones)]
                st2 = sorted({s0, s1, *keep_, *edges})
            st2 = np.array(st2)
            ratio = sp["scatter.bow_L_over"]
            bow_a = 0.0
            ang = 0.0
            rb = rng_for(f"bow:{b.name}")
            if rb is not None and ratio > 0:
                bow_a = amp * L / ratio * rb.uniform(0.3, 1.7)
                ang = rb.uniform(0, 2 * math.pi)
            bow_dir = math.cos(ang) * w + math.sin(ang) * nrm
            ids = []
            for s in st2:
                p = c + s * u + bow_a * math.sin(math.pi * (s - s0) / L) * bow_dir
                ids.append(bld.node(p, bi))
            ids = np.array(ids)
            body_nodes[bi] = ids
            bar_axis[bi] = (c, u, st2)
            A_true = vol / L
            # (a member with butt joints at their places: five layers through its plies,
            # so the plies cut at a joint are resolved; the same 15 fibres)
            ny, nz = (5, 3) if Wp >= T and not zones else (3, 5)
            # section width (along w) and its centre offset at each element: constant for
            # a straight member, cut from the profile for a tapered one
            n_e = len(ids) - 1
            wid = np.full(n_e, A_true / T if sh.fill < 0.75 else Wp)
            off = np.zeros(n_e)
            if sh.fill < 0.75 and sh.poly is not None:
                w2 = np.array([w @ sh.e1, w @ sh.e2])
                for k in range(n_e):
                    sm = 0.5 * (st2[k] + st2[k + 1])
                    p2 = np.array([(c + sm * u - sh.origin) @ sh.e1,
                                   (c + sm * u - sh.origin) @ sh.e2])
                    cut = sh.poly.intersection(LineString([p2 - 2 * Wp * w2,
                                                           p2 + 2 * Wp * w2]))
                    if cut.length > 0:
                        wid[k] = cut.length
                        off[k] = float((np.asarray(cut.centroid.coords[0]) - p2) @ w2)
                dl = np.diff(st2)
                wid = np.maximum(wid, 0.2 * Wp)
                wid *= vol / (T * float((wid * dl).sum()))  # same volume as the solid
            gz = ((np.arange(nz) + 0.5) / nz - 0.5) * T * _exact(nz)
            Wm = float(np.mean(wid))
            E0 = np.stack([u, w, nrm], axis=1)
            # each zone's net section, and the factors its elements' strengths keep: the
            # member's butt-joint factors over the net share of its weakest zone (the
            # glue-line paths round the joints are in them), 1 elsewhere
            z_sec = [zone_layers(n_ply, cut, nz, T, open_frac) for _x, cut in zones]
            net_min = min((float(kp.mean()) for kp, _sh in z_sec), default=1.0)
            kt_z = min(1.0, kt / max(net_min, 1e-9))
            kc_z = min(1.0, kc / max(net_min, 1e-9))
            if zones:
                info.butt_zones = [round(float(xc), 1) for xc, _c in zones]
            for k_e, (a_, b_) in enumerate(zip(ids[:-1], ids[1:], strict=True)):
                We = float(wid[k_e])
                A_e = We * T if sh.fill < 0.75 else A_true
                gy = off[k_e] + ((np.arange(ny) + 0.5) / ny - 0.5) * We * _exact(ny)
                sm_e = 0.5 * (st2[k_e] + st2[k_e + 1]) - s0
                zi = next((i for i, (xc, _c) in enumerate(zones)
                           if abs(sm_e - xc) < lz / 2), None)
                kt_e, kc_e = (kt, kc) if not zones else (1.0, 1.0)
                if zi is None:
                    FY, FZ = np.meshgrid(gy, gz, indexing="ij")
                    fA = np.full(ny * nz, A_e / (ny * nz))
                else:
                    kp, shz = z_sec[zi]
                    FY, FZ = np.meshgrid(gy, gz + shz, indexing="ij")
                    fA = (np.ones((ny, nz)) * kp[None, :]).ravel() * A_e / (ny * nz)
                    kt_e, kc_e = kt_z, kc_z
                J, Wt2, Wt3 = rect_torsion(We, T)
                L0 = float(np.linalg.norm(bld.X[b_] - bld.X[a_]))
                e1v = (bld.X[b_] - bld.X[a_]) / L0
                e3v = np.cross(e1v, w)
                e3v /= np.linalg.norm(e3v)
                e2v = np.cross(e3v, e1v)
                Ein = np.stack([e1v, e2v, e3v], axis=1)
                beams["i"].append(a_)
                beams["j"].append(b_)
                beams["L0"].append(L0)
                beams["E0"].append(Ein)
                beams["fy"].append(FY.ravel())
                beams["fz"].append(FZ.ravel())
                beams["fA"].append(fA)
                beams["E"].append(lam.E_x * Ef * kE)
                beams["ft"].append(lam.f_t * Ef * Sf * kt_e)
                beams["fc"].append(lam.f_c * Ef * Sf * kc_e)
                beams["Gf"].append(G_f * Sf)
                beams["GJ"].append(lam.G_xy * Ef * J)
                beams["A"].append(A_e)
                beams["fv"].append(fv_in * Sf)
                beams["fv3"].append(fv_lam * Sf)
                beams["nply"].append(float(max(2, len(lam.angles))))
                beams["Wt"].append((Wt2, Wt3))
                beams["kGA"].append(5 / 6 * lam.G_xy * Ef * A_e)
                beams["body"].append(bi)
                info.beams.append(len(beams["i"]) - 1)
                # lumped mass
                me = mass_t * L0 * A_e / vol
                bld.mass[a_] += me / 2
                bld.mass[b_] += me / 2
                r2 = (We**2 + T**2) / 12 + off[k_e] ** 2
                for nn in (a_, b_):
                    bld.rot[nn] += me / 2 * (L0**2 / 12 + r2)
            del E0
            info.E_eff_MPa, info.f_t_eff_MPa, info.f_c_eff_MPa = (lam.E_x * kE, lam.f_t * kt,
                                                                  lam.f_c * kc)
            info.section_mm = (round(Wm if sh.fill < 0.75 else Wp, 2), round(T, 2))
            info.length_mm = L
            info.nodes = list(ids)
        elif sh.kind == "shell":
            T = sh.T * ssc
            # a 0/90 sheet of sticks with unglued edges is softer and weaker in in-plane
            # shear (cross-laminated timber, Bogensperger 2010)
            ply_s = ply
            if sp.values.get("glue.plate_edges_glued", 0.0) < 0.5:
                gk, sk = cross_ply_sheet_shear(sh.layup.angles,
                                               mat.stick.thickness_mm / mat.stick.width_mm,
                                               crossing_tor / max(ply.f_v, 1e-9))
                if gk < 1.0:
                    ply_s = dataclasses.replace(ply, G_LT=ply.G_LT * gk, f_v=ply.f_v * sk)
                    sheets_unglued.append(b.name)
            lam_ref = laminate(ply_s, sh.layup.angles, T)
            o_mid = sh.origin - sh.n * T / 2
            area = sh.poly.area
            h = _shell_h(sh) * mesh_scale
            P2, tri, h = pre_shell[bi] if bi in pre_shell else \
                _mesh_robust(sh.poly, h, b.name)  # glue attaches by rigid arms
            ids = np.array([bld.node(o_mid + p[0] * sh.e1 + p[1] * sh.e2, bi) for p in P2])
            body_nodes[bi] = ids
            shell_tree[bi] = cKDTree(np.array([bld.X[i] for i in ids]))
            trib = np.zeros(len(ids))
            for t3 in tri:
                a3 = P2[t3]
                trib[t3] += abs(0.5 * ((a3[1, 0] - a3[0, 0]) * (a3[2, 1] - a3[0, 1])
                                       - (a3[2, 0] - a3[0, 0]) * (a3[1, 1] - a3[0, 1]))) / 3
            shell_mesh[bi] = (ids, trib, h)
            # grain direction per element
            gdir = sh.u
            if sh.layup.grain == "span":
                gx = np.array([1.0, 0, 0]) - sh.n * sh.n[0]
                if np.linalg.norm(gx) > 0.3:
                    gdir = gx / np.linalg.norm(gx)
            edges = []
            if sh.fill < 0.8 and sh.layup.grain == "long":
                ext = np.asarray(sh.poly.exterior.coords)
                segl = np.linalg.norm(np.diff(ext, axis=0), axis=1)
                for k in np.where(segl >= 0.25 * segl.max())[0]:
                    edges.append((ext[k], ext[k + 1]))
            Gm = (sp["wood.G_f_N_per_mm"] * Sf, sp["wood.G_f90_N_per_mm"],
                  sp["wood.G_v_N_per_mm"])
            # butt joints: the plies of a direction longer than one stick
            g2 = np.array([float(gdir @ sh.e1), float(gdir @ sh.e2)])
            g2 /= max(np.linalg.norm(g2), 1e-12)
            pc = np.asarray(sh.poly.exterior.coords)
            kt_pl = plate_butt_factor(
                b.name, sh, mat, o, questions, float(np.ptp(pc @ g2)),
                float(np.ptp(pc @ np.array([-g2[1], g2[0]]))), list(sh.layup.angles),
                mat.f_v_MPa.value, il_strength["f_rolling"], il_strength["bond"],
                il_strength["butt_eff"])
            kc_pl = butt_compression(b.name, kt_pl, o)
            for t3 in tri:
                xy3 = P2[t3]
                X3 = np.array([bld.X[ids[k]] for k in t3])
                ex = X3[1] - X3[0]
                ex /= np.linalg.norm(ex)
                ez = np.cross(X3[1] - X3[0], X3[2] - X3[0])
                ez /= np.linalg.norm(ez)
                ey = np.cross(ez, ex)
                E0 = np.stack([ex, ey, ez], axis=1)
                cc = X3.mean(0)
                Ploc = (X3 - cc) @ E0
                g = gdir
                if edges:
                    c2 = xy3.mean(0)
                    best = min(edges, key=lambda e: _seg_dist(c2, e[0], e[1]))
                    d2 = best[1] - best[0]
                    g = (d2[0] * sh.e1 + d2[1] * sh.e2)
                    g /= np.linalg.norm(g)
                gang = math.degrees(math.atan2(float(g @ ey), float(g @ ex)))
                angs = [gang + a for a in sh.layup.angles]
                if float(ez @ sh.n) < 0:  # plies listed from the -n face up
                    angs = angs[::-1]
                ed = shellmod.element_data(Ploc[:, :2], angs, T, ply_s, Ef, Sf, Gm,
                                           lam_ref.E_x * Ef * T, kt=kt_pl, kc=kc_pl)
                if ed["snapback"]:
                    n_snap += 1
                    snap_excess = min(snap_excess, ed["excess"])
                    snap_modes.update(ed["snap_modes"])
                snap_el.append(bool(ed["snapback"]))
                snap3.append(np.asarray(ed["snap3"], bool))
                shells["nodes"].append(ids[t3])
                shells["E0"].append(E0)
                shells["P"].append(Ploc)
                shells["t"].append(T)
                shells["el"].append(ed)
                shells["body"].append(bi)
                info.shells.append(len(shells["nodes"]) - 1)
                ar = 0.5 * np.linalg.norm(np.cross(X3[1] - X3[0], X3[2] - X3[0]))
                me = mass_t * ar / area
                for k in t3:
                    bld.mass[ids[k]] += me / 3
                    bld.rot[ids[k]] += me / 3 * (h * h / 12 + T * T / 12)
            info.E_eff_MPa = lam_ref.E_x
            info.f_t_eff_MPa, info.f_c_eff_MPa = lam_ref.f_t * kt_pl, lam_ref.f_c * kc_pl
            info.thickness_mm = T
            info.nodes = list(ids)
        elif sh.kind == "frame":
            _frame_body(bi, b, sh, info, o, ssc, Ef, Sf, mass_t)
        elif sh.kind == "cable":
            info.nodes = []  # a floss cable acts between the parts it is tied to
        elif sh.kind == "block" and cell_mm > 0 and bi not in rigid_set:
            # a compact block of the wood's cells (rigid-body-spring model, Kawai 1978):
            # each cell a rigid node with its mass and inertia, the cells joined across
            # the planes between them by the wood's own stiffness and strength (below)
            ev = np.linalg.eigvalsh(b.mesh.inertia * rho * (1 + glue_frac))
            lat = block_lattice(b, sh, cell_mm, mass_t, float(np.mean(ev)),
                                int(sp.values.get("glue.points_per_side", 3)))
            ids = []
            for g in range(lat.n_cells):
                nid = bld.node(lat.X[g], bi)
                bld.mass[nid] += float(lat.mass[g])
                bld.rot[nid] += float(lat.rot[g])
                ids.append(nid)
            body_nodes[bi] = np.array(ids)
            info.nodes = list(ids)
            block_cells[bi] = lat
            block_sf[bi] = Ef * Sf  # the block's seeded strength factor (and the user's)
            kk = "x".join(str(int(v)) for v in lat.k)
            info.notes[0] += (f"; {lat.n_cells} cell(s) ({kk} along its axes, "
                              f"block.cell_mm {cell_mm:g}) joined by the wood's stiffness "
                              "and strength")
            info.cells = {"centre": lat.centre, "axes": lat.axes, "lo": lat.lo,
                          "size": lat.size, "k": lat.k, "group": lat.group,
                          "nodes": np.array(ids)}
        elif sh.kind == "block":
            nid = bld.node(b.mesh.centroid, bi)
            bld.mass[nid] += mass_t
            ev = np.linalg.eigvalsh(b.mesh.inertia * rho * (1 + glue_frac))
            bld.rot[nid] += float(np.mean(ev))
            body_nodes[bi] = np.array([nid])
            info.nodes = [nid]
        infos.append(info)

    def attach(bi: int, p: np.ndarray) -> int:
        sh = shapes[bi]
        if sh.kind == "bar":
            c, u, st2 = bar_axis[bi]
            s = float((p - c) @ u)
            k = int(np.argmin(np.abs(st2 - s)))
            return bld.attach(int(body_nodes[bi][k]), p)
        if sh.kind == "shell":
            _, k = shell_tree[bi].query(p)
            return bld.attach(int(body_nodes[bi][k]), p)
        if sh.kind == "frame":
            _, k = frame_tree[bi].query(p)
            return bld.attach(int(body_nodes[bi][k]), p)
        if bi in block_cells:  # the cell of a block that holds the point
            return bld.attach(int(body_nodes[bi][block_cells[bi].cell_at(p)]), p)
        return bld.attach(int(body_nodes[bi][0]), p)

    # ---------------------------------------------------------------- glue points
    # Where a shell is glued, the glue points are the shell's own nodes inside the glued
    # area, each with its share (tributary area) of the glue, so a node carries glue in
    # proportion to its mass. Elsewhere (bars, blocks) a grid at the element size.
    def elem_size(bi: int) -> float:
        sh = shapes[bi]
        if sh.kind == "shell":
            return shell_mesh[bi][2]  # already scaled
        if sh.kind == "bar":
            return min(30.0, sh.Lp / 4) * mesh_scale
        if sh.kind == "frame":
            return frame_h[bi]
        return 20.0

    patch_pts3: list[tuple[int, int, np.ndarray, np.ndarray, float, np.ndarray,
                           np.ndarray]] = []
    face_of: dict[int, np.ndarray] = {}  # each patch's outline (for fit-up gaps)
    # each patch's overlap and its two faces' outlines in its own basis (how far a gusset
    # reaches past a member lapped on it: panel_factor)
    sides_of: dict[int, tuple] = {}
    mesh_ratio_of: dict[int, float] = {}
    for k_raw, (ia, ib, inter, oa, e1, e2, na) in enumerate(raw_patches):
        if shapes[ia].kind in ("ignore", "cable") or shapes[ib].kind in ("ignore", "cable"):
            continue
        # the shell's own nodes carry the glue where the patch lies on its face, or joins
        # two plates (stacked face to face, meshed node on node over their overlap, or
        # edge to edge): a plate's edge glued to a member or a block gets its own grid
        # (the literature review of round 25: a 40 mm^2 deck edge glued on a light
        # truss's vertical was one row of the deck's mid-surface nodes, a hinge, or a
        # 2x2 grid, depending on the mesh)
        try:
            _u2, _w2, _Lu, _Lw = _mrr(inter if isinstance(inter, Polygon)
                                      else inter.convex_hull)
            p_min = float(_Lw)
        except Exception:  # noqa: BLE001
            p_min = 0.0
        both_plates = shapes[ia].kind == "shell" and shapes[ib].kind == "shell"
        lead = [b_ for b_ in (ia, ib) if shapes[b_].kind == "shell"
                and (abs(float(na @ shapes[b_].n)) > 0.9 or both_plates)]
        pts3 = None
        if lead:
            bl = min(lead, key=lambda b_: shell_mesh[b_][2])
            ids, trib, hl = shell_mesh[bl]
            # where the CAD puts the plate's nodes: the glue points are placed in the
            # CAD (attach adds the lean once), and which nodes carry the glue does not
            # change with the seed's lean (it moved a node row out of a narrow band)
            X3 = np.array([bld.X_cad[i] for i in ids])
            proj = X3 - np.outer((X3 - oa) @ na, na)
            q2 = np.c_[(proj - oa) @ e1, (proj - oa) @ e2]
            inside = shapely.contains_xy(inter, q2[:, 0], q2[:, 1])
            # nodes within half an element of its edge carry half their share, but never
            # more than half the patch's width away: a joint narrower than the plate's
            # elements would spread onto nodes far beyond it (a wide, stiff foot that
            # peels at its far nodes: the 10 mm truss's deck joints read 17 kgf at quick
            # against 46 at standard), so it gets its own grid instead
            band = shapely.contains_xy(inter.buffer(min(0.5 * hl, 0.5 * p_min)),
                                       q2[:, 0], q2[:, 1]) & ~inside
            if ib in stacked.get(ia, ()):
                # plates meshed node on node: the overlap's edge has nodes on both, so
                # the glue is at the nodes on or inside it (none hangs beyond the edge)
                inside = shapely.contains_xy(inter.buffer(1e-3 * hl), q2[:, 0], q2[:, 1])
                band = np.zeros_like(inside)
            w = np.where(inside, trib, np.where(band, 0.5 * trib, 0.0))
            sel = w > 0
            # a face patch needs points_per_side^2 nodes (a grid that resists turning
            # about both axes in its plane), plates node on node three; fewer: own grid
            # (Bridge_v10's 2100 mm^2 diagonal laps had 3 of the gusset's nodes at quick)
            n_min = 3 if both_plates else int(sp.values.get("glue.points_per_side", 3)) ** 2
            # and at least two rows of them across the patch's narrow width: nine nodes in
            # one row are a hinge about it (the realism review of round 26: the count
            # alone let a long, narrow patch on one row of a plate's nodes pass)
            rows_ok = True
            if not both_plates and sel.sum() >= n_min and p_min > 0:
                wq = np.round(q2[sel] @ np.asarray(_w2, float) / max(0.25 * hl, 1e-6))
                rows_ok = len(np.unique(wq)) >= 2
            if sel.sum() >= n_min and rows_ok:
                pts3 = proj[sel]
                wts = w[sel] / w[sel].sum() * inter.area
        if pts3 is None:
            sp_ = max(3.0, min(elem_size(ia), elem_size(ib)))
            # at least two points across each side: a glued patch resists turning
            # about any axis in its plane (its glue is pulled at one edge and pressed
            # at the other), which one point, or one row, would make a hinge
            pts2 = _sample_polygon(inter, sp_, max_pts=400, min_per_side=int(
                sp.values.get("glue.points_per_side", 3)), exact_moment=True)
            pts3 = oa + np.outer(pts2[:, 0], e1) + np.outer(pts2[:, 1], e2)
            wts = np.full(len(pts3), inter.area / len(pts3))
        # where the joint is, from the CAD overlap alone (not from the mesh): the key of
        # its random glue strength, so a seed draws the same bridge at every fidelity
        cc = inter.centroid
        cad_c = oa + cc.x * e1 + cc.y * e2
        if inter.area > 0 and inter.convex_hull.geom_type == "Polygon":
            hull = np.asarray(inter.convex_hull.exterior.coords)
            face_of[len(patch_pts3)] = oa + np.outer(hull[:, 0], e1) + np.outer(hull[:, 1],
                                                                              e2)
        sides_of[len(patch_pts3)] = (inter, e1, e2) + tuple(raw_outlines[k_raw])
        mesh_ratio_of[len(patch_pts3)] = max(
            [shell_mesh[b_][2] / max(p_min, 1e-6) for b_ in (ia, ib)
             if shapes[b_].kind == "shell"] or [0.0])
        patch_pts3.append((ia, ib, pts3, na, inter.area, cad_c, wts))

    # ---------------------------------------------------------------- glue springs
    sp_glue = {k: sp[f"glue.{k}"] for k in ("K_n_N_per_mm3", "K_s_N_per_mm3",
                                            "peel_to_shear", "G_Ic_N_per_mm",
                                            "G_IIc_N_per_mm", "friction")}
    # peak cohesive strength such that the simulated 20 mm lap test gives tau_g. How
    # much the peak exceeds the average depends on how many nodes the glue points hang
    # on: with one node per side the glue shears uniformly (1.0); as soon as a side's
    # stretching is resolved the shear-lag peak appears. Fitted to the lap coupon with
    # its glue on 1, 2, 3, 5 and 9 nodes per side (1.00, 1.04, 1.09, 1.07, 1.065 with
    # the round-21 glue law; tests/test_sim.py checks it), so a joint's strength does not
    # jump with the mesh
    peak_over_avg = sp.values.get("glue.tau_peak_over_avg", 1.065)

    def peak_factor(n_nodes: int) -> float:
        return 1.0 + (peak_over_avg - 1.0) * (1.0 - math.exp(-2.2 * (n_nodes - 1)))

    end_factor = sp.values.get("glue.end_grain_factor", 0.75)
    end_tension = sp.values.get("glue.end_grain_tension_factor", 0.0)
    t90_m = sp.values.get("wood.t90_weibull_m", 0.0)
    f_roll = float(sp.values.get("wood.f_rolling_MPa", 4.0))
    spread_c = float(sp.values.get("wood.bearing_spread_across", 0.33))
    g_roll = float(sp.values.get("wood.G_rolling_MPa", 150.0))
    # each body's glued faces: (patch, outward normal, centre), to tell a member glued on
    # both its faces (between two gussets) from one glued on one
    faces_on: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for k_, (ia_, ib_, p3_, n_, *_r) in enumerate(patch_pts3):
        c_ = np.asarray(p3_, float).reshape(-1, 3)  # its glue points (a long face: many)
        faces_on.setdefault(ia_, []).append((k_, np.asarray(n_, float), c_))
        faces_on.setdefault(ib_, []).append((k_, -np.asarray(n_, float), c_))

    def glued_behind(k: int, bi: int, p: np.ndarray, n: np.ndarray, area: float) -> bool:
        """Whether a part glued on a face of outward normal n at p is glued or bears on
        its opposite face there too (a member between two gussets, a stub on its tenon,
        a stringer under a deck): a glue point of such a face within the face's width."""
        r = math.sqrt(max(area, 1e-9))
        for kk, nn, cc in faces_on.get(bi, ()):
            if kk == k or float(nn @ n) >= -0.9:
                continue
            d = cc - p
            d = d - np.outer(d @ n, n)
            if float(np.min(np.linalg.norm(d, axis=1))) < r:
                return True
        return False

    def transfer_cap(bi: int, nrm: np.ndarray) -> float:
        """The average shear a joint on body bi's ply face can pass into its laminate:
        the laminate's interlaminar strength x n / (n - 1) (inf for a joint on its edge,
        a body of one ply or one without plies)."""
        sh = shapes[bi]
        nply = len(sh.layup.angles)
        if sh.layup.kind != "laminate" or nply < 2 or abs(float(nrm @ sh.n)) < 0.9:
            return math.inf
        il = interlaminar_strength(sh.layup.angles, mat.f_v_MPa.value,
                                   il_strength["f_rolling"], il_strength["bond"])
        return il * nply / (nply - 1)
    gap_max = float(sp.values.get("scatter.bearing_gap_mm", 0.0))
    gap_fill = float(sp.values.get("glue.gap_fill_mm", 0.15))
    n_gapped = 0
    zone = sp.values.get("wood.crush_zone_depth_factor", 3.0)
    frame_axes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def grain_at(bi: int, p: np.ndarray) -> list[np.ndarray]:
        """The grain of each ply of a body at the point p: a frame whose plies follow its
        arms (UD, or grain: arm) has them along the arm nearest to p; anything else as
        the body (grain_vectors)."""
        sh = shapes[bi]
        multi = len({round(a) % 180 for a in sh.layup.angles}) > 1
        if sh.kind != "frame" or sh.layup.grain == "span" or (
                multi and sh.layup.grain != "arm") or not infos[bi].beams:
            return grain_vectors(sh)
        if bi not in frame_axes:
            idx = np.asarray(infos[bi].beams)
            xi = np.array([bld.X[beams["i"][e]] for e in idx])
            xj = np.array([bld.X[beams["j"][e]] for e in idx])
            ax = xj - xi
            frame_axes[bi] = (0.5 * (xi + xj),
                              ax / np.maximum(np.linalg.norm(ax, axis=1), 1e-9)[:, None])
        mids, axes = frame_axes[bi]
        e1 = axes[int(np.argmin(np.linalg.norm(mids - p, axis=1)))]
        w = np.cross(sh.n, e1)
        return [math.cos(math.radians(a)) * e1 + math.sin(math.radians(a)) * w
                for a in sh.layup.angles or [0.0]]

    def bearing_at(bi: int, p: np.ndarray, n: np.ndarray, k: int | None = None,
                   area: float = 0.0) -> tuple[float, float]:
        """grain_bearing with the grain at the point p (grain_at). With the patch ``k``
        (a glued or bearing face of outward normal n): pressed across the grain under a
        face shorter than the part, the fibres beside it help, f_c90 x
        sqrt(l_ef / l) at most 4 (prEN 1995-1-1:2021 8.1.5 via Aloisio et al. 2023, Eur.
        J. Wood Wood Prod. 81:633; van der Put's model, the only one of several to
        explain 685 tests, Leijten et al. 2010): l the face's length along the grain,
        l_ef that spread at 45 degrees on each side over the part's depth where it is
        borne behind the face, 0.4 x its depth where it is not, as far as the part
        reaches (the literature review of round 24: 1.3x at Bridge_v10's knee, 1.8x at
        the diagonal's foot)."""
        gs = grain_at(bi, p)
        if not gs:
            return ply.f_c90, 0.0
        cs = [abs(float(g @ n)) for g in gs]
        f90 = [ply.f_c90] * len(gs)
        if k is not None and k in sides_of:
            inter, e1, e2 = sides_of[k][:3]
            outl = sides_of[k][3] if bi == patch_pts3[k][0] else sides_of[k][4]
            hull = np.asarray(inter.convex_hull.exterior.coords) \
                if inter.convex_hull.geom_type == "Polygon" else None
            ring = np.asarray(outl.convex_hull.exterior.coords) \
                if outl.convex_hull.geom_type == "Polygon" else None
            H = part_depth(bi, n)
            h_ef = H if glued_behind(k, bi, p, n, area) else 0.4 * H
            for i, g in enumerate(gs):
                gi = g - float(g @ n) * n
                if cs[i] > 0.7 or hull is None or ring is None or \
                        float(np.linalg.norm(gi)) < 1e-6:
                    continue
                gi = gi / np.linalg.norm(gi)
                g2 = np.array([float(gi @ e1), float(gi @ e2)])
                lo, hi = float((hull @ g2).min()), float((hull @ g2).max())
                rlo, rhi = float((ring @ g2).min()), float((ring @ g2).max())
                ell = max(hi - lo, 0.5)
                l_ef = ell + min(h_ef, max(lo - rlo, 0.0)) + min(h_ef, max(rhi - hi, 0.0))
                f90[i] = ply.f_c90 * min(math.sqrt(l_ef / ell), 4.0)
        return (float(np.mean([hankinson(ply.f_c, f, c) for f, c in zip(f90, cs,
                                                                         strict=True)])),
                float(np.mean([end_grain_share(c, end_factor) for c in cs])))

    def tension_at(bi: int, p: np.ndarray, n: np.ndarray, solid: bool = False,
                   vf: float = 1.0) -> float:
        """The wood's tension strength across a face of normal n at the point p (along
        its grain on end grain, f_t90 on side grain; Hankinson in between): what a glue
        line's peel pulls on. ``solid``: a face inside one piece of wood, where a part
        with no grain known (a block) has its fibres across it (a tenon's root). ``vf``:
        the size factor on f_t90 of the small volume a joint stresses (t90_size_factor)."""
        gs = grain_at(bi, p)
        if not gs:
            return ply.f_t if solid else ply.f_t90 * vf
        return float(np.mean([hankinson(ply.f_t, ply.f_t90 * vf, abs(float(g @ n)))
                              for g in gs]))

    def face_grain(bi: int, p: np.ndarray, n: np.ndarray) -> list[np.ndarray]:
        """The grain of the ply a glue line of normal n at p lies on: a plate glued on
        its face, the ply on that face (plies run from the -n face to the +n one); a
        member, or a plate glued on its edge, every ply."""
        gs = grain_at(bi, p)
        sh = shapes[bi]
        if len(gs) > 1 and sh.kind in ("shell", "frame") and abs(float(n @ sh.n)) > 0.9:
            side = float((p - sh.centroid) @ sh.n)
            return [gs[-1]] if side > 0 else [gs[0]]
        return gs

    def lap_direction(i1: int, i2: int, n: np.ndarray) -> np.ndarray | None:
        """The direction of the force a lap of normal n passes, in its plane: along the
        member (a bar's axis, else the smaller part's long axis); None for a member glued
        end-on."""
        sh1, sh2 = shapes[i1], shapes[i2]
        if (sh1.kind == "bar") != (sh2.kind == "bar"):
            web = i1 if sh1.kind == "bar" else i2
        else:
            web = i1 if sh1.Lp * sh1.Wp <= sh2.Lp * sh2.Wp else i2
        u = shapes[web].u - float(shapes[web].u @ n) * n
        if float(np.linalg.norm(u)) < 0.5:
            return None
        return u / np.linalg.norm(u)

    def part_depth(bi: int, n: np.ndarray) -> float:
        """How deep a part is across a face of normal n: a member's section (its
        thickness and width as n meets them, not its outline's projection, which reads
        an inclined knee 115 mm deep instead of 23), a plate's thickness, a block's
        extent."""
        sh = shapes[bi]
        if sh.kind == "bar":
            nrm = np.cross(sh.u, sh.w)
            return abs(float(n @ nrm)) * sh.T + abs(float(n @ sh.w)) * sh.Wp \
                + abs(float(n @ sh.u)) * sh.Lp
        if sh.kind in ("shell", "frame") and abs(float(n @ sh.n)) > 0.7:
            return float(sh.T)
        return float(np.ptp(bodies[bi].mesh.vertices @ n))

    def shear_compliance(k: int, bi: int, p: np.ndarray, n: np.ndarray, area: float,
                         u: np.ndarray | None) -> float:
        """The slip per unit shear stress (mm^3/N) a part adds to a glued face of normal
        n (its outward one) through its own thickness: the shear falls from the face to
        nothing across the part, t / (3 G) (Tsai, Oplinger & Morton 1998), half the
        thickness when the part is glued on its opposite face too (a member between two
        gussets), the thickness at most the face's width; G of each ply in the plane of
        the force u and n: G_LT with the grain along the force or along n, the rolling
        modulus across it (wood.G_rolling_MPa), between them in series (as Hankinson).
        Beams and plates keep their sections plane, so this is not in them. It
        reproduces glued crossings of spruce boards (7-9.5 against 8.3 N/mm^3 measured,
        Flaig & Meyer 2014; the literature review of round 24). A block cut into cells:
        block_face."""
        if bi in block_cells:
            return block_face(k, bi, n, area)[1]
        t = min(part_depth(bi, n), math.sqrt(max(area, 1e-9)))
        if glued_behind(k, bi, p, n, area):
            t *= 0.5
        gs = grain_at(bi, p)
        if not gs:
            G = ply.G_LT
        else:
            inv = []
            for g in gs:
                if u is None:  # the force's direction unknown: both ways in the plane
                    inv.append(0.5 / ply.G_LT + 0.5 / g_roll)
                    continue
                gn = abs(float(g @ n))
                gu = abs(float(g @ u))
                across = max(0.0, 1.0 - gn * gn - gu * gu)  # grain across the force
                inv.append((1.0 - across) / ply.G_LT + across / g_roll)
            G = 1.0 / float(np.mean(inv))
        return t / (3.0 * max(G, 1e-9))

    def lap_caps(k: int, i1: int, i2: int, p: np.ndarray, n: np.ndarray
                 ) -> tuple[float, float, float]:
        """What the wood on either side of a lap lets through in shear (MPa): the lower of
        the plies the glue lies on, each sheared at its grain's angle to the force
        (face_shear_strength), with each side's panel factor (panel_factor: a gusset
        under a member wider than 8 x its thickness passes less), and the panel factors
        of the two sides (for their laminates' transfer caps). The force runs along the
        member: a bar's axis, else the smaller part's long axis. A side meeting the glue
        end-on is left to the end-grain factor; a member glued end-on takes no lap cap."""
        u = lap_direction(i1, i2, n)
        if u is None:
            return math.inf, 1.0, 1.0
        v = np.cross(n, u)
        inter, e1, e2, out1, out2 = sides_of[k]
        v2 = np.array([float(v @ e1), float(v @ e2)])
        hull = np.asarray(inter.convex_hull.exterior.coords) \
            if inter.convex_hull.geom_type == "Polygon" else np.zeros((1, 2))
        c2 = np.asarray(inter.centroid.coords[0])
        pv = (hull - c2) @ v2
        b_w = float(np.ptp(pv))
        caps, pfs = [], []
        for bi, outl in ((i1, out1), (i2, out2)):
            # how far this side's face reaches past the lap across the force, each way
            sides = 0
            h_f = float(np.ptp(bodies[bi].mesh.vertices @ n))
            try:
                ln = LineString([c2 - 1e4 * v2, c2 + 1e4 * v2]).intersection(outl)
                segs = list(getattr(ln, "geoms", [ln]))
                ts = [((np.asarray(g.coords) - c2) @ v2) for g in segs
                      if g.geom_type == "LineString" and len(g.coords) >= 2]
                own = [t for t in ts if t.min() <= pv.max() and t.max() >= pv.min()]
                if own:
                    lo_, hi_ = min(t.min() for t in own), max(t.max() for t in own)
                    reach = 2.0 * h_f  # half the 4 h_f a side spreads over
                    sides = int(pv.min() - lo_ > reach) + int(hi_ - pv.max() > reach)
            except Exception:  # noqa: BLE001
                sides = 0
            pf = panel_factor(h_f, b_w, sides)
            pfs.append(pf)
            fs = []
            for g in face_grain(bi, p, n):
                gp = g - float(g @ n) * n
                lg = float(np.linalg.norm(gp))
                if lg >= 0.5:
                    fs.append(face_shear_strength(mat.f_v_MPa.value, f_roll,
                                                  abs(float(gp @ u)) / lg))
            if fs:
                caps.append(float(np.mean(fs)) * pf)
        return (min(caps) if caps else math.inf), pfs[0], pfs[1]

    def cross_share(bi: int, p: np.ndarray, n: np.ndarray) -> float:
        """Share of a part's fibres that cross a face inside one piece of wood (1 for a
        block with no grain known, a tenon's root)."""
        gs = grain_at(bi, p)
        if not gs:
            return 1.0
        return float(np.mean([abs(float(g @ n)) ** 2 for g in gs]))

    def wood_normal_stiffness(k: int, bi: int, p: np.ndarray, n: np.ndarray,
                              area: float) -> float:
        """How stiff the wood under a face of outward normal n is when pressed or pulled
        normal to it, per mm^2 (N/mm^3): its modulus at its grain's angle to the normal
        (E_L on end grain, E_T on side grain, Hankinson's form between) over the depth
        the stress reaches into it. Pressed across the grain the stress spreads at 45
        degrees along the grain and at wood.bearing_spread_across across it (none when
        the face spans the part's width), over the part's depth across the face, or half
        of it when the part is not borne on its opposite face there (it carries the load
        away along itself): h_eq = l b / (2 (b - c l)) ln(b (l + 2 h) / (l (b + 2 c h)))
        for a face l along the grain and b across (Blass & Gorlacher 2004, WCTE: all
        their glulam tests on one E_90 with it; the literature review of round 24, an
        orthotropic FE model: -9/+7 % on full-width faces, -6/+16 % on narrower ones,
        where round 24's depth, min(extent, 0.5 sqrt(area)), was 0.64-2.25x off). End
        grain: the part's depth, at most half the face's width. A block whose grain is
        not given bears across it (as its crushing strength does). A block cut into
        cells: block_face."""
        if bi in block_cells:
            return block_face(k, bi, n, area)[0]
        gs = face_grain(bi, p, n) or grain_at(bi, p)
        H = part_depth(bi, n)
        if not glued_behind(k, bi, p, n, area):
            H *= 0.5
        H = max(H, 0.5)
        inter, e1, e2 = sides_of[k][:3]
        outl = sides_of[k][3] if bi == patch_pts3[k][0] else sides_of[k][4]
        hull = np.asarray(inter.convex_hull.exterior.coords) \
            if inter.convex_hull.geom_type == "Polygon" else None
        ring = np.asarray(outl.convex_hull.exterior.coords) \
            if outl.convex_hull.geom_type == "Polygon" else None
        ks = []
        for g in gs or [None]:
            if g is None:
                gn, gi = 0.0, None
            else:
                gn = abs(float(g @ n))
                gi = g - float(g @ n) * n
            E = ply.E_T if g is None else hankinson(ply.E_L, ply.E_T, gn)
            if g is None or gn > 0.7 or hull is None or float(np.linalg.norm(gi)) < 1e-6:
                if gn > 0.7:  # end grain: the stress stays in the fibres under the face
                    h = min(part_depth(bi, n), BEARING_DEPTH_SHARE * math.sqrt(area))
                else:  # no grain known: as across, spreading both ways
                    sq = math.sqrt(max(area, 1e-9))
                    h = bearing_depth(sq, sq, H, spread_c, spread_c)
                ks.append(E / max(h, 0.5))
                continue
            gi = gi / np.linalg.norm(gi)
            g2 = np.array([float(gi @ e1), float(gi @ e2)])
            a2 = np.array([-g2[1], g2[0]])
            ell = max(float(np.ptp(hull @ g2)), 0.5)
            bw = max(float(np.ptp(hull @ a2)), 0.5)
            full = ring is not None and bw >= 0.9 * float(np.ptp(ring @ a2))
            h = bearing_depth(ell, bw, H, 1.0, 0.0 if full else spread_c)
            ks.append(E / max(h, 0.5))
        return float(np.mean(ks))

    G_f_w, G_f90_w, G_v_w = (sp["wood.G_f_N_per_mm"], sp["wood.G_f90_N_per_mm"],
                             sp["wood.G_v_N_per_mm"])
    t90_m_w = sp.values.get("wood.t90_weibull_m", 0.0)

    def block_face(k: int, bi: int, n: np.ndarray, area: float) -> tuple[float, float]:
        """The wood of a block cut into cells under one of its glued or bearing faces (of
        outward normal n, patch k): (normal stiffness per mm^2, N/mm^3; shear compliance,
        mm^3/N). Only the wood between the face and the centroid of the cell it hangs on,
        d deep (half a cell for a box): the cells beyond are joined by their own springs,
        so the block's depth is not counted twice (round 27 took the block's whole extent
        under each face, glued behind or not: a block pressed between two faces had twice
        its compliance). Normal: E at the grain's angle (wood_across) over the depth the
        stress reaches within d, spreading as in wood_normal_stiffness only where the
        block's face reaches past the patch (a face covering the block's whole face does
        not spread); end grain: d, or half the face's width when the face is smaller.
        Shear: the same slab sheared evenly, min(d, the face's width) / G, G both ways in
        the plane (wood_across), as the cells' own springs, so that a block sheared between
        two faces has the compliance of its depth whatever its cells."""
        lat = block_cells[bi]
        pts = np.asarray(patch_pts3[k][2], float).reshape(-1, 3)
        d = float(np.mean([abs(float((lat.X[lat.cell_at(q)] - q) @ n)) for q in pts]))
        H = max(d, 0.5)
        wa = wood_across(ply, grain_vectors(shapes[bi]), n, area, f_roll, g_roll, t90_m_w,
                         G_f_w, G_f90_w, G_v_w)
        inter, e1, e2 = sides_of[k][:3]
        outl = sides_of[k][3] if bi == patch_pts3[k][0] else sides_of[k][4]
        hull = np.asarray(inter.convex_hull.exterior.coords) \
            if inter.convex_hull.geom_type == "Polygon" else None
        ring = np.asarray(outl.convex_hull.exterior.coords) \
            if outl.convex_hull.geom_type == "Polygon" else None
        gs = grain_vectors(shapes[bi]) or [None]
        ks = []
        for g in gs:
            gn = 0.0 if g is None else abs(float(g @ n))
            E = hankinson(ply.E_L, ply.E_T, gn)
            gi = None if g is None else g - float(g @ n) * n
            if gi is not None and float(np.linalg.norm(gi)) > 1e-6:
                gi = gi / np.linalg.norm(gi)
                a1 = np.array([float(gi @ e1), float(gi @ e2)])
                c1, c2 = 1.0, spread_c  # 45 degrees along the grain, less across it
            else:  # end grain or no grain known: across both ways
                a1 = np.array([1.0, 0.0])
                c1 = c2 = spread_c
            a1 = a1 / max(float(np.linalg.norm(a1)), 1e-12)
            a2 = np.array([-a1[1], a1[0]])
            if hull is None or ring is None:
                ell = bw = math.sqrt(max(area, 1e-9))
                full1 = full2 = False
            else:
                ell = max(float(np.ptp(hull @ a1)), 0.5)
                bw = max(float(np.ptp(hull @ a2)), 0.5)
                full1 = ell >= 0.9 * float(np.ptp(ring @ a1))
                full2 = bw >= 0.9 * float(np.ptp(ring @ a2))
            if gn > 0.7:  # end grain: the stress stays in the fibres under the face
                h = H if (full1 and full2) else min(H, BEARING_DEPTH_SHARE * math.sqrt(area))
            else:
                h = bearing_depth(ell, bw, H, 0.0 if full1 else c1, 0.0 if full2 else c2)
            ks.append(E / max(h, 0.5))
        t = min(H, math.sqrt(max(area, 1e-9)))
        return float(np.mean(ks)), t / max(wa.G, 1e-9)

    def crush_depth(i: int, n: np.ndarray) -> float:
        ext = float(np.ptp(bodies[i].mesh.vertices @ n))
        sh = shapes[i]
        dims = [x for x in (sh.T, sh.Wp) if x and x > 0]
        return max(0.5, min(ext, zone * min(dims)) if dims else ext)

    springs: dict[str, list] = {k: [] for k in ("a", "b", "area", "n0", "Kn", "Ks", "sig",
                                                "tau", "GIc", "GIIc", "mu", "patch",
                                                "breakable", "d0", "fc", "cdep")}
    end_grain_joints: list[tuple[str, str, float]] = []
    patches: list[Patch] = []
    # the workmanship of the whole build: one factor on every joint of the bridge
    gf_bridge = scatter(sp.values.get("scatter.glue_bridge_cov", 0.0), "glue:bridge")
    # several patches between the same two bodies: each by its rank, the largest first
    # (to 1 mm^2), then along the bridge (its centre's x, then y, then z, to 1 mm); not by
    # its centre rounded to 10 mm, whose halves round to even and moved a draw when a part
    # moved 5 mm or was read from STL instead of SLDPRT (the adversarial review of round
    # 23); the largest keeps the key of a joint alone between its bodies, so a sliver of
    # contact gained or lost beside it does not redraw it (round 24)
    by_pair: dict[tuple[int, int], list[tuple[tuple, int]]] = {}
    for k_, (ia_, ib_, _p, _n, a_, cad_c_, _w) in enumerate(patch_pts3):
        by_pair.setdefault((min(ia_, ib_), max(ia_, ib_)), []).append(
            ((-int(round(float(a_))),) + tuple(
                np.floor(np.asarray(cad_c_, float) + 0.5).astype(int).tolist()), k_))
    pair_rank: dict[int, int] = {}
    for lst in by_pair.values():
        for r_, (_c, k_) in enumerate(sorted(lst)):
            pair_rank[k_] = r_
    for k_patch, (ia, ib, pts3, na, area, _cad_c, wts) in enumerate(patch_pts3):
        key = "|".join(sorted((bodies[ia].name, bodies[ib].name)))
        # the stream of its random glue strength: a joint alone between its two bodies by
        # their names (an edit that moves it keeps its draw, so ab compares the design,
        # not new luck at the joint changed); several between the same two bodies, the
        # largest so, the others also by their rank (pair_rank)
        where = "" if pair_rank[k_patch] == 0 else f":{pair_rank[k_patch]}"
        gf = gf_bridge * scatter(sp["scatter.glue_cov"], f"glue:{key}{where}")
        g_sc = gf  # the drawn workmanship; the overrides below are the user's own
        gf *= float(ov.for_body(bodies[ia].name).get("glue_factor", 1.0))
        gf *= float(ov.for_body(bodies[ib].name).get("glue_factor", 1.0))
        for key, val in ov.glue.items():
            if _joint_matches(key, bodies[ia].name, bodies[ib].name):
                gf *= val
                ov.used.add(key)
        # the user's factor (a part glued badly, or a what-if's bigger glued area)
        g_user = gf / g_sc if g_sc > 0 else gf
        # two bodies drawn apart but one piece of wood (a tenon and its beam, a seat
        # laminated with its chord): the face between them breaks at the wood's own
        # strength, not a glue line's
        solid = False
        for key in ov.solid:
            if _joint_matches(key, bodies[ia].name, bodies[ib].name):
                solid = True
                ov.used.add(key)
        # how much of the face is really glued (and how well): a joint glued over less
        # of its face (starved edges, gaps) is softer in shear as well as weaker, in
        # proportion; bearing in compression keeps the whole contact
        k_cov = min(max(gf, 0.2), 3.0)
        pid = len(patches)
        cen = pts3.mean(0)
        poly3 = None
        pc = patches.append(Patch(ia, ib, float(area), cen, na, polygon=poly3))
        del pc
        # the wood on either side crushes, across or along its grain (Hankinson), when
        # the joint is pressed together; glue on end grain holds poorly
        fa, ea = bearing_at(ia, cen, na, k_patch, float(area))
        fb, eb = bearing_at(ib, cen, -na, k_patch, float(area))
        f_bear = min(fa, fb)
        # the depth of wood that can crush: the weaker side's extent along the normal,
        # at most a few times its thickness (the zone a local load compacts)
        c_dep = min(crush_depth(i, na) for i, f in ((ia, fa), (ib, fb))
                    if f <= 1.05 * f_bear)
        e_end = max(ea, eb)  # share of the plies on a side meeting the glue end-on
        # peel pulls the wood on either side apart across the face: a glue line holds in
        # peel no more than its weaker side's tension across the face
        vf = t90_size_factor(float(area), t90_m)
        f_peel = min(tension_at(ia, cen, na, vf=vf), tension_at(ib, cen, na, vf=vf))
        if solid:  # one piece: no glue line, no fit-up, the wood's strength across it
            gf, e_end, k_cov = 1.0, 0.0, 1.0
            f_peel = min(tension_at(ia, cen, na, solid=True),
                         tension_at(ib, cen, na, solid=True))
        gf_side = gf  # before the end-grain share: what the line's tension is scaled by
        gf *= 1.0 - e_end * (1.0 - end_factor)
        # a glue line on end grain pulled apart: the fibres run along the pull, so it
        # holds a share of the wood's tension strength along the grain
        # (glue.end_grain_tension_factor x f_t: end-grain butt joints reach ~25 % of it
        # with special techniques, USDA Wood Handbook ch. 10; a hand-made PVA vertical's
        # end-grain foot held 2.6 MPa before it broke in a published truss test, Moraes
        # et al. 2019), not the side-grain peel strength x the end-grain factor (0.7 MPa
        # until round 21, a tenth of that: validation README 3.14)
        # (capped by the other side pulled apart across its face: a vertical's end glued
        # on a chord's side lifts the chord's surface at its tension across the grain)
        sig_end_glue = end_tension * max(tension_at(ia, cen, na), tension_at(ib, cen, na))
        if e_end > 0.5 and not solid:
            end_grain_joints.append((bodies[ia].name, bodies[ib].name, float(area)))
        # fit-up: a member's end cut to bear on another part is not square. It touches
        # at one edge and stands off by up to scatter.bearing_gap_mm at the other
        # (seeded per face, in a random direction across it); the glue filling the gap
        # is weaker the wider it is (gap_strength), and where it is too wide to bridge
        # the face is dry and bears only once the gap has closed
        gaps = np.zeros(len(pts3))
        if e_end > 0.5 and gap_max > 0 and len(pts3) > 1 and not solid:
            names2 = "|".join(sorted((bodies[ia].name, bodies[ib].name)))
            rg = rng_for(f"gap:{names2}{where}")
            if rg is not None:
                g = amp * float(rg.uniform(0.0, gap_max))
                t1 = np.cross(na, [0.0, 0.0, 1.0] if abs(na[2]) < 0.9 else [1.0, 0.0, 0.0])
                t1 /= np.linalg.norm(t1)
                phi = float(rg.uniform(0.0, 2 * math.pi))
                u = math.cos(phi) * t1 + math.sin(phi) * np.cross(na, t1)
                tt = (pts3 - cen) @ u
                # 0 at the face's edge that touches, g at the opposite edge (its outline,
                # not its outermost glue points, which stand inside it)
                tf = (face_of.get(k_patch, pts3) - cen) @ u
                if np.ptp(tf) > 1e-9:
                    gaps = g * np.clip((tt - tf.min()) / np.ptp(tf), 0.0, 1.0)
                    n_gapped += 1
        # a PVA line that fills a gap is weaker the thicker it is (gap_strength): full
        # strength up to gap_fill, none (dry) from about a millimetre
        fill = np.array([gap_strength(gi, gap_fill) for gi in gaps])
        dry = fill <= 0.0
        att_a = [attach(ia, p) for p in pts3]
        att_b = [attach(ib, p + gi * na) if d_ else attach(ib, p)
                 for p, gi, d_ in zip(pts3, gaps, dry, strict=True)]
        n_nodes = max(len({bld.att_node[a] for a in att_a}),
                      len({bld.att_node[b] for b in att_b}))
        tau_p = mat.glue.tau_g_MPa.value * peak_factor(n_nodes) * lap_size_factor(
            float(area), sp.values.get("glue.size_ref_mm2", 200.0),
            sp.values.get("glue.weibull_m", 10.0), sp.values.get("glue.size_cap_mm2", 3000.0))
        # shear stiffness: the glue line and its surfaces (glue.K_s_N_per_mm3) in series
        # with the two parts' shear through their thickness (shear_compliance; until
        # round 24 one constant for every joint, 60, from nothing: the literature review
        # of round 24 put a stick coupon at ~400-600 with a pressed line, a 13-ply member
        # between two gussets at ~50-70)
        u_lap = lap_direction(ia, ib, na)
        c_wood = (shear_compliance(k_patch, ia, cen, na, float(area), u_lap)
                  + shear_compliance(k_patch, ib, cen, -na, float(area), u_lap))
        ks_line = 1.0 / (1.0 / sp_glue["K_s_N_per_mm3"] + c_wood)
        tau_peel = tau_p  # a glue line's peel does not shear the wood across its fibres
        # a lap holds the lower of its glue line and the wood beside it: the ply the glue
        # lies on sheared at its grain's angle to the force (across its fibres: rolling
        # shear), a gusset under a wide member passing less (panel_factor). A bonded joint
        # holds what its weakest part holds: glued birch-plywood laps failed in the face
        # veneer at 30-90 degrees and followed its strength at its angle (Wang et al.
        # 2024), gusset coupons in the face veneer's rolling shear (Debertolis et al.
        # 2025); a weak glue line fails with little regard to the angle (Rheme et al.
        # 2016; Frolovs et al. 2017). Round 23 took 1 - w (1 - f(angle)) x the glue line,
        # w the share failing in the wood, drawn 0-1 (the literature review of round 23)
        tau_face, pf_a, pf_b = (math.inf, 1.0, 1.0) if solid else lap_caps(
            k_patch, ia, ib, cen, na)
        tau_face *= peak_factor(n_nodes)
        patches[pid].strength_MPa = min(tau_p, tau_face) / peak_factor(n_nodes)
        # two plates glued face to face (a deck's plies drawn as separate bodies): a
        # lamination, pressed flat over its whole face, as the glue lines inside one
        # laminated body are, so it is as strong and stiff as they are (the same bond
        # line drawn as one body or two gives the same bridge), with no size effect
        # (a lamination covers most of the smaller plate: a short overlap of two plates,
        # a deck's tongue slid into its groove on build day, is a hand-made lap; in
        # between, from LAMINATION_RAMP[0] to [1] of the smaller plate covered, the joint
        # goes over from one to the other: a gusset covering 45 % or 55 % of itself is
        # glued much the same way, and its strength does not jump 2.5x at half, the
        # adversarial review of round 21)
        # by geometry, whatever the plates are meshed as (a strap narrow enough to be a
        # frame of members is glued as flat as a wide one: the adversarial review of round
        # 22 found its joint 4.8x weaker as a frame)
        parallel = (not solid and shapes[ia].kind in ("shell", "frame")
                    and shapes[ib].kind in ("shell", "frame")
                    and abs(float(na @ shapes[ia].n)) > 0.99
                    and abs(float(na @ shapes[ib].n)) > 0.99)  # as ``stacked``
        cover = float(area) / max(min(shapes[ia].poly.area, shapes[ib].poly.area), 1e-9) \
            if parallel else 0.0
        lo, hi = LAMINATION_RAMP
        w_lam = min(max((cover - lo) / (hi - lo), 0.0), 1.0)
        if w_lam > 0:
            # and is a plate, not a stick: a single stick glued flat along a gusset is a
            # hand-made lap like tau_g's coupon (two sticks face to face); from one stick
            # wide to three the joint goes over to a lamination
            small = shapes[ia] if shapes[ia].poly.area <= shapes[ib].poly.area \
                else shapes[ib]
            w_eff = 2.0 * small.poly.area / max(small.poly.length, 1e-9)
            ws = float(mat.stick.width_mm)
            w_lam *= min(max((w_eff - ws) / (2.0 * ws), 0.0), 1.0)
        if w_lam > 0:
            # as the same line inside one laminated body: the bond line's strength, or what
            # the two plies it lies on take at the angle they cross at (rolling shear)
            ga, gb = face_grain(ia, cen, na), face_grain(ib, cen, na)
            rel = [math.degrees(math.acos(min(1.0, abs(float(g1 @ g2)))))
                   for g1 in ga for g2 in gb] or [0.0]
            # (a plate wider than 8 x the other's thickness glued inside it passes less, as
            # a member lapped on it does: the panel factor, 1 for plates glued over each
            # other's whole width, a deck's plies)
            tau_lam = interlaminar_strength(
                [0.0] + rel, mat.f_v_MPa.value, sp.values.get("wood.f_rolling_MPa", 4.0),
                sp.values.get("glue.interlaminar_MPa", 4.0)) * peak_factor(n_nodes) * min(
                    pf_a, pf_b)
            # (the lap's own share holds no more than its face's wood, as above; the
            # lamination's is its plies' crossing already)
            tau_p = (1.0 - w_lam) * min(tau_p, tau_face) + w_lam * tau_lam
            tau_face = math.inf
            tau_peel = (1.0 - w_lam) * tau_peel + w_lam * tau_lam
            ks_line = (1.0 - w_lam) * ks_line + w_lam / (1.0 / sp.values.get(
                "glue.lamination_K_s_N_per_mm3", sp_glue["K_s_N_per_mm3"]) + c_wood)
            patches[pid].strength_MPa = tau_p / peak_factor(n_nodes)
            patches[pid].lamination = w_lam >= 0.5
        if not solid:
            # what the laminates on either side pass on to their inner plies: the ply the
            # glue is on hands (n - 1) / n of the joint's load across its first glue
            # line, which holds the laminate's interlaminar strength (rolling shear
            # through a cross ply): a joint is no stronger than that x n / (n - 1). The
            # layered shells do not delaminate by themselves (the adversarial review of
            # round 20: a gusset lap drawn at 5.1 MPa on a 0/90 gusset of 2.5 MPa rolling
            # shear)
            # (a gusset under a wide member passes less into its inner plies too: the
            # critical rolling-shear plane is the first cross ply, APA PDS Supplement 3
            # 3.6.1, with the same panel factor)
            cap = min(transfer_cap(ia, na) * pf_a, transfer_cap(ib, na) * pf_b)
            if cap < patches[pid].strength_MPa:
                patches[pid].strength_MPa = cap
            tau_cap = cap * peak_factor(n_nodes)
        if solid:
            # the wood's shear on the section, averaged as a code check does (EN 1995
            # 6.1.7 and 6.5.2: 1.5 V / A <= f_v at a notched end), and its fracture
            # energies: in shear along the grain, across the fibres where they cross it
            tau_p = mat.f_v_MPa.value / 1.5 * peak_factor(n_nodes)
            patches[pid].strength_MPa = mat.f_v_MPa.value / 1.5
            cross = min(cross_share(ia, cen, na), cross_share(ib, cen, na))
            g1 = sp["wood.G_f90_N_per_mm"] + cross * (sp["wood.G_f_N_per_mm"]
                                                      - sp["wood.G_f90_N_per_mm"])
            g2 = sp["wood.G_v_N_per_mm"]
            patches[pid].kind = "solid"
        patches[pid].peak_factor = peak_factor(n_nodes)
        patches[pid].mesh_ratio = float(mesh_ratio_of.get(k_patch, 0.0))
        # normal stiffness: the glue line and the surfaces it bonds (glue.K_n_N_per_mm3)
        # in series with the wood on either side, pressed across or along its grain (until
        # round 23 the constant alone: a side-grain face as stiff as an end-grain one, the
        # bridge near-linear to its peak, the realism review of round 23)
        kn_line = 1.0 / (1.0 / sp_glue["K_n_N_per_mm3"]
                         + 1.0 / wood_normal_stiffness(k_patch, ia, cen, na, float(area))
                         + 1.0 / wood_normal_stiffness(k_patch, ib, cen, -na,
                                                       float(area)))
        for wa, a_, b_, d_, fl in zip(wts, att_a, att_b, dry, fill, strict=True):
            # glue_factor 0: not glued; past the glue's reach across a fit-up gap: dry
            springs["d0"].append(1.0 if gf <= 0 or d_ else 0.0)
            springs["a"].append(a_)
            springs["b"].append(b_)
            springs["area"].append(float(wa))
            springs["n0"].append(na)
            springs["Kn"].append(kn_line)
            springs["Ks"].append(ks_line * k_cov)
            gq = max(gf * fl, 1e-3)  # glue_factor 0: handled as "already broken" (d0 = 1)
            # the glue's drawn scatter (workmanship, how much of the face is glued)
            # scales the glue line; the wood's caps (laminate transfer, tension across
            # the face) only shrink with it (a starved face), never grow past the wood;
            # the user's glue_factor scales both (a what-if's bigger glued area passes
            # more into the wood, a badly glued part less)
            wq = max(min(g_sc * fl, 1.0) * g_user, 1e-3)
            if solid:
                springs["tau"].append(tau_p * gq)
                springs["sig"].append(f_peel)
                springs["GIc"].append(g1)
                springs["GIIc"].append(g2)
            else:
                tau_s = min(tau_p * gq, tau_cap * wq, tau_face * wq)
                springs["tau"].append(tau_s)
                side = min(min(tau_peel * gq, tau_cap * wq) * sp_glue["peel_to_shear"],
                           f_peel * wq)
                gqs = max(gf_side * fl, 1e-3)
                end = min(sig_end_glue * gqs, f_peel * wq)
                springs["sig"].append(side + e_end * max(end - side, 0.0))
                springs["GIc"].append(sp_glue["G_Ic_N_per_mm"] * gq)
                springs["GIIc"].append(sp_glue["G_IIc_N_per_mm"] * gq)
            springs["mu"].append(sp_glue["friction"])
            springs["patch"].append(pid)
            springs["breakable"].append(True)
            springs["fc"].append(f_bear)
            springs["cdep"].append(c_dep)
            patches[pid].springs.append(len(springs["a"]) - 1)
    if end_grain_joints:
        names = sorted({f"{a} | {b}" for a, b, _ in end_grain_joints})
        groups = sorted({" | ".join(sorted((_group(a), _group(b))))
                         for a, b, _ in end_grain_joints})
        questions.append({
            "id": "end_grain_joints", "body": end_grain_joints[0][0], "topic": "connection",
            "severity": "check",
            "question": f"{len(names)} glued joint(s) are mostly end grain (a member's "
                        "end glued flat to another part): " + "; ".join(groups[:12])
                        + (" ..." if len(groups) > 12 else "") + ". Glue on end grain "
                        "holds about half as well in shear as on side grain; pulled apart "
                        "it holds a share of the wood's tension along the grain, no more "
                        "than the other part takes across its face. Are these butt joints "
                        "as drawn, or are they reinforced (gussets, laps, sized end grain)?",
            "assumed": f"glue strength in shear x {end_factor:g} on the end-grain share "
                       "of these joints (glue.end_grain_factor; a 0/90 plate's edge, half "
                       f"end grain, gets the mean), in tension {end_tension:g} x the "
                       "wood's tension along the grain (glue.end_grain_tension_factor) "
                       "capped by the other part's across its face; each cut end out "
                       f"of square, standing off its seat by up to {gap_max:g} mm at one "
                       "edge (seeded per face, scatter.bearing_gap_mm), dry where that "
                       f"is wider than {gap_fill:g} mm (glue.gap_fill_mm) and bearing "
                       "there only once the gap has closed",
            "answer_with": {"params": {"glue.end_grain_factor":
                                       "<0.5 for plain end-grain joints, 1 if sized or "
                                       "measured as strong as side grain>",
                                       "scatter.bearing_gap_mm":
                                       "<the widest gap under a cut end held on its seat "
                                       "(feeler gauge or paper of known thickness), 0 if "
                                       "each end is sanded to fit>"},
                            "glue": {names[0]: "<factor for one joint>"}}})

    if sheets_unglued:
        groups = sorted({_group(n) for n in sheets_unglued})
        gk_, sk_ = cross_ply_sheet_shear([0, 90], mat.stick.thickness_mm / mat.stick.width_mm,
                                         crossing_tor / max(mat.f_v_MPa.value, 1e-9))
        questions.append({
            "id": "plate_edges", "body": sheets_unglued[0], "topic": "material",
            "severity": "check",
            "question": "Are the sticks of the 0/90 plates glued edge to edge, or only face "
                        "to face? " + "; ".join(groups[:12]) + (" ..." if len(groups) > 12
                                                              else "")
                        + ". With the edges unglued a sheet carries in-plane shear through "
                          "the glued crossings of its plies, like cross-laminated timber: "
                          f"x{gk_:.2f} of its shear stiffness and x{sk_:.2f} of its shear "
                          "strength for 2 plies (the glued crossings twist).",
            "assumed": "edges not glued (glue spread on the faces only)",
            "answer_with": {"params": {"glue.plate_edges_glued":
                                       "<1 if every stick is glued to its neighbours, 0 if "
                                       "not>"}}})

    # bodies touching nothing: tie them to the nearest body (and ask: a part a CAD slip
    # left a little off its neighbour is usually meant to be glued there)
    allV = [b.mesh.vertices for b in bodies]
    surf: dict[int, np.ndarray] = {}

    def surface_points(i: int) -> np.ndarray:
        """Points over body i's faces (5 mm apart at most), so the distance between two
        bodies is measured face to face, not corner to corner."""
        if i not in surf:
            V, T = bodies[i].mesh.vertices, bodies[i].mesh.triangles
            pts = [V]
            for t3 in V[T]:
                n = int(min(40, math.ceil(max(np.linalg.norm(t3[1] - t3[0]),
                                              np.linalg.norm(t3[2] - t3[0]),
                                              np.linalg.norm(t3[2] - t3[1])) / 5.0)))
                if n > 1:
                    ij = [(a, b_) for a in range(n + 1) for b_ in range(n + 1 - a)]
                    w = np.array([(a / n, b_ / n) for a, b_ in ij])
                    pts.append(t3[0] + w[:, :1] * (t3[1] - t3[0]) + w[:, 1:] * (t3[2] - t3[0]))
            surf[i] = np.vstack(pts)
        return surf[i]

    for bi, b in enumerate(bodies):
        if shapes[bi].kind in ("ignore", "cable") or touching[bi]:
            continue
        best, bd, bp = None, math.inf, None
        tree_b = cKDTree(surface_points(bi))
        for bj in range(len(bodies)):
            if bj == bi or shapes[bj].kind in ("ignore", "cable"):
                continue
            sj = surface_points(bj)
            d, k = tree_b.query(sj)
            j = int(np.argmin(d))
            if d[j] < bd:
                bd, best, bp = float(d[j]), bj, sj[j]
        if best is None or bd > 10.0:
            questions.append({
                "id": f"floating:{b.name}", "body": b.name, "topic": "connection",
                "severity": "important",
                "question": f"'{b.name}' does not touch any other part. How is it held? (A "
                            "part drawn a little off where it is glued is a CAD slip: move "
                            "it onto its neighbour.)",
                "assumed": "it is free (it will fall off)",
                "answer_with": {"bodies": {b.name: {"kind": "ignore"}},
                                "cad": f"move '{b.name}' onto the part it is glued to"}})
            continue
        pid = len(patches)
        patches.append(Patch(bi, best, 4.0, bp, np.array([0.0, 1, 0]), kind="tie"))
        # three tie points spread over the loose part (one would let it swing)
        Vb = allV[bi]
        far = Vb[np.argsort(np.linalg.norm(Vb - bp, axis=1))[::-1]]
        ties = [bp, far[0]]
        d2 = np.linalg.norm(np.cross(far - bp, far[0] - bp), axis=1)
        ties.append(far[int(np.argmax(d2))])
        for tp in ties:
            springs["a"].append(attach(bi, tp))
            springs["b"].append(attach(best, tp))
            springs["area"].append(4.0)
            springs["n0"].append(np.array([0.0, 1.0, 0.0]))
            springs["Kn"].append(sp_glue["K_n_N_per_mm3"])
            springs["Ks"].append(sp_glue["K_s_N_per_mm3"])
            # dabs of glue across the gap: they hold and break as glue does (until round
            # 18 an unbreakable tie, an infinitely strong joint from a CAD slip)
            springs["tau"].append(mat.glue.tau_g_MPa.value)
            springs["sig"].append(mat.glue.tau_g_MPa.value * sp_glue["peel_to_shear"])
            springs["GIc"].append(sp_glue["G_Ic_N_per_mm"])
            springs["GIIc"].append(sp_glue["G_IIc_N_per_mm"])
            springs["mu"].append(0.5)
            springs["patch"].append(pid)
            springs["breakable"].append(True)
            springs["d0"].append(0.0)
            springs["fc"].append(ply.f_c90)
            springs["cdep"].append(1e30)
            patches[pid].springs.append(len(springs["a"]) - 1)
        questions.append({
            "id": f"loose:{b.name}", "body": b.name, "topic": "connection",
            "severity": "important",
            "question": f"'{b.name}' has no glued face; its nearest part is "
                        f"'{bodies[best].name}' ({bd:.1f} mm away, face to face). Is it "
                        "glued there (a CAD slip: move it onto that part) or tied or "
                        "wedged (e.g. a floss toggle)?",
            "assumed": f"held to '{bodies[best].name}' by three dabs of glue at its nearest "
                       "points (they break as glue does)",
            "answer_with": {"bodies": {b.name: {"kind": "ignore"}},
                            "cad": f"move '{b.name}' onto '{bodies[best].name}' if it is "
                                   "glued there"}})

    # ---------------------------------------------------------------- inside the blocks
    # each plane between two cells of a block (a lattice face): cohesive points of the
    # wood itself, as a glue line's grid (glue.points_per_side a side, so the plane resists
    # turning), with the stiffness of the wood between the two cells' centroids, K_n =
    # E / h, K_s = G / h (h their distance along the normal: the rigid-body-spring model,
    # Kawai 1978), its strength across the plane (wood_across) times the block's seeded
    # strength factor, the wood's fracture energies; no glue scatter, fit-up gap, glue
    # rate law or creep (crush.py takes the creep out); it crushes as a glue line's wood
    # does, the cells' wood compacting over h
    n_block_faces = 0
    for bi, lat in block_cells.items():
        grains = grain_vectors(shapes[bi])
        ids = body_nodes[bi]
        for ax, g1, g2, ctr, pts, area in lat.faces:
            nrm = lat.axes[ax].copy()
            h = max(abs(float((lat.X[g2] - lat.X[g1]) @ nrm)), 0.25 * float(lat.size[ax]))
            wa = wood_across(ply, grains, nrm, float(area), f_roll, g_roll, t90_m_w, G_f_w,
                             G_f90_w, G_v_w)
            pid = len(patches)
            patches.append(Patch(bi, bi, float(area), np.asarray(ctr, float), nrm,
                                 kind="block", strength_MPa=wa.f_v * block_sf[bi],
                                 split=wa.split))
            n_block_faces += 1
            for p in pts:
                springs["a"].append(bld.attach(int(ids[g1]), p))
                springs["b"].append(bld.attach(int(ids[g2]), p))
                springs["area"].append(float(area) / len(pts))
                springs["n0"].append(nrm)
                springs["Kn"].append(wa.E / h)
                springs["Ks"].append(wa.G / h)
                springs["sig"].append(wa.f_t * block_sf[bi])
                springs["tau"].append(wa.f_v * block_sf[bi])
                springs["GIc"].append(wa.G_I)
                springs["GIIc"].append(wa.G_II)
                springs["mu"].append(sp_glue["friction"])  # wood on wood once split
                springs["patch"].append(pid)
                springs["breakable"].append(True)
                springs["d0"].append(0.0)
                springs["fc"].append(wa.f_c)
                springs["cdep"].append(h)
                patches[pid].springs.append(len(springs["a"]) - 1)

    # ---------------------------------------------------------------- floss cables
    cables: dict[str, list] = {k: [] for k in ("a", "b", "L0", "EA", "Fb", "body")}
    for bi, (b, sh) in enumerate(zip(bodies, shapes, strict=True)):
        if sh.kind != "cable":
            continue
        c = b.mesh.centroid
        u = sh.u
        ends = [c - u * sh.Lp / 2, c + u * sh.Lp / 2]
        att = []
        for e in ends:
            best, bd = None, math.inf
            for bj in range(len(bodies)):
                if bj == bi or shapes[bj].kind in ("ignore", "cable"):
                    continue
                d = float(np.min(np.linalg.norm(allV[bj] - e, axis=1)))
                lo_, hi_ = bodies[bj].mesh.bbox
                if (e >= lo_ - 1).all() and (e <= hi_ + 1).all():
                    d = min(d, 0.0 + 0.5 * d)
                if d < bd:
                    bd, best = d, bj
            if best is None or bd > 15.0:
                att = []
                questions.append({
                    "id": f"floss_end:{b.name}", "body": b.name, "topic": "connection",
                    "question": f"One end of floss '{b.name}' is not near any part. Where "
                                "is it tied?",
                    "assumed": "the floss is left out", "answer_with": {
                        "bodies": {b.name: {"kind": "ignore"}}}})
                break
            att.append(attach(best, e))
        if len(att) != 2:
            continue
        n_str = sp["floss.strands"]
        cables["a"].append(att[0])
        cables["b"].append(att[1])
        cables["L0"].append(float(np.linalg.norm(ends[1] - ends[0])))
        cables["EA"].append(n_str * sp["floss.strand_EA_N"])
        cables["Fb"].append(n_str * sp["floss.strand_break_N"] * scatter(0.1, f"floss:{b.name}"))
        cables["body"].append(bi)
        for a_ in att:  # the floss's own (tiny) mass goes to the parts it is tied to
            bld.mass[bld.att_node[a_]] += 0.5 * rho * b.mesh.volume
        infos[bi].cables.append(len(cables["a"]) - 1)
        infos[bi].notes.append(f"floss: {n_str:g} strands, breaks at "
                               f"{n_str * sp['floss.strand_break_N']:.0f} N")
    if sp.sources.get("floss.strands") == "assumed" and cables["a"]:
        questions.append({
            "id": "floss:strands", "topic": "material",
            "question": "How many strands of floss does each modelled floss tie have?",
            "assumed": f"{sp['floss.strands']:g} strands of "
                       f"{sp['floss.strand_break_N']:g} N each",
            "answer_with": {"params": {"floss.strands": "<number of strands>"}}})

    # ---------------------------------------------------------------- table & plate
    Vall = np.vstack(allV)
    table_y = float(Vall[:, 1].min())
    active = [i for i in range(len(bodies)) if shapes[i].kind not in ("ignore",)]
    UP = np.array([0.0, 1.0, 0.0])

    def bearing(bi: int, p: np.ndarray) -> float:
        """Crushing strength (MPa) of a body pressed vertically at p, by the angle of
        each ply's grain there to the vertical (Hankinson, averaged over the plies, as at
        the glue lines): f_c for a standing UD bar or a frame's standing UD leg, f_c90 for
        anything lying flat, in between for a standing +-45 plate."""
        sh = shapes[bi]
        if sh.kind == "bar" and sh.layup.kind == "toothpick":
            return ply.f_c90  # a round dowel bears on its side
        return bearing_at(bi, p, UP)[0]

    V_AREA = 16.0  # mm^2 a mesh corner stands for
    p_att, p_sup, p_fmax = [], [], []
    for bi in active:
        if shapes[bi].kind == "cable":
            continue
        V = bodies[bi].mesh.vertices
        extra, areas = [], [np.full(len(V), V_AREA)]
        for f in bodies[bi].faces:
            if f.kind == "plane" and f.normal[1] < -0.9 and \
                    abs(f.loops[0][0][1] - table_y) < 1.0:
                poly, o, e1, e2 = face_polygon(f)
                pts = _sample_polygon(poly, 8.0, 60)
                extra.append(o + np.outer(pts[:, 0], e1) + np.outer(pts[:, 1], e2))
                areas.append(np.full(len(pts), poly.area / max(len(pts), 1)))
        Vb = np.vstack([V] + extra) if extra else V
        for p, a in zip(Vb, np.concatenate(areas), strict=True):
            p_att.append(attach(bi, p))
            p_sup.append(p[1] - table_y < 0.5)
            p_fmax.append(bearing(bi, p) * a)
    sup = np.array([bld.X[bld.att_node[a]] + bld.att_r0[a] for a, s in zip(p_att, p_sup,
                                                                        strict=True) if s])
    if not len(sup):
        raise ValueError("no part of the bridge touches the table")
    xc0 = 0.5 * (sup[:, 0].min() + sup[:, 0].max())
    left = sup[sup[:, 0] < xc0]
    right = sup[sup[:, 0] >= xc0]
    if not len(left) or not len(right):
        raise ValueError("the bridge does not rest on supports at both ends")
    # a built bridge never stands on four corners at once: each support corner (left or
    # right, near or far) is lifted by up to scatter.support_level_mm, one of them not at
    # all, so the bridge rocks on three until the load beds it down
    p_lift = np.zeros(len(p_att))
    rl = rng_for("support level")
    if rl is not None and sp["scatter.support_level_mm"] > 0:
        Pp = np.array([bld.X[bld.att_node[a]] + bld.att_r0[a] for a in p_att])
        zc0 = 0.5 * (sup[:, 2].min() + sup[:, 2].max())
        lv = amp * rl.uniform(0, sp["scatter.support_level_mm"], 4)
        lv -= lv.min()
        corner = (Pp[:, 0] >= xc0).astype(int) * 2 + (Pp[:, 2] >= zc0).astype(int)
        p_lift = np.where(np.array(p_sup, bool), lv[corner], 0.0)
    # the plate is placed by hand: off-centre by up to scatter.placement_mm each way
    off = np.asarray(plate_offset_mm, float)
    rp = rng_for("plate")
    if rp is not None and sp["scatter.placement_mm"] > 0 and not off.any():
        off = amp * rp.uniform(-1, 1, 2) * sp["scatter.placement_mm"]
    mid_x = 0.5 * (left[:, 0].mean() + right[:, 0].mean()) + off[0]
    zmin, zmax = Vall[:, 2].min(), Vall[:, 2].max()
    deck_z = 0.5 * (zmin + zmax) + off[1]
    PL, PW = 200.0, 90.0  # rulebook §12.5 plate
    PT = 30.0  # two 15 mm plates (Appendix B)
    fp = Polygon([(mid_x - PL / 2, deck_z - PW / 2), (mid_x + PL / 2, deck_z - PW / 2),
                  (mid_x + PL / 2, deck_z + PW / 2), (mid_x - PL / 2, deck_z + PW / 2)])
    up_faces = []
    for bi in active:
        if shapes[bi].kind == "cable":
            continue
        for f in bodies[bi].faces:
            if f.kind != "plane" or f.normal[1] < 0.95:
                continue
            P3 = np.vstack(f.loops)
            y = float(P3[:, 1].mean())
            pxz = Polygon(f.loops[max(range(len(f.loops)), key=lambda i: len(f.loops[i]))]
                          [:, [0, 2]]).buffer(0)
            cover = pxz.intersection(fp).area
            if cover > 1.0:
                up_faces.append((y, cover, bi, f))
    if not up_faces:
        raise ValueError("no deck surface under the loading plate at mid-span")
    big = [u for u in up_faces if u[1] >= 0.15 * fp.area]
    # the deck: the highest surface covering a good part of the footprint and low enough
    # for the plate to slide in under everything above it (the clear opening, §8.9)
    deck_y = max(u[0] for u in big) if big else max(u[0] for u in up_faces)
    plate_bottom = max([u[0] for u in up_faces if deck_y - 0.5 <= u[0] <= deck_y + 40]
                       + [deck_y])
    deck_bodies = sorted({bodies[u[2]].name for u in up_faces if abs(u[0] - deck_y) < 0.5})
    q_att = []
    # down to 40 mm below the deck: if the deck under the plate breaks, the plate bears on
    # what is underneath (ribs, stringers, a saddle)
    region_lo = np.array([mid_x - PL / 2 - 20, plate_bottom - 40, deck_z - PW / 2 - 20])
    region_hi = np.array([mid_x + PL / 2 + 20, plate_bottom + PT + 150, deck_z + PW / 2 + 20])
    q_fmax = []
    q_body: list[int] = []
    for bi in active:
        if shapes[bi].kind == "cable":
            continue
        V = bodies[bi].mesh.vertices
        sel = ((V >= region_lo) & (V <= region_hi)).all(1)
        pts = [V[sel]]
        areas = [np.full(int(sel.sum()), V_AREA)]
        for f in bodies[bi].faces:
            if f.kind != "plane":
                continue
            P3 = np.vstack(f.loops)
            if ((P3.max(0) < region_lo) | (P3.min(0) > region_hi)).any():
                continue
            poly, o, e1, e2 = face_polygon(f)
            g = _sample_polygon(poly, 8.0, 400)
            g3 = o + np.outer(g[:, 0], e1) + np.outer(g[:, 1], e2)
            keep = ((g3 >= region_lo) & (g3 <= region_hi)).all(1)
            pts.append(g3[keep])
            areas.append(np.full(int(keep.sum()), poly.area / max(len(g), 1)))
        for p, a in zip(np.vstack(pts), np.concatenate(areas), strict=True):
            q_att.append(attach(bi, p))
            q_fmax.append(bearing(bi, p) * a)
            q_body.append(bi)

    plate_mass = sp["crusher.plate_mass_kg"] * TONNE_PER_KG
    pc = np.array([mid_x, plate_bottom + PT / 2 + 0.02, deck_z])
    plate_node = bld.node(pc, -1)
    bld.mass[plate_node] = plate_mass
    bld.rot[plate_node] = plate_mass * (PL**2 + PW**2) / 12
    # the plate has to fit where it goes. A part it would start inside (trusses closer
    # together than the plate is wide, a strut lower than its top) is a design problem
    # for which the rules ban the crush test (§8.9, §8.2.3.1); simulated, such a part is
    # taken as clear of the plate (as if the plate were trimmed to fit) instead of being
    # thrown off by a contact that starts deep inside it
    half = np.array([PL / 2, PT / 2, PW / 2])
    q_in, misfit_mm, misfit_bodies = [], 0.0, set()
    for i, a in enumerate(q_att):
        gap = half - np.abs(np.asarray(bld.X[bld.att_node[a]]) + bld.att_r0[a] - pc)
        if (gap > 0.05).all():
            q_in.append(i)
            misfit_mm = max(misfit_mm, float(gap.min()))
            misfit_bodies.add(bodies[q_body[i]].name)
    if q_in:
        out_ = set(q_in)
        q_att = [a for i, a in enumerate(q_att) if i not in out_]
        q_fmax = [f for i, f in enumerate(q_fmax) if i not in out_]
        q_body = [b for i, b in enumerate(q_body) if i not in out_]
    jr = sp["crusher.jack_diameter_mm"] / 2
    fw = sp["crusher.frame_clear_width_mm"]

    # ---------------------------------------------------------------- assemble
    def arr(lst, shape=None, dtype=float):
        if not lst:
            return np.zeros(shape or (0,), dtype=dtype)
        return np.array(lst, dtype=dtype)

    nf = len(beams["fA"][0]) if beams["fA"] else 15
    mass = np.array(bld.mass)
    rot = np.array(bld.rot)
    floor_m = 1e-12
    mass = np.maximum(mass, floor_m)
    rot = np.maximum(rot, floor_m)
    # each glue point's rate exponent: a glue line's (glue.rate_exponent), the wood's for
    # a face through one piece (wood.rate_exponent: ~4 % per decade, USDA Wood Handbook
    # ch. 5); unbreakable ties and overlaps none
    kinds = np.array([p_.kind for p_ in patches] + ["glue"])
    sp_patch = np.asarray(springs["patch"], int)
    c_rate = np.where(np.isin(kinds[np.where(sp_patch >= 0, sp_patch, len(patches))],
                              ("solid", "block")),
                      sp.values.get("wood.rate_exponent", 0.0),
                      sp.values.get("glue.rate_exponent", 0.0)) if len(sp_patch) else \
        np.zeros(0)
    c_rate = np.where(np.asarray(springs["breakable"], bool), c_rate, 0.0) \
        if len(sp_patch) else c_rate
    model = Model(
        X=np.array(bld.X), mass=mass, rot_inertia=rot, node_body=np.array(bld.node_body),
        b_i=arr(beams["i"], dtype=int), b_j=arr(beams["j"], dtype=int),
        b_L0=arr(beams["L0"]), b_E0=arr(beams["E0"], (0, 3, 3)),
        b_fy=arr(beams["fy"], (0, nf)), b_fz=arr(beams["fz"], (0, nf)),
        b_fA=arr(beams["fA"], (0, nf)), b_E=arr(beams["E"]), b_ft=arr(beams["ft"]),
        b_fc=arr(beams["fc"]), b_Gf=arr(beams["Gf"]), b_GJ=arr(beams["GJ"]),
        b_A=arr(beams["A"]), b_fv=arr(beams["fv"]), b_fv3=arr(beams["fv3"]),
        b_nply=arr(beams["nply"]),
        b_Wt=arr(beams["Wt"], (0, 2)), b_kGA=arr(beams["kGA"]),
        b_body=arr(beams["body"], dtype=int),
        s_nodes=arr(shells["nodes"], (0, 3), int),
        s_E0=arr(shells["E0"], (0, 3, 3)), s_P=arr(shells["P"], (0, 3, 3)),
        s_t=arr(shells["t"]), s_body=arr(shells["body"], dtype=int),
        **(shellmod.pack(shells["el"]) if shells["el"] else shellmod.empty()),
        a_node=np.array(bld.att_node, int), a_r0=np.array(bld.att_r0),
        c_a=arr(springs["a"], dtype=int), c_b=arr(springs["b"], dtype=int),
        c_area=arr(springs["area"]), c_n0=arr(springs["n0"], (0, 3)),
        c_Kn=arr(springs["Kn"]), c_Ks=arr(springs["Ks"]), c_sig=arr(springs["sig"]),
        c_tau=arr(springs["tau"]), c_GIc=arr(springs["GIc"]), c_GIIc=arr(springs["GIIc"]),
        c_mu=arr(springs["mu"]), c_patch=arr(springs["patch"], dtype=int),
        c_breakable=arr(springs["breakable"], dtype=bool), c_rate=np.asarray(c_rate, float),
        k_a=arr(cables["a"], dtype=int), k_b=arr(cables["b"], dtype=int),
        k_L0=arr(cables["L0"]), k_EA=arr(cables["EA"]), k_Fb=arr(cables["Fb"]),
        k_body=arr(cables["body"], dtype=int),
        p_att=np.array(p_att, int), p_support=np.array(p_sup, bool),
        q_att=np.array(q_att, int), plate_node=plate_node,
        plate_half=np.array([PL / 2, PT / 2, PW / 2]), jack_xz=np.array([mid_x, deck_z]),
        jack_radius=jr, frame_x=(mid_x - fw / 2, mid_x + fw / 2),
        bodies=infos, patches=patches, table_y=table_y, deck_top_y=plate_bottom,
        mid_x=mid_x, deck_z=deck_z,
        bridge_mass_kg=float(sum(i.mass_kg for i in infos if i.kind != "ignore")),
        params={"crush_softening": sp["wood.crush_softening"],
                "glue_creep_factor": sp.values.get("glue.creep_factor", 0.0),
                "glue_prepeak_onset": sp.values.get("glue.prepeak_onset", 1.0),
                "glue_coupon_time_s": sp.values.get("glue.coupon_time_s", 60.0),
                "glue_peak_slip_ratio": sp.values.get("glue.peak_slip_ratio", 1.0),
                "glue_creep_time_s": sp.values.get("glue.creep_time_s", 600.0),
                "fc_proportional": sp.values.get("wood.fc_proportional", 1.0),
                "friction_platform": sp["contact.friction_platform"],
                "friction_plate": sp["contact.friction_plate"],
                "crusher_k_N_per_mm": sp.values.get("crusher.stiffness_N_per_mm", 10000.0),
                "crusher_k_lat_N_per_mm": sp.values.get("crusher.lateral_N_per_mm", 2000.0),
                "deck_bodies": deck_bodies, "deck_y": deck_y,
                "test_rate_mm_per_min": sp["crusher.test_rate_mm_per_min"],
                "c_d0": np.array(springs["d0"], float),
                "fmax_p": np.array(p_fmax, float), "fmax_q": np.array(q_fmax, float),
                "p_lift": p_lift, "seating_mm": sp["contact.seating_mm"],
                "mass_change_kg": mass_change_t / TONNE_PER_KG,
                "glue_fc": np.array(springs["fc"], float),
                "glue_cdep": np.array(springs["cdep"], float),
                "densification_strain": sp.values.get("wood.densification_strain", 0.45),
                "densified_E": sp["wood.E_T_MPa"],
                "bearing_cap": float(ply.f_c),
                "plate_offset_mm": [float(off[0]), float(off[1])],
                "seed": seed},
        questions=_merge_butt_questions(questions), notes=notes)
    model.notes += ov.problems([b.name for b in bodies])
    if misfit_bodies:
        names = sorted(misfit_bodies)
        model.questions.insert(0, {
            "id": "plate_fit", "body": names[0], "topic": "rules", "severity": "important",
            "question": f"The loading plate ({PL:g} x {PW:g} x {PT:g} mm, §12.5) does not "
                        f"fit where it goes at mid-span: it would start up to "
                        f"{misfit_mm:.1f} mm inside " + ", ".join(names[:8])
                        + (" ..." if len(names) > 8 else "") + ". The rules ban the "
                        "crushing test for a bridge the plate does not fit (§8.9 clear "
                        "opening, §8.2.3.1 deck width): widen the deck or move those parts.",
            "assumed": "those parts are clear of the plate (as if it were trimmed to fit)",
            "answer_with": {"cad": "make the clear width between the parts at mid-span "
                                   f"more than {PW:g} mm and the clear height above the "
                                   "deck more than the plate and the jack need"}})
    # a rigid (unbreakable) piece directly under the loading plate takes the load without
    # any crushing: say so
    glued_to: dict[int, set] = {}
    for pa in patches:
        if pa.kind == "block":  # a plane inside a block is not a joint
            continue
        glued_to.setdefault(pa.body_a, set()).add(pa.body_b)
        glued_to.setdefault(pa.body_b, set()).add(pa.body_a)
    rigid_path = []
    grain_q = ("<x | y | z: the axis its sticks' fibres run along (a seat whose layers run "
               "along the post it carries bears on end grain, ~6x stronger than across the "
               "grain); leave out if not known>")
    for bi, sh in enumerate(shapes):
        if sh.kind != "block" or ov.for_body(bodies[bi].name).get("kind"):
            continue  # an explicit kind answers the question
        if len(glued_to.get(bi, ())) >= 2:
            rigid_path.append(bodies[bi].name)
        lo_, hi_ = bodies[bi].mesh.bbox
        if (lo_[0] < mid_x + PL / 2 and hi_[0] > mid_x - PL / 2 and lo_[2] < deck_z + PW / 2
                and hi_[2] > deck_z - PW / 2 and hi_[1] > plate_bottom - 45):
            nm = bodies[bi].name
            if bi in block_cells:
                q_ = (f"'{nm}' sits under the loading plate and is modelled as a compact "
                      f"block of {block_cells[bi].n_cells} cell(s) of its wood (about "
                      f"{cell_mm:g} mm) joined by the wood's own stiffness and strength: it "
                      "deforms and can split between its cells, but it is neither a member "
                      "nor a plate (no bending within a cell, no Poisson effect). Is it "
                      "really a solid block of glued sticks, and which way do their fibres "
                      "run? If it is built from plates or members, model it as such.")
                a_ = ("a block of the wood's cells, its fibres in every plane between them "
                      "unless its grain is given (E_T and f_t90 across them; bearing under "
                      "the plate at the wood's crushing strength across the grain)")
            else:
                q_ = (f"'{nm}' sits under the loading plate and is modelled as a rigid "
                      "block (it is compact, so it is neither a member nor a plate): it "
                      "cannot crush or split. Is it really a solid block of glued sticks? If "
                      "it is built from plates or members, model it as such.")
                a_ = ("rigid, unbreakable (bearing under the plate is limited to the wood's "
                      "crushing strength across the grain)")
            model.questions.insert(0, {
                "id": f"rigid_under_plate:{nm}", "body": nm,
                "topic": "design", "severity": "important", "question": q_, "assumed": a_,
                "answer_with": {"bodies": {nm: {
                    "kind": "<block to confirm it | rigid to keep it one rigid body | bar if "
                            "it is a member>",
                    "grain": grain_q,
                    "axis": "<x | y | z: the direction it spans between its supports, "
                            "for a bar (a floor beam wider than it is long)>",
                    "layup": "<N-ply UD: its sticks, for a bar>"}}}})
    if rigid_path:
        groups = sorted({_group(n) for n in rigid_path})
        cells_on = any(bodies[bi].name in rigid_path for bi in block_cells)
        model.questions.append({
            "id": "rigid_blocks", "body": rigid_path[0], "topic": "geometry",
            "severity": "check",
            "question": (
                f"{len(rigid_path)} compact part(s) glued between other parts are modelled "
                f"as blocks of the wood's cells (about {cell_mm:g} mm, block.cell_mm) joined "
                "by the wood's own stiffness and strength: they deform and can split "
                "between their cells (block_fracture), with their fibres in every plane "
                "between the cells unless their grain is given; they do not bend as "
                "members within a cell: " if cells_on else
                f"{len(rigid_path)} compact part(s) glued between other parts are modelled "
                "as rigid blocks (they neither bend nor break; their glue lines and bearing "
                "faces still crush): ") + "; ".join(groups[:15])
                        + (" ..." if len(groups) > 15 else "") + ". Fine for solid blocks of "
                        "glued sticks; if one is a thin-walled or hollow build, or could "
                        "snap, model it as a member; which way do their sticks' fibres run?",
            "assumed": ("blocks of the wood's cells, fibres in the planes between them (E_T, "
                        "f_t90, rolling shear half the way)" if cells_on else "rigid blocks"),
            "answer_with": {"bodies": {groups[0] + "*": {
                "kind": "<block | rigid | bar>", "grain": grain_q}},
                "acknowledge": ["rigid_blocks"]}})
    capped = [f"{b.name}: {b.mesh.capped_loops} face(s) could not be decoded from the CAD "
              "and were closed over (usually a small hole)" for b in bodies
              if b.mesh.capped_loops]
    model.notes += capped
    # faces read only approximately: a surface other than a plane or a cylinder (spline,
    # cone, sphere, freeform) becomes a flat best-fit facet; an undecoded face is closed
    # over. One question per body; important when a twentieth of its surface is affected
    for bi, b in enumerate(bodies):
        if shapes[bi].kind not in ("bar", "shell", "frame", "block"):
            continue
        other = [k for k, f in enumerate(b.faces) if f.kind == "other"]
        if not other and not b.mesh.capped_loops:
            continue
        Vt = b.mesh.vertices[b.mesh.triangles]
        ta = 0.5 * np.linalg.norm(np.cross(Vt[:, 1] - Vt[:, 0], Vt[:, 2] - Vt[:, 0]), axis=1)
        frac = float(ta[np.isin(b.mesh.tri_face, other + [-1])].sum() / max(ta.sum(), 1e-9))
        what = ", ".join(x for x in (
            f"{len(other)} curved face(s) of a spline, cone or freeform surface flattened"
            if other else "",
            f"{b.mesh.capped_loops} undecoded face(s) closed over"
            if b.mesh.capped_loops else "") if x)
        model.questions.append({
            "id": f"cad_approximated:{b.name}", "body": b.name, "topic": "geometry",
            "severity": "important" if frac >= 0.05 else "check",
            "question": f"'{b.name}' has faces bridgesim reads only approximately ({what}: "
                        f"{frac:.1%} of its surface). Its volume, mass and glued area are "
                        "off there. Is any of those faces glued or bearing load?",
            "assumed": "the faces as read (flat facets)",
            "answer_with": {"cad": "draw those faces as planes or cylinders (a polyline "
                                   "instead of a spline), or export the bridge as a "
                                   "multi-body STL and inspect that",
                            "acknowledge": [f"cad_approximated:{b.name}"]}})
    # plates made of narrow arms (an X, a frame): about one element across an arm makes
    # the arms too stiff in in-plane bending and places their cracks coarsely
    narrow: dict[str, tuple[str, float, float]] = {}
    for bi, (_ids, _trib, h_) in shell_mesh.items():
        sh = shapes[bi]
        arm = 2 * sh.poly.area / max(sh.poly.length, 1e-9)
        if arm < h_:
            narrow.setdefault(_group(bodies[bi].name), (bodies[bi].name, arm, h_))
    frames: dict[str, tuple[str, int, float, float]] = {}
    for bi, sh in enumerate(shapes):
        if sh.kind == "frame":
            frames.setdefault(_group(bodies[bi].name), (
                bodies[bi].name, len(sh.frame), 2 * sh.poly.area / max(sh.poly.length, 1e-9),
                sh.T))
    if frames:
        model.questions.append({
            "id": "frame_plates", "body": next(iter(frames.values()))[0],
            "topic": "geometry", "severity": "info",
            "question": f"{len(frames)} plate(s) made of narrow arms are modelled as frames "
                        "of members along the centre lines of their arms (each arm a beam "
                        "with its local width and the plate's thickness, sticks along the "
                        "arm, butt joints included, the arms joined rigidly where they "
                        "meet): " + "; ".join(
                            f"{g} ({n} arm(s), ~{a:.0f} mm wide, {t:.1f} mm thick)"
                            for g, (_n, n, a, t) in list(frames.items())[:8])
                        + ". If one is really a solid plate whose sticks all run one way, "
                        "set its kind to shell.",
            "assumed": "frames as listed",
            "answer_with": {"bodies": {next(iter(frames)) + "*": {"kind": "<frame | shell>"}},
                            "acknowledge": ["frame_plates"]}})
    if narrow:
        model.questions.append({
            "id": "narrow_plates", "body": next(iter(narrow.values()))[0],
            "topic": "geometry", "severity": "check",
            "question": f"{len(narrow)} plate(s) are made of narrow arms (an X, a frame), "
                        "about one shell element across an arm: " + "; ".join(
                            f"{g} (arms ~{a:.0f} mm wide, elements {h:.0f} mm)"
                            for g, (_n, a, h) in list(narrow.items())[:8])
                        + ". Their arms come out ~4x too stiff in in-plane bending and their "
                        "cracks are placed coarsely (one grain direction for the whole "
                        "plate, too). If such a part is really members crossing or framing "
                        "together, draw each member as its own body: it becomes a bar with "
                        "its own grain and section.",
            "assumed": "plates with their elements as listed",
            "answer_with": {"cad": "split the part into one body per member (SolidWorks: "
                                   "Split or separate extrusions), then inspect again",
                            "acknowledge": ["narrow_plates"]}})
    model.params["mesh_scale"] = float(mesh_scale)
    if block_cells:  # compact blocks cut into cells (block.cell_mm)
        model.params["block_cell_mm"] = cell_mm
        model.params["block_cells"] = {bodies[bi].name: lat.n_cells
                                       for bi, lat in block_cells.items()}
        model.params["block_faces"] = n_block_faces
    model.params["s_snap"] = np.array(snap_el, bool)
    model.params["s_snap3"] = np.array(snap3, bool).reshape(-1, 3)
    if n_snap:
        model.notes.append(
            f"{n_snap} shell element(s) are larger than the crack band their fracture energy "
            f"allows ({', '.join(sorted(snap_modes))}): their strength in that mode is "
            f"lowered (down to {snap_excess:.0%}) so that they dissipate the fracture energy "
            "(Bazant & Oh's crack band limit): where they crack first depends on the mesh; a "
            "finer mesh (--fidelity fine) reduces this")
    return model


def _mesh_robust(poly, h: float, name: str, fixed=None):
    """Mesh a plate outline; if the element size is too coarse for the outline (a narrow
    plate at --fidelity quick), refine until the mesh covers the plate."""
    last = None
    for _ in range(6):
        try:
            P2, tri = mesh_polygon(poly, h, fixed=fixed)
            if len(tri) >= 2:
                a = P2[tri]
                area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                                    - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))
                if area.sum() >= 0.8 * poly.area:
                    return P2, tri, h
        except (ValueError, IndexError) as exc:  # too few points for a triangulation
            last = exc
        h *= 0.6
    raise ValueError(f"model build failed: could not mesh the plate '{name}' "
                     f"({poly.area:.0f} mm^2){': ' + str(last) if last else ''}")


THICK_H = 1.5  # smallest shell element / plate thickness (Kirchhoff plates)

#: a plate whose arms are narrower than this many shell elements (at the standard element
#: size, so the choice is the same at every fidelity) is a frame of members along its
#: arms; up to FRAME_SLENDER_ACROSS elements when its arms are slender (their centre
#: lines at least FRAME_SLENDER times as long as they are wide): a strip two or three
#: constant-strain elements wide is still about twice too stiff in in-plane bending
FRAME_MAX_ACROSS = 1.6
FRAME_SLENDER_ACROSS = 3.0
FRAME_SLENDER = 4.0


def _frame_chains(sh: Shape, force: bool = False, patches=()) -> list | None:
    """The arms of a plate made of narrow arms (an X, a frame, a curved rib, a thick
    strut-like plate): their centre lines (:func:`skeleton.medial_chains`), when the
    plate's mean arm width 2 A / P is below ``FRAME_MAX_ACROSS`` standard shell elements
    (``FRAME_SLENDER_ACROSS`` when the arms are slender; or ``force``), the arms, at
    their local width, cover at least 80 % of the plate, and what is glued to it lies on
    its arms: at least half of each glued face within a quarter arm width of them, each
    glued edge within three quarters (the arms' ends reach their end faces through rigid
    arms); otherwise it stays a plate. ``patches``: (overlap polygon, origin, e1, e2) of
    each glued patch."""
    from bridgesim.sim.skeleton import Chain, medial_chains, resample

    poly = sh.poly
    if poly is None or poly.area <= 0:
        return None
    arm = 2 * poly.area / max(poly.length, 1e-9)
    h = _shell_h(sh)
    if not force and arm >= FRAME_SLENDER_ACROSS * h:
        return None
    try:
        chains = [c for c in medial_chains(poly) if c.length > 1e-6]
    except (ValueError, IndexError, RuntimeError):  # a degenerate outline
        return None
    if not chains:
        return None
    widths = np.concatenate([c.width for c in chains])
    width = float(np.median(widths))
    # slender arms: long against their width, and narrow along (nearly) all of it: a
    # plate with a deep part (a 140 mm gusset with 60 mm tails) is a plate there, not a
    # strut, whatever its tails make the median width
    slender = sum(c.length for c in chains) >= FRAME_SLENDER * width and \
        float(np.percentile(widths, 90)) <= FRAME_SLENDER_ACROSS * h
    if not force and arm >= FRAME_MAX_ACROSS * h and not slender:
        return None
    bnd = poly.boundary
    parts = []
    for c in chains:
        r = resample(c, max(arm / 2, 2.0), bnd)
        parts.append(LineString(r.pts).buffer(max(float(r.width.min()) / 2, 0.1)))
        parts += [Point(q).buffer(w / 2) for q, w in zip(r.pts, r.width, strict=True)]
    region = shapely.union_all(parts)
    if region.intersection(poly).area / poly.area < 0.8 and not force:
        return None
    if force:
        return chains
    inside = poly.buffer(1e-6)
    spurs = 0
    for inter, oa, e1, e2 in patches:
        if inter.geom_type != "Polygon" or inter.area <= 0:
            continue
        P3 = oa + np.outer(np.asarray(inter.exterior.coords)[:, 0], e1) \
            + np.outer(np.asarray(inter.exterior.coords)[:, 1], e2)
        P2 = np.c_[(P3 - sh.origin) @ sh.e1, (P3 - sh.origin) @ sh.e2]
        flat = Polygon(P2).buffer(0)
        face = flat.area >= 0.25 * inter.area  # glued on the plate's face, else its edge

        def carried(reg, face=face, flat=flat, P2=P2):
            if face:
                return flat.intersection(reg.buffer(0.25 * arm)).area >= 0.5 * flat.area
            return reg.buffer(0.75 * arm).contains(Point(P2.mean(0)))

        if carried(region):
            continue
        # a glued part off the arms (a wing of the plate): an arm out to it from the
        # nearest point of the centre lines, if that runs inside the plate
        q = np.asarray(flat.representative_point().coords[0] if face else P2.mean(0))
        if spurs >= 8 or not inside.buffer(1e-3).contains(Point(q)):  # edges: on it
            return None
        best = min((LineString(c.pts) for c in chains), key=lambda ln: ln.distance(Point(q)))
        p0 = np.asarray(best.interpolate(best.project(Point(q))).coords[0])
        seg = LineString([p0, q])
        if seg.length < 1e-6 or not inside.buffer(1e-3).contains(seg):
            return None
        sp_ = Chain(np.array([p0, q]),
                    np.array([2 * bnd.distance(Point(x)) for x in (p0, q)]), spur=True)
        chains.append(sp_)
        spurs += 1
        region = region.union(seg.buffer(max(float(sp_.width.min()) / 2, 0.25 * arm)))
        if not carried(region):
            return None
    return chains


def _shell_h(sh: Shape) -> float:
    """Shell element size: ~1/5 of the plate's size, 10-30 mm, at least 2.5 elements
    across a strip, and at least 2.5 times the plate's thickness."""
    h = float(np.clip(math.sqrt(sh.poly.area) / 5, 10.0, 30.0))
    h = min(h, max(sh.Wp / 2.5, 6.0))
    # Kirchhoff (DKT) plates ignore shear deformation, which is only right for elements
    # several times larger than the thickness: thick plates get coarse elements
    return max(h, min(THICK_H * sh.T, 60.0))


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


__all__ = ["Overrides", "build_model", "find_patches", "mesh_polygon", "analyse_shape",
           "LaminateProps", "G_MM_S2", "Point", "plane_basis"]
