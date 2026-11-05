"""Turn CAD bodies into the structural model of the crush simulation.

Every solid body is looked at the way an engineer would idealise it:

* **bar** (a member: long, with a compact cross-section): a chain of corotational fibre
  beam elements on its centre line, with its real cross-section and laminate;
* **shell** (a plate: gusset, web, deck, diaphragm): triangular shell elements on its
  mid-surface, meshed from the body's real outline (holes included);
* **block** (a chunky piece: cleat, saddle, toggle): one rigid body with its real mass
  and rotational inertia;
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
MIN_PATCH_MM2 = 2.0


@dataclass
class Overrides:
    """What a designer (or an AI agent) tells the simulation that the CAD file does not.

    ``bodies`` maps a body name or glob pattern (``"Floss*"``) to fields: ``kind``
    (bar/shell/block/cable/ignore), ``layup`` (any text bridgesim can parse, e.g.
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
    acknowledge: list[str] = field(default_factory=list)  # question ids (or patterns)
    unknown: list[str] = field(default_factory=list)  # top-level keys we do not read
    used: set = field(default_factory=set)  # patterns and glue keys that matched

    KINDS = ("bar", "shell", "frame", "block", "cable", "ignore")
    FIELDS = ("kind", "layup", "grain", "glue_factor", "strength_factor", "butt_stagger_mm",
              "section_scale")

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
        out = [f"overrides: top-level key {k!r} is not read (bodies, glue, params and "
               "acknowledge are; 'cad' and 'cli' in a question's answer_with are things to "
               "do, not overrides)" for k in self.unknown]
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
                elif k == "kind" and v not in self.KINDS:
                    out.append(f"overrides: kind {v!r} for {pat!r} is not one of "
                               f"{', '.join(self.KINDS)}")
                elif k == "layup" and parse_override_layup(str(v)) is None:
                    out.append(f"overrides: layup {v!r} for {pat!r} cannot be read (write "
                               "e.g. '10-ply UD', '2-ply 0-90', '4-ply +-45', 'block')")
                elif k == "grain" and v not in ("long", "span", "arm"):
                    out.append(f"overrides: grain {v!r} for {pat!r} must be long, span or "
                               "arm (a frame's plies laid along each of its arms)")
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

        return cls(bodies=dict(d.get("bodies", {}) or {}),
                   params={k: num("params", k, v)
                           for k, v in (d.get("params", {}) or {}).items()},
                   glue={k: num("glue", k, v) for k, v in (d.get("glue", {}) or {}).items()},
                   acknowledge=[str(a) for a in (ack if isinstance(ack, list) else [ack])],
                   unknown=[str(k) for k in d if k not in ("bodies", "params", "glue",
                                                           "acknowledge")])

    def acknowledged(self, qid: str) -> bool:
        """Whether a question was acknowledged ('the design is as drawn', 'confirmed')."""
        return any(qid == a or fnmatch.fnmatchcase(qid, a) for a in self.acknowledge)


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


def analyse_shape(b: CadBody, layup: Layup, forced: str | None) -> Shape:
    m = b.mesh
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
            if err <= 0.12 and (best is None or poly.area > best[0].area):
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


def find_patches(bodies: list[CadBody]) -> list[tuple[int, int, Polygon, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]]:
    """(body a, body b, overlap polygon in a's face basis, origin, e1, e2, normal a->b)."""
    lo = np.array([b.mesh.bbox[0] for b in bodies]) - CONTACT_TOL_MM
    hi = np.array([b.mesh.bbox[1] for b in bodies]) + CONTACT_TOL_MM
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
                    if abs(float(np.dot(ob - oa, na))) > CONTACT_TOL_MM:
                        continue
                    ring = [np.c_[(lp - oa) @ e1, (lp - oa) @ e2] for lp in fbf.loops]
                    k = int(np.argmax([Polygon(r).area for r in ring]))
                    pbp = Polygon(ring[k], [r for i, r in enumerate(ring) if i != k]
                                  ).buffer(0)
                    inter = pa.intersection(pbp)
                    if inter.area >= MIN_PATCH_MM2:
                        out.append((ia, ib, inter, oa, e1, e2, na))
    return out


def _sample_polygon(poly, spacing: float, max_pts: int = 25) -> np.ndarray:
    """Points spread over a (multi)polygon on a grid aligned with its minimum rotated
    rectangle, so long thin strips (a deck glued along a stringer) get a row of points
    along their length instead of one point."""
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
        nu = max(1, int(round(Lu / sp_)))
        nw = max(1, int(round(Lw / sp_)))
        if nu * nw > max_pts:
            sp_ *= 1.25
            continue
        a = (np.arange(nu) + 0.5) / nu * Lu - Lu / 2
        b = (np.arange(nw) + 0.5) / nw * Lw - Lw / 2
        A, B = np.meshgrid(a, b, indexing="ij")
        pts = c0 + np.outer(A.ravel(), u) + np.outer(B.ravel(), w)
        pts = pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]
        if (len(pts) >= min(3, nu * nw) or sp_ < 0.5) and len(pts):
            return pts
        sp_ /= 1.5
    c = np.array(poly.representative_point().coords[0])
    return c[None, :]


# ------------------------------------------------------------------------------- meshing


def mesh_polygon(poly: Polygon, h: float, seeds: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a polygon (holes allowed) with elements of about size h. Holes
    smaller than about h (floss holes, pin holes) are filled: they do not change the
    plate's stiffness or strength noticeably and would force tiny elements."""
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
    # merge points closer than 0.3 h (short CAD edges, seeds next to the boundary): they
    # would make sliver elements and a tiny stable time step
    tree = cKDTree(allp)
    drop = np.zeros(len(allp), bool)
    for i, j in sorted(tree.query_pairs(0.3 * h)):
        if not drop[i] and not drop[j]:
            drop[j] = True
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

    def _at(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return p if self.imperf is None else self.imperf(p)

    def node(self, p, body: int) -> int:
        self.X.append(self._at(p))
        self.mass.append(0.0)
        self.rot.append(0.0)
        self.node_body.append(body)
        return len(self.X) - 1

    def attach(self, node: int, p) -> int:
        self.att_node.append(node)
        self.att_r0.append(self._at(p) - self.X[node])
        return len(self.att_node) - 1


def grain_vectors(sh) -> list[np.ndarray]:
    """Grain direction of each ply of a bar or a plate (unit vectors); none for a block."""
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


def grain_bearing(sh, n: np.ndarray, f_c: float, f_c90: float) -> tuple[float, float]:
    """Crushing strength of a body pressed along n (MPa), averaged over its plies, and
    the share of its plies that meet n with their end grain (within 30 degrees)."""
    gs = grain_vectors(sh)
    if not gs:
        return f_c90, 0.0
    cs = [abs(float(g @ n)) for g in gs]
    return (float(np.mean([hankinson(f_c, f_c90, c) for c in cs])),
            float(np.mean([c > 0.866 for c in cs])))


def _group(name: str) -> str:
    """Name without its side / copy suffix ('Top Chord (..) - Far' -> 'Top Chord (..)')."""
    return re.sub(r"( - .*| \(mirror copy\).*| #\d+)$", "", name).strip()


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


def _laminated_member(name: str, sh, lam, L: float, mat: Material, o: dict,
                      questions: list, f_rolling: float = 2.0,
                      bond: float = 6.0) -> tuple[float, float]:
    """Tension-strength factor and shear strength of a member laminated from sticks.

    A ply longer than one stick has butt joints; its load goes round each joint through
    the glue lines to the next plies. With joints staggered by ``s`` the weakest tension
    path cuts k plies at their joints and shears the glue between them:
    ``P_k = (n - k) P_ply + (k - 1) tau_g w s``, the smallest over k (k = 1 is the net
    section, k = n a path through every joint). Bending shear between plies runs in the
    glue lines between them: :func:`interlaminar_strength` (the bond line, pressed while
    it cured, ``bond``; the plies along and across their grain, ``f_rolling``)."""
    n = len(lam.angles)
    f_v = mat.f_v_MPa.value
    if sh.layup.kind != "laminate" or n < 2:
        return 1.0, f_v
    gf = float(o.get("glue_factor", 1.0))
    tau = mat.glue.tau_g_MPa.value * max(gf, 1e-3)
    fv = interlaminar_strength(lam.angles, f_v, f_rolling, bond * max(gf, 1e-3))
    Ls = mat.stick.length_mm
    if L <= 1.02 * Ls or o.get("butt_stagger_mm") == 0:
        return 1.0, fv
    s = float(o.get("butt_stagger_mm") or Ls / 2)  # brick pattern: half a stick
    w = sh.Wp  # glue-line width (plies stacked through the thickness)
    P_ply = lam.f_t * sh.Wp * sh.T / n
    P = min((n - k) * P_ply + (k - 1) * tau * w * s for k in range(1, n + 1))
    kt = min(1.0, P / (n * P_ply))
    g = _group(name)
    if o.get("butt_stagger_mm") is None and not any(
            q["id"] == f"butt_joints:{g}" for q in questions):
        questions.append({
            "member": f"'{g}' ({L:.0f} mm long, {kt:.0%} of a solid member's tension "
                      f"strength with joints {s:.0f} mm apart)",
            "id": f"butt_joints:{g}", "body": name, "topic": "design", "severity": "check",
            "question": f"'{g}' is {L:.0f} mm long, longer than one stick "
                        f"({Ls:.0f} mm), so its plies have butt joints. How far apart are "
                        "the joints of neighbouring plies? This sets its tension "
                        f"strength: {kt:.0%} of a solid member with joints {s:.0f} mm "
                        "apart; joints closer together let the glue between them shear "
                        "first.",
            "assumed": f"brick pattern: each ply's joints {s:.1f} mm (half a stick) from "
                       "its neighbours'",
            "answer_with": {"bodies": {g + "*": {
                "butt_stagger_mm": "<distance between joints of neighbouring plies, mm; "
                                   "0 if the plies have no joints>"}}}})
    return kt, fv


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

    # ---------------------------------------------------------------- body analysis
    shapes: list[Shape] = []
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
        if forced in ("block", "ignore") and questions and \
                questions[-1].get("id") == f"layup:{b.name}":
            questions.pop()  # set rigid or left out: what it is made of changes nothing
        sh = analyse_shape(b, lay, forced)
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
            questions[-1]["question"] += (" It is compact, so it is modelled as a rigid "
                                          "block whatever it is made of; if it is built "
                                          "from plates or members, set its kind too.")
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
    raw_patches = find_patches(bodies)
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
    bar_axis: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    shell_tree: dict[int, cKDTree] = {}
    shell_mesh: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    frame_tree: dict[int, cKDTree] = {}
    frame_h: dict[int, float] = {}
    beams: dict[str, list] = {k: [] for k in ("i", "j", "L0", "E0", "fy", "fz", "fA", "E",
                                              "ft", "fc", "Gf", "GJ", "A", "fv", "fv3",
                                              "nply", "Wt", "kGA", "body")}
    shells: dict[str, list] = {k: [] for k in ("nodes", "E0", "P", "t", "el", "body")}
    n_snap = 0
    snap_excess = 1.0
    snap_modes: set = set()
    G_f = sp["wood.G_f_N_per_mm"]
    il_strength = {"f_rolling": sp.values.get("wood.f_rolling_MPa", 2.0),
                   "bond": sp.values.get("glue.interlaminar_MPa", 6.0)}

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
            beams["fc"].append(lam.f_c * Ef * Sf)
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
            fv_in = inplane_split_strength(lam.angles, mat.f_v_MPa.value,
                                           0.5 * min(ply.f_t, ply.f_c))
            # uniform elements; joints attach to the nearest node through a rigid arm
            h = min(30.0 * mesh_scale, L / 4 * min(mesh_scale, 1.0)) if L >= 60 \
                else max(L / 2, 1e-3)
            n_el = max(2, int(math.ceil(L / h - 1e-9)))
            st2 = list(np.linspace(s0, s1, n_el + 1))
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
            ny, nz = (5, 3) if Wp >= T else (3, 5)
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
            for k_e, (a_, b_) in enumerate(zip(ids[:-1], ids[1:], strict=True)):
                We = float(wid[k_e])
                A_e = We * T if sh.fill < 0.75 else A_true
                gy = off[k_e] + ((np.arange(ny) + 0.5) / ny - 0.5) * We * _exact(ny)
                FY, FZ = np.meshgrid(gy, gz, indexing="ij")
                fA = np.full(ny * nz, A_e / (ny * nz))
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
                beams["E"].append(lam.E_x * Ef)
                beams["ft"].append(lam.f_t * Ef * Sf * kt)
                beams["fc"].append(lam.f_c * Ef * Sf)
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
            info.E_eff_MPa, info.f_t_eff_MPa, info.f_c_eff_MPa = lam.E_x, lam.f_t * kt, lam.f_c
            info.section_mm = (round(Wm if sh.fill < 0.75 else Wp, 2), round(T, 2))
            info.length_mm = L
            info.nodes = list(ids)
        elif sh.kind == "shell":
            T = sh.T * ssc
            lam_ref = laminate(ply, sh.layup.angles, T)
            o_mid = sh.origin - sh.n * T / 2
            area = sh.poly.area
            h = _shell_h(sh) * mesh_scale
            P2, tri, h = _mesh_robust(sh.poly, h, b.name)  # glue attaches by rigid arms
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
                ed = shellmod.element_data(Ploc[:, :2], angs, T, ply, Ef, Sf, Gm,
                                           lam_ref.E_x * Ef * T)
                if ed["snapback"]:
                    n_snap += 1
                    snap_excess = max(snap_excess, ed["excess"])
                    snap_modes.update(ed["snap_modes"])
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
            info.f_t_eff_MPa, info.f_c_eff_MPa = lam_ref.f_t, lam_ref.f_c
            info.thickness_mm = T
            info.nodes = list(ids)
        elif sh.kind == "frame":
            _frame_body(bi, b, sh, info, o, ssc, Ef, Sf, mass_t)
        elif sh.kind == "cable":
            info.nodes = []  # a floss cable acts between the parts it is tied to
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
    for ia, ib, inter, oa, e1, e2, na in raw_patches:
        if shapes[ia].kind in ("ignore", "cable") or shapes[ib].kind in ("ignore", "cable"):
            continue
        lead = [b_ for b_ in (ia, ib) if shapes[b_].kind == "shell"]
        pts3 = None
        if lead:
            bl = min(lead, key=lambda b_: shell_mesh[b_][2])
            ids, trib, hl = shell_mesh[bl]
            X3 = np.array([bld.X[i] for i in ids])
            proj = X3 - np.outer((X3 - oa) @ na, na)
            q2 = np.c_[(proj - oa) @ e1, (proj - oa) @ e2]
            inside = shapely.contains_xy(inter, q2[:, 0], q2[:, 1])
            band = shapely.contains_xy(inter.buffer(0.5 * hl), q2[:, 0], q2[:, 1]) & ~inside
            w = np.where(inside, trib, np.where(band, 0.5 * trib, 0.0))
            sel = w > 0
            if sel.sum() >= 3:
                pts3 = proj[sel]
                wts = w[sel] / w[sel].sum() * inter.area
        if pts3 is None:
            sp_ = max(3.0, min(elem_size(ia), elem_size(ib)))
            pts2 = _sample_polygon(inter, sp_, max_pts=400)
            pts3 = oa + np.outer(pts2[:, 0], e1) + np.outer(pts2[:, 1], e2)
            wts = np.full(len(pts3), inter.area / len(pts3))
        # where the joint is, from the CAD overlap alone (not from the mesh): the key of
        # its random glue strength, so a seed draws the same bridge at every fidelity
        cc = inter.centroid
        cad_c = oa + cc.x * e1 + cc.y * e2
        patch_pts3.append((ia, ib, pts3, na, inter.area, cad_c, wts))

    # ---------------------------------------------------------------- glue springs
    sp_glue = {k: sp[f"glue.{k}"] for k in ("K_n_N_per_mm3", "K_s_N_per_mm3",
                                            "peel_to_shear", "G_Ic_N_per_mm",
                                            "G_IIc_N_per_mm", "friction")}
    # peak cohesive strength such that the simulated 20 mm lap test gives tau_g. How
    # much the peak exceeds the average depends on how many nodes the glue points hang
    # on: with one node per side the glue shears uniformly (1.0); as soon as a side's
    # stretching is resolved the shear-lag peak appears. Fitted to the lap coupon with
    # its glue on 1, 2, 3, 4, 5 and 9 nodes per side (1.01, 1.11, 1.20, 1.15, 1.16, 1.15;
    # tests/test_sim.py checks it), so a joint's strength does not jump with the mesh
    peak_over_avg = sp.values.get("glue.tau_peak_over_avg", 1.155)

    def peak_factor(n_nodes: int) -> float:
        return 1.0 + (peak_over_avg - 1.0) * (1.0 - math.exp(-2.2 * (n_nodes - 1)))

    end_factor = sp.values.get("glue.end_grain_factor", 0.5)
    zone = sp.values.get("wood.crush_zone_depth_factor", 3.0)

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
    for ia, ib, pts3, na, area, cad_c, wts in patch_pts3:
        key = "|".join(sorted((bodies[ia].name, bodies[ib].name)))
        gf = gf_bridge * scatter(sp["scatter.glue_cov"],
                                 f"glue:{key}:{np.round(cad_c).astype(int).tolist()}")
        gf *= float(ov.for_body(bodies[ia].name).get("glue_factor", 1.0))
        gf *= float(ov.for_body(bodies[ib].name).get("glue_factor", 1.0))
        for key, val in ov.glue.items():
            if _joint_matches(key, bodies[ia].name, bodies[ib].name):
                gf *= val
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
        fa, ea = grain_bearing(shapes[ia], na, ply.f_c, ply.f_c90)
        fb, eb = grain_bearing(shapes[ib], na, ply.f_c, ply.f_c90)
        f_bear = min(fa, fb)
        # the depth of wood that can crush: the weaker side's extent along the normal,
        # at most a few times its thickness (the zone a local load compacts)
        c_dep = min(crush_depth(i, na) for i, f in ((ia, fa), (ib, fb))
                    if f <= 1.05 * f_bear)
        e_end = max(ea, eb)  # share of the plies on a side meeting the glue end-on
        gf *= 1.0 - e_end * (1.0 - end_factor)
        if e_end > 0.5:
            end_grain_joints.append((bodies[ia].name, bodies[ib].name, float(area)))
        att_a = [attach(ia, p) for p in pts3]
        att_b = [attach(ib, p) for p in pts3]
        n_nodes = max(len({bld.att_node[a] for a in att_a}),
                      len({bld.att_node[b] for b in att_b}))
        tau_p = mat.glue.tau_g_MPa.value * peak_factor(n_nodes)
        for wa, a_, b_ in zip(wts, att_a, att_b, strict=True):
            springs["d0"].append(1.0 if gf <= 0 else 0.0)  # glue_factor 0: not glued
            springs["a"].append(a_)
            springs["b"].append(b_)
            springs["area"].append(float(wa))
            springs["n0"].append(na)
            springs["Kn"].append(sp_glue["K_n_N_per_mm3"])
            springs["Ks"].append(sp_glue["K_s_N_per_mm3"] * k_cov)
            gq = max(gf, 1e-3)  # glue_factor 0 is handled as "already broken" (d0 = 1)
            springs["tau"].append(tau_p * gq)
            springs["sig"].append(tau_p * gq * sp_glue["peel_to_shear"])
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
                        "holds about half as well as on side grain. Are these butt joints "
                        "as drawn, or are they reinforced (gussets, laps, sized end grain)?",
            "assumed": f"glue strength x {end_factor:g} on the end-grain share of these "
                       "joints (glue.end_grain_factor; a 0/90 plate's edge, half end "
                       "grain, gets the mean)",
            "answer_with": {"params": {"glue.end_grain_factor":
                                       "<0.5 for plain end-grain joints, 1 if sized or "
                                       "measured as strong as side grain>"},
                            "glue": {names[0]: "<factor for one joint>"}}})

    # bodies touching nothing: tie them to the nearest body (and ask)
    allV = [b.mesh.vertices for b in bodies]
    for bi, b in enumerate(bodies):
        if shapes[bi].kind in ("ignore", "cable") or touching[bi]:
            continue
        best, bd, bp = None, math.inf, None
        tree_b = cKDTree(allV[bi])
        for bj in range(len(bodies)):
            if bj == bi or shapes[bj].kind in ("ignore", "cable"):
                continue
            d, k = tree_b.query(allV[bj])
            j = int(np.argmin(d))
            if d[j] < bd:
                bd, best, bp = float(d[j]), bj, allV[bj][j]
        if best is None or bd > 10.0:
            questions.append({
                "id": f"floating:{b.name}", "body": b.name, "topic": "connection",
                "question": f"'{b.name}' does not touch any other part. How is it held?",
                "assumed": "it is free (it will fall off)",
                "answer_with": {"bodies": {b.name: {"kind": "ignore"}}}})
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
            springs["tau"].append(1e3)
            springs["sig"].append(1e3)
            springs["GIc"].append(1e3)
            springs["GIIc"].append(1e3)
            springs["mu"].append(0.5)
            springs["patch"].append(pid)
            springs["breakable"].append(False)
            springs["d0"].append(0.0)
            springs["fc"].append(1e30)
            springs["cdep"].append(1e30)
            patches[pid].springs.append(len(springs["a"]) - 1)
        questions.append({
            "id": f"loose:{b.name}", "body": b.name, "topic": "connection",
            "question": f"'{b.name}' has no glued face; its nearest part is "
                        f"'{bodies[best].name}' ({bd:.1f} mm away). Is it tied or wedged "
                        "there (e.g. a floss toggle)?",
            "assumed": f"held to '{bodies[best].name}' at its nearest point "
                       "(cannot come loose)",
            "answer_with": {"bodies": {b.name: {"kind": "ignore"}}}})

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
    def bearing(bi: int) -> float:
        """Crushing strength (MPa) of a body pressed vertically, by the angle of each
        ply's grain to the vertical (Hankinson, averaged over the plies, as at the glue
        lines): f_c for a standing UD bar, f_c90 for anything lying flat, in between
        for a standing +-45 plate."""
        sh = shapes[bi]
        if sh.kind == "bar" and sh.layup.kind == "toothpick":
            return ply.f_c90  # a round dowel bears on its side
        return grain_bearing(sh, np.array([0.0, 1.0, 0.0]), ply.f_c, ply.f_c90)[0]

    V_AREA = 16.0  # mm^2 a mesh corner stands for
    p_att, p_sup, p_fmax = [], [], []
    for bi in active:
        if shapes[bi].kind == "cable":
            continue
        V = bodies[bi].mesh.vertices
        fb = bearing(bi)
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
            p_fmax.append(fb * a)
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
    for bi in active:
        if shapes[bi].kind == "cable":
            continue
        V = bodies[bi].mesh.vertices
        sel = ((V >= region_lo) & (V <= region_hi)).all(1)
        pts = [V[sel]]
        fb = bearing(bi)
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
            q_fmax.append(fb * a)

    plate_mass = sp["crusher.plate_mass_kg"] * TONNE_PER_KG
    pc = np.array([mid_x, plate_bottom + PT / 2 + 0.02, deck_z])
    plate_node = bld.node(pc, -1)
    bld.mass[plate_node] = plate_mass
    bld.rot[plate_node] = plate_mass * (PL**2 + PW**2) / 12
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
        c_breakable=arr(springs["breakable"], dtype=bool),
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
                "glue_creep_time_s": sp.values.get("glue.creep_time_s", 600.0),
                "fc_proportional": sp.values.get("wood.fc_proportional", 1.0),
                "friction_platform": sp["contact.friction_platform"],
                "friction_plate": sp["contact.friction_plate"],
                "crusher_k_N_per_mm": sp.values.get("crusher.stiffness_N_per_mm", 20000.0),
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
                "plate_offset_mm": [float(off[0]), float(off[1])],
                "seed": seed},
        questions=_merge_butt_questions(questions), notes=notes)
    model.notes += ov.problems([b.name for b in bodies])
    # a rigid (unbreakable) piece directly under the loading plate takes the load without
    # any crushing: say so
    glued_to: dict[int, set] = {}
    for pa in patches:
        glued_to.setdefault(pa.body_a, set()).add(pa.body_b)
        glued_to.setdefault(pa.body_b, set()).add(pa.body_a)
    rigid_path = []
    for bi, sh in enumerate(shapes):
        if sh.kind != "block" or ov.for_body(bodies[bi].name).get("kind"):
            continue  # an explicit kind answers the question
        if len(glued_to.get(bi, ())) >= 2:
            rigid_path.append(bodies[bi].name)
        lo_, hi_ = bodies[bi].mesh.bbox
        if (lo_[0] < mid_x + PL / 2 and hi_[0] > mid_x - PL / 2 and lo_[2] < deck_z + PW / 2
                and hi_[2] > deck_z - PW / 2 and hi_[1] > plate_bottom - 45):
            model.questions.insert(0, {
                "id": f"rigid_under_plate:{bodies[bi].name}", "body": bodies[bi].name,
                "topic": "design", "severity": "important",
                "question": f"'{bodies[bi].name}' sits under the loading plate and is "
                            "modelled as a rigid block (it is compact, so it is neither a "
                            "member nor a plate): it cannot crush or split. Is it really a "
                            "solid block of glued sticks? If it is built from plates or "
                            "members, model it as such.",
                "assumed": "rigid, unbreakable (bearing under the plate is limited to the "
                           "wood's crushing strength across the grain)",
                "answer_with": {"bodies": {bodies[bi].name: {
                    "kind": "<block to confirm it | bar if it is a member>"}}}})
    if rigid_path:
        groups = sorted({_group(n) for n in rigid_path})
        model.questions.append({
            "id": "rigid_blocks", "body": rigid_path[0], "topic": "geometry",
            "severity": "check",
            "question": f"{len(rigid_path)} compact part(s) glued between other parts are "
                        "modelled as rigid blocks (they neither bend nor break; their glue "
                        "lines and bearing faces still crush): " + "; ".join(groups[:15])
                        + (" ..." if len(groups) > 15 else "") + ". Fine for solid blocks of "
                        "glued sticks; if one is a thin-walled or hollow build, or could "
                        "snap, model it as a member.",
            "assumed": "rigid blocks",
            "answer_with": {"bodies": {groups[0] + "*": {"kind": "<block | bar>"}},
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
    if n_snap:
        model.notes.append(
            f"{n_snap} shell element(s) are larger than the crack band their fracture energy "
            f"allows ({', '.join(sorted(snap_modes))}): they keep their strength and fail as "
            f"brittly as possible, dissipating up to {snap_excess:.1f}x the fracture energy "
            "of that mode; a finer mesh (--fidelity fine) reduces this")
    return model


def _mesh_robust(poly, h: float, name: str):
    """Mesh a plate outline; if the element size is too coarse for the outline (a narrow
    plate at --fidelity quick), refine until the mesh covers the plate."""
    last = None
    for _ in range(6):
        try:
            P2, tri = mesh_polygon(poly, h)
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
    width = float(np.median(np.concatenate([c.width for c in chains])))
    slender = sum(c.length for c in chains) >= FRAME_SLENDER * width
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
