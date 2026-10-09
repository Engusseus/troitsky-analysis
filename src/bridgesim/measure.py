"""Geometric measurements used by the rule checks (rulebook §8).

Members are treated as solid prisms: the centre line plus the cross-section half-size
along each global axis (``geometry.half_extents``). Most checks sample points along each
member every ``SAMPLE_MM`` mm. The table is at the lowest support-node elevation.
Measurements do not depend on the analysis; they describe the unloaded bridge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bridgesim.geometry import half_extents, inclination_from_vertical_deg
from bridgesim.loads import mid_span_x
from bridgesim.schema import Bridge, Material

SAMPLE_MM = 2.0
#: Upper bound on samples over the whole bridge (about 10 MB of points). The schema limits
#: the total member length (``MAX_TOTAL_MEMBER_LENGTH_MM``) so that every member is always
#: sampled at SAMPLE_MM; sampling more coarsely could let a member slip through a clearance.
MAX_TOTAL_SAMPLES = 400_000
_TOL = 0.5  # mm
#: A member standing on the table is cut flush there. The cut may run this many section
#: sizes (max(b, d)) along the member; beyond it the section must clear the table, so a
#: member lying almost flat on the table is still reported.
CUT_ZONE_SIZES = 2.0

DEFAULT_CONSTANTS: dict[str, float] = {
    "cart_width_mm": 150.0,
    "cart_height_mm": 200.0,
    "clear_box_length_mm": 1000.0,
    "clear_box_height_mm": 150.0,
    "clear_opening_mm": 100.0,
    "pier_max_inclination_deg": 1.0,
}


@dataclass
class _Samples:
    member: str
    group: str
    pts: np.ndarray  # (k, 3)
    h: np.ndarray  # (3,) half extents along X, Y, Z
    touches_table: bool
    size: float  # max(b, d) of the section


@dataclass
class Measurements:
    values: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]


def _names(items: list[str], limit: int = 8) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f", … ({len(items)} in total)" if len(items) > limit else "")


def _sample_counts(lengths: list[float]) -> list[int]:
    """Samples per member, every SAMPLE_MM (never coarser, so no rule outcome changes)."""
    ks = [max(2, int(math.ceil(L / SAMPLE_MM)) + 1) for L in lengths]
    if sum(ks) > MAX_TOTAL_SAMPLES:  # unreachable for a schema-valid bridge
        raise ValueError(f"Model too large for the geometric rule checks ({sum(ks)} samples)")
    return ks


def _samples(bridge: Bridge, material: Material, table_y: float) -> list[_Samples]:
    nodes = bridge.node_map()
    secs = bridge.section_map()
    ends = [(np.array(nodes[m.i].xyz), np.array(nodes[m.j].xyz)) for m in bridge.members]
    counts = _sample_counts([float(np.linalg.norm(b - a)) for a, b in ends])
    out = []
    for m, (a, b), k in zip(bridge.members, ends, counts, strict=True):
        t = np.linspace(0.0, 1.0, k)[:, None]
        bm, dm = secs[m.section].dims(material.stick)
        touches = min(a[1], b[1]) <= table_y + _TOL
        out.append(_Samples(m.id, m.group, a + t * (b - a),
                            half_extents(tuple(a), tuple(b), bm, dm), touches, max(bm, dm)))
    return out


def _extreme(S: list[_Samples], axis: int, sign: int, value: float, deck_value: float) -> str:
    """The members (or the deck) whose surface reaches ``value`` along ``axis``."""
    hits = []
    for s in S:
        reach = s.pts[:, axis] + sign * s.h[axis]
        if abs(float(reach.max() if sign > 0 else reach.min()) - value) <= 1e-6:
            hits.append(s.member)
    if hits:
        return _names(hits)
    return "deck" if abs(deck_value - value) <= 1e-6 else "?"


def measure(
    bridge: Bridge, material: Material, constants: dict[str, float] | None = None
) -> Measurements:
    c = {**DEFAULT_CONSTANTS, **(constants or {})}
    nodes = bridge.node_map()
    deck = bridge.deck
    sup = bridge.support_nodes()
    table_y = min(nodes[n].y_mm for n in sup) if sup else 0.0
    xc = mid_span_x(bridge)
    zc = deck.z_center_mm
    S = _samples(bridge, material, table_y)
    out = Measurements()
    v, d = out.values, out.details
    #: Per measurement, what causes its value (shown when a rule using it fails).
    culprits: dict[str, str] = d.setdefault("culprits", {})

    # ---- lengths (§8.2.1) -------------------------------------------------------------
    sx = sorted(nodes[n].x_mm for n in sup)
    left_sup = [x for x in sx if x < xc]
    right_sup = [x for x in sx if x > xc]
    v["span_cc_mm"] = (
        float(np.mean(right_sup) - np.mean(left_sup)) if left_sup and right_sup else 0.0
    )
    supporting = [s for s in S if s.touches_table]
    left = [s for s in supporting if s.pts[:, 0].mean() < xc]
    right = [s for s in supporting if s.pts[:, 0].mean() > xc]
    left_inner = max((float((s.pts[:, 0] + s.h[0]).max()) for s in left), default=xc)
    right_inner = min((float((s.pts[:, 0] - s.h[0]).min()) for s in right), default=xc)
    v["clear_span_mm"] = right_inner - left_inner
    d["clear_span_faces_mm"] = (left_inner, right_inner)
    d["supporting_members"] = [s.member for s in supporting]
    faces = [[s.member for s in left if abs(float((s.pts[:, 0] + s.h[0]).max()) - left_inner)
              <= 1e-6],
             [s.member for s in right if abs(float((s.pts[:, 0] - s.h[0]).min()) - right_inner)
              <= 1e-6]]
    culprits["clear_span_mm"] = (f"inner faces of the supports: {_names(faces[0]) or 'none'} "
                                 f"/ {_names(faces[1]) or 'none'}")

    xmin = min([float((s.pts[:, 0] - s.h[0]).min()) for s in S] + [deck.x_start_mm])
    xmax = max([float((s.pts[:, 0] + s.h[0]).max()) for s in S] + [deck.x_end_mm])
    v["deck_length_mm"] = deck.length_mm
    v["total_length_mm"] = xmax - xmin
    culprits["total_length_mm"] = (f"ends: {_extreme(S, 0, -1, xmin, deck.x_start_mm)} / "
                                   f"{_extreme(S, 0, 1, xmax, deck.x_end_mm)}")

    # ---- heights (§8.2.2) -------------------------------------------------------------
    # Nothing can sit below the table: stick ends touching it are cut flush.
    ymin = max(table_y, min([float((s.pts[:, 1] - s.h[1]).min()) for s in S]
                            + [deck.top_elevation_mm]))
    ymax = max([float((s.pts[:, 1] + s.h[1]).max()) for s in S] + [deck.top_elevation_mm])
    v["deck_height_mm"] = deck.top_elevation_mm - table_y
    v["total_height_mm"] = ymax - ymin
    culprits["total_height_mm"] = (
        f"highest: {_extreme(S, 1, 1, ymax, deck.top_elevation_mm)}")
    # The bridge rests on the base platform, so no node and no deck can be lower than its
    # supports. Report it rather than clip it away: it is a modelling error.
    # A leg standing on the table (exactly one end on it) is cut flush there, so near that
    # end its section may dip below the centreline (for CUT_ZONE_SIZES section sizes); any
    # other member, including one lying along the table, must stay clear of it entirely.
    low = [n.id for n in bridge.nodes if n.y_mm < table_y - 1e-6]
    deck_bottom = deck.top_elevation_mm - deck.thickness_mm
    lowest = min([n.y_mm for n in bridge.nodes] + [deck_bottom])
    through = []
    for s in S:
        bottom = s.pts[:, 1] - s.h[1]
        on = s.pts[[0, -1], 1] <= table_y + 1e-6
        if int(on.sum()) == 1:
            end = s.pts[0] if on[0] else s.pts[-1]
            bottom = bottom[np.linalg.norm(s.pts - end, axis=1) > CUT_ZONE_SIZES * s.size]
        if bottom.size and float(bottom.min()) < table_y - 1e-6:
            through.append(s.member)
            lowest = min(lowest, float(bottom.min()))
    # All supports stand on the one flat platform; a higher one would be a reaction from
    # nowhere (the table is taken at the lowest support).
    floating = [n for n in sup if nodes[n].y_mm > table_y + 1e-6]
    v["below_table_mm"] = max(0.0, table_y - lowest)
    v["above_table_ok"] = float(not low and not through and not floating
                                and deck_bottom >= table_y - 1e-6)
    d["below_table_nodes"] = low
    d["below_table_members"] = through
    d["floating_supports"] = floating
    below = ([f"node {_names(low)}"] if low else []) + (
        [f"member {_names(through)}"] if through else []) + (
        ["deck"] if deck_bottom < table_y - 1e-6 else [])
    culprits["above_table_ok"] = "; ".join(
        ([f"below the table: {', '.join(below)}"] if below else [])
        + ([f"support above the lowest one: {_names(floating)}"] if floating else []))

    # ---- widths (§8.2.3) --------------------------------------------------------------
    zmin = min([float((s.pts[:, 2] - s.h[2]).min()) for s in S] + [zc - deck.clear_width_mm / 2])
    zmax = max([float((s.pts[:, 2] + s.h[2]).max()) for s in S] + [zc + deck.clear_width_mm / 2])
    v["deck_width_mm"] = deck.clear_width_mm
    v["total_width_mm"] = zmax - zmin
    culprits["total_width_mm"] = (
        f"outermost: {_extreme(S, 2, -1, zmin, zc - deck.clear_width_mm / 2)} / "
        f"{_extreme(S, 2, 1, zmax, zc + deck.clear_width_mm / 2)}")

    # ---- cart envelope above the deck (§8.2.2.2, §8.5) -------------------------------
    top = deck.top_elevation_mm
    half_cart = c["cart_width_mm"] / 2.0
    clearance = math.inf
    # The deck edges bound the path even where no member stands beside it.
    gap_left = gap_right = deck.clear_width_mm / 2.0
    blocking = None
    in_path: list[str] = []
    side = {"left": "", "right": ""}
    for s in S:
        p, h = s.pts, s.h
        on_deck = (p[:, 0] + h[0] > deck.x_start_mm) & (p[:, 0] - h[0] < deck.x_end_mm)
        above = (p[:, 1] + h[1] > top + _TOL) & on_deck
        if not above.any():
            continue
        q = p[above]
        dz = q[:, 2] - zc
        in_band = np.abs(dz) - h[2] < half_cart
        if in_band.any():
            cl = float((q[in_band, 1] - h[1]).min() - top)
            if cl < clearance:
                clearance, blocking = cl, s.member
            if cl < c["cart_height_mm"]:
                in_path.append(s.member)
        low = (q[:, 1] - h[1]) < top + c["cart_height_mm"]
        r = low & (dz - h[2] > 0)
        lft = low & (dz + h[2] < 0)
        if r.any() and float((dz[r] - h[2]).min()) < gap_right:
            gap_right, side["right"] = float((dz[r] - h[2]).min()), s.member
        if lft.any() and float((-dz[lft] - h[2]).min()) < gap_left:
            gap_left, side["left"] = float((-dz[lft] - h[2]).min()), s.member
    v["clearance_above_deck_mm"] = clearance
    v["cart_clear_width_mm"] = gap_left + gap_right
    d["clearance_blocking_member"] = blocking
    v["cart_envelope_ok"] = float(
        clearance >= c["cart_height_mm"] and gap_left + gap_right >= c["cart_width_mm"]
    )
    # Where no member stands beside the path, the deck edge bounds it.
    narrowing = (list(dict.fromkeys(side[k] or "deck edge" for k in ("left", "right")))
                 if gap_left + gap_right < c["cart_width_mm"] else [])
    culprits["cart_envelope_ok"] = "; ".join(
        ([f"in the cart path: {_names(in_path)}"] if in_path else [])
        + ([f"path bounded by: {' / '.join(narrowing)}"] if narrowing else []))

    # ---- clear span box (§8.6) --------------------------------------------------------
    box_h = table_y + c["clear_box_height_mm"]
    intervals = []
    for s in S:
        # Support members lie wholly outside (left_inner, right_inner) by construction, so
        # the x filter skips them; anything else touching the table there (e.g. a tie
        # between the pier bases) blocks the box.
        p, h = s.pts, s.h
        low = (p[:, 1] - h[1] < box_h) & (p[:, 0] + h[0] > left_inner) & (
            p[:, 0] - h[0] < right_inner)
        if low.any():
            xs = p[low, 0]
            intervals.append((float(xs.min() - h[0]), float(xs.max() + h[0]), s.member))
    intervals.sort()
    free, cursor, obstructions = 0.0, left_inner, []
    for a, b, mid in intervals:
        free = max(free, a - cursor)
        cursor = max(cursor, b)
        obstructions.append(mid)
    free = max(free, right_inner - cursor)
    v["clear_box_free_length_mm"] = max(0.0, free)
    d["clear_box_obstructions"] = sorted(set(obstructions))
    culprits["clear_box_free_length_mm"] = "; ".join(
        ([f"in the box: {_names(sorted(set(obstructions)))}"] if obstructions else [])
        + ([culprits["clear_span_mm"]]
           if right_inner - left_inner < c["clear_box_length_mm"] else []))

    # ---- clear opening above mid-span (§8.9) -----------------------------------------
    half_open = c["clear_opening_mm"] / 2.0
    margin = math.inf
    crossing = []
    for s in S:
        p, h = s.pts, s.h
        above = p[:, 1] + h[1] > top + _TOL
        if not above.any():
            continue
        q = p[above]
        gap = np.maximum(np.abs(q[:, 0] - xc) - h[0], np.abs(q[:, 2] - zc) - h[2]) - half_open
        g = float(gap.min())
        margin = min(margin, g)
        if g < 0:
            crossing.append(s.member)
    v["clear_opening_margin_mm"] = margin
    v["clear_opening_ok"] = float(margin >= 0)
    d["clear_opening_crossing"] = crossing
    if crossing:
        culprits["clear_opening_ok"] = f"crossing the opening: {_names(crossing)}"
    d["mid_span_x_mm"] = xc

    # ---- piers vertical (§8.1) --------------------------------------------------------
    # Every leg the bridge stands on counts, whatever its group: members grouped as piers
    # plus any non-bracing member with exactly one end on the table.
    legs = {s.member for s in S
            if s.group not in ("pier_brace", "top_brace", "bottom_brace")
            and int((s.pts[[0, -1], 1] <= table_y + _TOL).sum()) == 1}
    piers = [m for m in bridge.members if m.group == "pier" or m.id in legs]
    incl = [
        inclination_from_vertical_deg(nodes[m.i].xyz, nodes[m.j].xyz) for m in piers
    ]
    v["pier_max_inclination_deg"] = max(incl, default=0.0)
    v["piers_vertical_ok"] = float(
        v["pier_max_inclination_deg"] <= c["pier_max_inclination_deg"]
    )
    tilted = [m.id for m, a in zip(piers, incl, strict=True)
              if a > c["pier_max_inclination_deg"]]
    if tilted:
        culprits["piers_vertical_ok"] = f"inclined: {_names(tilted)}"
    return out
