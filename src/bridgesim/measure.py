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
_TOL = 0.5  # mm

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


@dataclass
class Measurements:
    values: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]


def _samples(bridge: Bridge, material: Material, table_y: float) -> list[_Samples]:
    nodes = bridge.node_map()
    secs = bridge.section_map()
    out = []
    for m in bridge.members:
        a, b = np.array(nodes[m.i].xyz), np.array(nodes[m.j].xyz)
        L = float(np.linalg.norm(b - a))
        k = max(2, int(math.ceil(L / SAMPLE_MM)) + 1)
        t = np.linspace(0.0, 1.0, k)[:, None]
        bm, dm = secs[m.section].dims(material.stick)
        touches = min(a[1], b[1]) <= table_y + _TOL
        out.append(_Samples(m.id, m.group, a + t * (b - a),
                            half_extents(tuple(a), tuple(b), bm, dm), touches))
    return out


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

    xmin = min([float((s.pts[:, 0] - s.h[0]).min()) for s in S] + [deck.x_start_mm])
    xmax = max([float((s.pts[:, 0] + s.h[0]).max()) for s in S] + [deck.x_end_mm])
    v["deck_length_mm"] = deck.length_mm
    v["total_length_mm"] = xmax - xmin

    # ---- heights (§8.2.2) -------------------------------------------------------------
    # Nothing can sit below the table: stick ends touching it are cut flush.
    ymin = max(table_y, min([float((s.pts[:, 1] - s.h[1]).min()) for s in S]
                            + [deck.top_elevation_mm]))
    ymax = max([float((s.pts[:, 1] + s.h[1]).max()) for s in S] + [deck.top_elevation_mm])
    v["deck_height_mm"] = deck.top_elevation_mm - table_y
    v["total_height_mm"] = ymax - ymin

    # ---- widths (§8.2.3) --------------------------------------------------------------
    zmin = min([float((s.pts[:, 2] - s.h[2]).min()) for s in S] + [zc - deck.clear_width_mm / 2])
    zmax = max([float((s.pts[:, 2] + s.h[2]).max()) for s in S] + [zc + deck.clear_width_mm / 2])
    v["deck_width_mm"] = deck.clear_width_mm
    v["total_width_mm"] = zmax - zmin

    # ---- cart envelope above the deck (§8.2.2.2, §8.5) -------------------------------
    top = deck.top_elevation_mm
    half_cart = c["cart_width_mm"] / 2.0
    clearance = math.inf
    gap_left = gap_right = math.inf
    blocking = None
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
        low = (q[:, 1] - h[1]) < top + c["cart_height_mm"]
        r = low & (dz - h[2] > 0)
        lft = low & (dz + h[2] < 0)
        if r.any():
            gap_right = min(gap_right, float((dz[r] - h[2]).min()))
        if lft.any():
            gap_left = min(gap_left, float((-dz[lft] - h[2]).min()))
    v["clearance_above_deck_mm"] = clearance
    v["cart_clear_width_mm"] = gap_left + gap_right
    d["clearance_blocking_member"] = blocking
    v["cart_envelope_ok"] = float(
        clearance >= c["cart_height_mm"] and gap_left + gap_right >= c["cart_width_mm"]
    )

    # ---- clear span box (§8.6) --------------------------------------------------------
    box_h = table_y + c["clear_box_height_mm"]
    intervals = []
    for s in S:
        if s.touches_table:
            continue
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
    d["mid_span_x_mm"] = xc

    # ---- piers vertical (§8.1) --------------------------------------------------------
    piers = [m for m in bridge.members if m.group == "pier"]
    if not piers:
        ids = {s.member for s in supporting if s.group not in ("pier_brace",)}
        piers = [m for m in bridge.members if m.id in ids]
    incl = [
        inclination_from_vertical_deg(nodes[m.i].xyz, nodes[m.j].xyz) for m in piers
    ]
    v["pier_max_inclination_deg"] = max(incl, default=0.0)
    v["piers_vertical_ok"] = float(
        v["pier_max_inclination_deg"] <= c["pier_max_inclination_deg"]
    )
    return out
