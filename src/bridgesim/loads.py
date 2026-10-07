"""Crusher-plate load -> nodal loads on the deck support nodes (floor-beam centres).

The plate (rulebook §12.5: 90 mm wide x 200 mm long) applies a uniform line load
w = P_ref / L_plate over [x_c - L_plate/2, x_c + L_plate/2] along X. The deck is treated
as simply supported strips spanning between adjacent support stations x_i < x_{i+1}
(spacing p_i = x_{i+1} - x_i). For each strip overlapping the plate on [a, b]:

    R = w (b - a),   xbar = (a + b) / 2
    F_i     += R (x_{i+1} - xbar) / p_i
    F_{i+1} += R (xbar - x_i) / p_i

(lever rule for a resultant on a simply supported span). Forces sum to P_ref exactly and
are symmetric for a symmetric layout. Load outside the outermost stations (if any) is
assigned to the nearest end station so the total is still P_ref.
"""

from __future__ import annotations

from dataclasses import dataclass

from bridgesim.schema import Bridge


@dataclass(frozen=True)
class PlatePlacement:
    x_center_mm: float
    x_start_mm: float
    x_end_mm: float
    z_center_mm: float
    width_mm: float
    deck_top_mm: float


def mid_span_x(bridge: Bridge) -> float:
    """Mid-span = midpoint between the mean X of the two support groups."""
    nodes = bridge.node_map()
    xs = sorted(nodes[n].x_mm for n in bridge.support_nodes())
    if not xs:
        return 0.5 * (bridge.deck.x_start_mm + bridge.deck.x_end_mm)
    return 0.5 * (xs[0] + xs[-1])


def plate_placement(bridge: Bridge) -> PlatePlacement:
    xc = bridge.load.x_center_mm if bridge.load.x_center_mm is not None else mid_span_x(bridge)
    half = bridge.load.plate_length_mm / 2.0
    return PlatePlacement(
        x_center_mm=xc,
        x_start_mm=xc - half,
        x_end_mm=xc + half,
        z_center_mm=bridge.deck.z_center_mm,
        width_mm=bridge.load.plate_width_mm,
        deck_top_mm=bridge.deck.top_elevation_mm,
    )


def distribute_plate_load(
    stations: list[tuple[str, float]], P_N: float, x_start_mm: float, x_end_mm: float
) -> dict[str, float]:
    """Split a uniform plate load over [x_start, x_end] onto ``stations`` (id, x).

    Returns downward force magnitudes in N keyed by station id (see module docstring).
    """
    if x_end_mm <= x_start_mm:
        raise ValueError("Plate length must be positive")
    st = sorted(stations, key=lambda s: s[1])
    if not st:
        raise ValueError("No deck support stations to carry the plate load")
    w = P_N / (x_end_mm - x_start_mm)
    forces = {sid: 0.0 for sid, _ in st}

    # Portions of the plate beyond the outermost stations go to the end station.
    if x_start_mm < st[0][1]:
        forces[st[0][0]] += w * (min(x_end_mm, st[0][1]) - x_start_mm)
    if x_end_mm > st[-1][1]:
        forces[st[-1][0]] += w * (x_end_mm - max(x_start_mm, st[-1][1]))

    for (id_i, x_i), (id_j, x_j) in zip(st[:-1], st[1:], strict=True):
        a, b = max(x_i, x_start_mm), min(x_j, x_end_mm)
        if b <= a or x_j <= x_i:
            continue
        R = w * (b - a)
        xbar = 0.5 * (a + b)
        p = x_j - x_i
        forces[id_i] += R * (x_j - xbar) / p
        forces[id_j] += R * (xbar - x_i) / p
    return {k: v for k, v in forces.items() if v != 0.0}


def plate_nodal_loads(bridge: Bridge, P_N: float | None = None) -> dict[str, float]:
    """Downward nodal forces (N, positive = down) for the bridge's crusher plate."""
    P = bridge.load.P_ref_N if P_N is None else P_N
    nodes = bridge.node_map()
    stations = [(nid, nodes[nid].x_mm) for nid in bridge.load.deck_support_nodes]
    pl = plate_placement(bridge)
    return distribute_plate_load(stations, P, pl.x_start_mm, pl.x_end_mm)
