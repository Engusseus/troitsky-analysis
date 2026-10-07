"""Crusher-plate load distribution onto deck-support stations (lever rule)."""

from __future__ import annotations

import pytest

from bridgesim.loads import distribute_plate_load, mid_span_x, plate_nodal_loads, plate_placement
from bridgesim.schema import Bridge

P = 1000.0
PANEL = 143.75
WARREN_STATIONS = [(f"C{i}", i * PANEL) for i in range(9)]  # default Warren: 1150 / 8


@pytest.mark.parametrize(
    ("x_start", "x_end"),
    [
        (475.0, 675.0),  # centred on the default mid-span
        (10.0, 60.0),  # inside one strip
        (400.0, 450.0),  # straddles a station
        (100.0, 900.0),  # spans many strips
        (-80.0, 50.0),  # hangs off the left end
        (1100.0, 1300.0),  # hangs off the right end
        (-200.0, 1400.0),  # wider than all stations
        (-300.0, -100.0),  # entirely left of the stations
        (1200.0, 1250.0),  # entirely right of the stations
        (431.25, 431.25 + 1e-3),  # tiny plate starting on a station
    ],
)
def test_forces_sum_to_P(x_start: float, x_end: float) -> None:
    """Equilibrium: sum of station forces = P for any plate placement (lever rule
    redistributes each strip resultant without loss; overhangs go to the end station).
    """
    f = distribute_plate_load(WARREN_STATIONS, P, x_start, x_end)
    assert sum(f.values()) == pytest.approx(P, rel=1e-12)
    assert all(v > 0 for v in f.values())


def test_default_warren_hand_case() -> None:
    """Stations every 143.75 mm, plate [475, 675], P = 1000 N, w = P / 200 = 5 N/mm.

    Strip C3-C4 (431.25..575) carries [475, 575]: R = 500 N at xbar = 525:
        F_C3 += 500 * (575 - 525) / 143.75 = 173.913 N
        F_C4 += 500 * (525 - 431.25) / 143.75 = 326.087 N
    Strip C4-C5 is the mirror image, so
        C3 = C5 = 173.913 N,  C4 = 2 * 326.087 = 652.174 N  (sum 1000 N).
    """
    f = distribute_plate_load(WARREN_STATIONS, P, 475.0, 675.0)
    assert set(f) == {"C3", "C4", "C5"}
    assert f["C3"] == pytest.approx(173.913, abs=1e-3)
    assert f["C5"] == pytest.approx(173.913, abs=1e-3)
    assert f["C4"] == pytest.approx(652.174, abs=1e-3)
    assert f["C3"] == pytest.approx(4000.0 / 23.0, rel=1e-12)  # exact: 500 * 50 / 143.75


def test_off_centre_inside_one_strip() -> None:
    """Plate [10, 60] inside strip 0..100: R = P at xbar = 35, so
    F_0 = P (100 - 35) / 100 = 650 N, F_1 = P (35 - 0) / 100 = 350 N.
    """
    f = distribute_plate_load([("a", 0.0), ("b", 100.0), ("c", 200.0)], P, 10.0, 60.0)
    assert f == pytest.approx({"a": 650.0, "b": 350.0})


def test_plate_beyond_outer_stations() -> None:
    """Stations 0, 100, 200; plate [-50, 250], P = 300 N so w = 1 N/mm.

        overhang [-50, 0]    -> 50 N to station 0
        strip [0, 100]: 100 N at 50 -> 50 / 50
        strip [100, 200]: 100 N at 150 -> 50 / 50
        overhang [200, 250]  -> 50 N to station 200
        => 100, 100, 100 N
    """
    f = distribute_plate_load([("a", 0.0), ("b", 100.0), ("c", 200.0)], 300.0, -50.0, 250.0)
    assert f == pytest.approx({"a": 100.0, "b": 100.0, "c": 100.0})


def test_plate_entirely_outside() -> None:
    """A plate wholly beyond an end station puts all of P on that end station."""
    st = [("a", 0.0), ("b", 100.0)]
    assert distribute_plate_load(st, P, 150.0, 250.0) == pytest.approx({"b": P})
    assert distribute_plate_load(st, P, -300.0, -100.0) == pytest.approx({"a": P})


def test_single_station_takes_everything() -> None:
    """One station: both plate halves are overhangs, so it carries P."""
    assert distribute_plate_load([("only", 5.0)], P, 0.0, 200.0) == pytest.approx({"only": P})


def test_station_order_does_not_matter() -> None:
    """Stations are sorted by x internally."""
    shuffled = [WARREN_STATIONS[i] for i in (4, 0, 8, 2, 6, 1, 3, 7, 5)]
    assert distribute_plate_load(shuffled, P, 475.0, 675.0) == pytest.approx(
        distribute_plate_load(WARREN_STATIONS, P, 475.0, 675.0))


@pytest.mark.parametrize("centre_station", [2, 3, 4, 5])
def test_symmetric_about_a_station(centre_station: int) -> None:
    """A plate centred on station k of a uniform layout loads k +- j equally.

    E.g. centred on C3 (431.25), plate [331.25, 531.25]: each strip carries 500 N at
    50 mm from C3, so C2 = C4 = 500 * 50 / 143.75 = 173.913 N and C3 = 652.174 N.
    """
    xc = centre_station * PANEL
    f = distribute_plate_load(WARREN_STATIONS, P, xc - 100.0, xc + 100.0)
    left, right = f"C{centre_station - 1}", f"C{centre_station + 1}"
    assert f[left] == pytest.approx(f[right], rel=1e-12)
    assert f[left] == pytest.approx(4000.0 / 23.0, rel=1e-12)
    assert f[f"C{centre_station}"] == pytest.approx(P - 8000.0 / 23.0, rel=1e-12)


def test_symmetric_layout_mirror() -> None:
    """Mirroring both the stations and the plate about mid-span mirrors the forces."""
    span = 8 * PANEL
    f = distribute_plate_load(WARREN_STATIONS, P, 300.0, 520.0)
    g = distribute_plate_load(WARREN_STATIONS, P, span - 520.0, span - 300.0)
    for i in range(9):
        assert f.get(f"C{i}", 0.0) == pytest.approx(g.get(f"C{8 - i}", 0.0), abs=1e-9)


def test_invalid_inputs() -> None:
    """Plate length must be positive and there must be at least one station."""
    with pytest.raises(ValueError):
        distribute_plate_load(WARREN_STATIONS, P, 500.0, 500.0)
    with pytest.raises(ValueError):
        distribute_plate_load(WARREN_STATIONS, P, 600.0, 500.0)
    with pytest.raises(ValueError):
        distribute_plate_load([], P, 0.0, 200.0)


def test_plate_placement_default_bridge(default_bridge: Bridge) -> None:
    """Default Warren: supports at x = 0 and 1150, so mid-span x = 575 and the 200 mm
    plate covers [475, 675] at the deck top (200 mm), 90 mm wide, on the centreline.
    """
    assert mid_span_x(default_bridge) == pytest.approx(575.0)
    pl = plate_placement(default_bridge)
    assert (pl.x_center_mm, pl.x_start_mm, pl.x_end_mm) == pytest.approx((575.0, 475.0, 675.0))
    assert pl.deck_top_mm == pytest.approx(200.0)
    assert pl.width_mm == pytest.approx(90.0)
    assert pl.z_center_mm == pytest.approx(0.0)


def test_plate_nodal_loads_default_bridge(default_bridge: Bridge) -> None:
    """Default bridge at P_ref = 1000 N: C3 = C5 = 173.913 N, C4 = 652.174 N; scaling
    P_N scales every force (P_N = 2500 N -> C4 = 1630.435 N).
    """
    f = plate_nodal_loads(default_bridge)
    assert f == pytest.approx({"C3": 4000.0 / 23.0, "C4": 15000.0 / 23.0, "C5": 4000.0 / 23.0})
    g = plate_nodal_loads(default_bridge, 2500.0)
    assert g == pytest.approx({k: 2.5 * v for k, v in f.items()})
    assert g["C4"] == pytest.approx(1630.435, abs=1e-3)


def test_plate_nodal_loads_explicit_centre(default_bridge: Bridge) -> None:
    """x_center_mm overrides mid-span: centred on C3 (431.25) the plate loads C2, C3, C4
    with 173.913 / 652.174 / 173.913 N.
    """
    data = default_bridge.model_dump()
    data["load"]["x_center_mm"] = 3 * PANEL
    moved = Bridge.model_validate(data)
    f = plate_nodal_loads(moved)
    assert f == pytest.approx({"C2": 4000.0 / 23.0, "C3": 15000.0 / 23.0, "C4": 4000.0 / 23.0})
