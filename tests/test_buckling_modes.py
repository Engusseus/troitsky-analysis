"""Describing and exporting global buckling modes (shape, not just one member's name)."""

import csv
import math

import pytest
from typer.testing import CliRunner

from bridgesim.analysis import analyze
from bridgesim.cli import app
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.report import to_markdown
from bridgesim.schema import Bridge, Deck, Member, Node, PlateLoad, Section, Supports
from bridgesim.stability import describe_mode
from bridgesim.viz import buckling_mode_figure


@pytest.fixture(scope="module")
def unbraced():
    """No top bracing: the top chords bow sideways between the portals."""
    return analyze(generate_warren(WarrenParams(top_bracing="none")))


@pytest.fixture(scope="module")
def swaying():
    """Thin piers: the whole bridge sways lengthwise, and that governs F_u,p."""
    p = WarrenParams()
    p.sections["pier"].sticks = 2
    return analyze(generate_warren(p))


def test_describe_names_direction_and_the_nodes_that_move_most() -> None:
    mode = {"T1n": (0.0, 0.0, 1.0), "T1f": (0.0, 0.0, 0.9), "HUB": (0.0, 0.0, 0.67),
            "E": (0.0, 0.0, 0.0)}
    assert describe_mode(mode, "RIB1") == ("sideways (Z) movement, largest at T1n, T1f; "
                                           "most strain energy in RIB1")


def test_describe_names_a_second_direction_only_when_it_matters() -> None:
    """Sums of squares: Z = 1, Y = 0.6^2 = 0.36 >= 1/4 of Z, X = 0.1^2 = 0.01 is not."""
    mode = {"A": (0.1, 0.6, 1.0), "B": (0.0, 0.0, 0.0), "C": (0.0, 0.0, 0.0)}
    assert describe_mode(mode, None).startswith("sideways (Z) and vertical (Y) movement")


def test_describe_counts_the_nodes_when_most_of_the_bridge_moves() -> None:
    mode = {f"N{i}": (1.0, 0.0, 0.0) for i in range(9)} | {"P": (0.0, 0.0, 0.0)}
    assert describe_mode(mode, "d1").startswith(
        "lengthwise (X) movement of most of the bridge (9 of 10 nodes)")


def test_describe_calls_a_mode_local_when_the_joints_hardly_move() -> None:
    """Movements are scaled to the largest anywhere, member midpoints included; if no joint
    moves half as much, a member is bowing between its joints."""
    assert describe_mode({"A": (0, 0.3, 0), "B": (0, 0, 0)}, "RIB1") == (
        "local: RIB1 bows between its joints (they move at most 30% as much as the member)")


def test_single_member_column_mode_is_local(material) -> None:
    """A pinned stick drawn as one member: both its nodes are held, so only the hidden
    midpoint moves (P_cr = pi^2 E I / L^2 = 49.75 N, see test_stability.py)."""
    col = Bridge(
        name="column", nodes=[Node(id="N0", x_mm=0, y_mm=0, z_mm=0),
                              Node(id="N1", x_mm=0, y_mm=115, z_mm=0)],
        members=[Member(id="M0", i="N0", j="N1", section="s")],
        sections=[Section(id="s", b_mm=10, d_mm=2)],
        deck=Deck(x_start_mm=-1, x_end_mm=1, top_elevation_mm=115, clear_width_mm=10),
        supports=Supports(pinned=["N0"], extra_restraints={"N0": ["RY"], "N1": ["DX", "DZ"]}),
        load=PlateLoad(P_ref_N=1.0, plate_length_mm=1e-3, x_center_mm=0.0,
                       deck_support_nodes=["N1"]))
    r = analyze(col, material)
    assert r.buckling.description.startswith("local: M0 bows between its joints")


def test_modes_are_listed_lowest_first_with_shapes(unbraced) -> None:
    b = unbraced.buckling
    assert len(b.lambdas) == len(b.modes_mm) == len(b.descriptions) == 3
    assert b.lambdas == sorted(b.lambdas)
    assert b.mode_mm == b.modes_mm[0] and b.lambda_cr == b.lambdas[0]
    assert b.description.startswith("sideways (Z) movement, largest at ")
    assert all(n in b.description for n in ("T3n", "T3f", "T4n", "T4f"))
    rows = unbraced.buckling_load_table()
    assert [r["mode"] for r in rows] == [1, 2, 3]
    assert rows[0]["F_cr_N"] == pytest.approx(unbraced.Fu_buckling_N, abs=0.05)
    assert rows[0]["shape"] == b.description


def test_mode_table_has_every_node_scaled_to_one(unbraced) -> None:
    rows = unbraced.buckling_mode_table(1)
    assert [r["node"] for r in rows] == [n.id for n in unbraced.bridge.nodes]
    top = max(rows, key=lambda r: r["d_rel"])
    assert 0.5 < top["d_rel"] <= 1.0 and abs(top["dz_rel"]) == pytest.approx(top["d_rel"],
                                                                              abs=0.05)
    assert unbraced.buckling_mode_table(4) == []


def test_global_buckling_is_described_by_its_shape_not_a_member(swaying) -> None:
    """The member with the most energy can be a small rib while the span bows: never call
    the mode "led by" it."""
    assert swaying.governing_mode == "global_buckling"
    assert swaying.governing_detail.startswith(" (lengthwise (X) movement")
    text = to_markdown(swaying)
    assert "mode led by" not in text and not any("mode led by" in w for w in swaying.warnings)
    assert "Global elastic buckling modes" in text


def test_cli_exports_the_mode(tmp_path, unbraced) -> None:
    path = tmp_path / "unbraced.yaml"
    unbraced.bridge.save(path)
    res = CliRunner().invoke(app, ["run", str(path), "--out", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    assert "Global buckling mode 1        : sideways (Z) movement" in res.output
    assert "Next global buckling loads" in res.output
    out = tmp_path / "out"
    with (out / "buckling_mode.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(unbraced.bridge.nodes)
    assert list(rows[0]) == ["node", "x_mm", "y_mm", "z_mm", "dx_rel", "dy_rel", "dz_rel",
                             "d_rel"]
    with (out / "buckling_loads.csv").open(encoding="utf-8") as fh:
        loads = list(csv.DictReader(fh))
    assert [int(r["mode"]) for r in loads] == [1, 2, 3]
    assert (out / "buckling_mode.png").read_bytes()[:4] == b"\x89PNG"


def test_no_mode_means_no_figure_and_no_tables(default_result) -> None:
    r = analyze(Bridge(
        nodes=[Node(id="A", x_mm=0, y_mm=200, z_mm=0), Node(id="B", x_mm=0, y_mm=100, z_mm=0)],
        members=[Member(id="M", i="A", j="B", section="s")],
        sections=[Section(id="s", b_mm=10, d_mm=2)],
        deck=Deck(x_start_mm=-1, x_end_mm=1, top_elevation_mm=100, clear_width_mm=10),
        supports=Supports(pinned=["A"], extra_restraints={"A": ["RX", "RY", "RZ"]}),
        load=PlateLoad(P_ref_N=1.0, plate_length_mm=1e-3, x_center_mm=0.0,
                       deck_support_nodes=["B"])))  # a hanging rod: all in tension
    assert math.isinf(r.Fu_buckling_N)
    assert buckling_mode_figure(r) is None
    assert r.buckling_load_table() == [] and r.buckling_mode_table() == []
    assert buckling_mode_figure(default_result) is not None
