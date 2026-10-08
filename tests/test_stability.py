"""Global elastic buckling (linear eigenvalue) analysis: validation against Euler."""

import math

import pytest

from bridgesim.analysis import analyze
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.materials import load_material
from bridgesim.schema import Bridge, Deck, Member, Node, PlateLoad, Section, Supports

E = 10_000.0  # MPa (placeholder material)
B, D = 10.0, 2.0  # one popsicle stick
L = 115.0
I_MIN = B * D**3 / 12.0  # 6.667 mm^4


def _column(supports: Supports, n_el: int = 8) -> Bridge:
    """Vertical stick from y = 0 to y = L, split into ``n_el`` elements, loaded at the top
    by P_ref = 1 N downward (via a tiny 'plate' over the top node)."""
    nodes = [Node(id=f"N{i}", x_mm=0.0, y_mm=L * i / n_el, z_mm=0.0) for i in range(n_el + 1)]
    members = [Member(id=f"M{i}", i=f"N{i}", j=f"N{i + 1}", section="s") for i in range(n_el)]
    return Bridge(
        name="column", nodes=nodes, members=members,
        sections=[Section(id="s", b_mm=B, d_mm=D)],
        deck=Deck(x_start_mm=-1, x_end_mm=1, top_elevation_mm=L, clear_width_mm=10),
        supports=supports,
        load=PlateLoad(P_ref_N=1.0, plate_length_mm=1e-3, x_center_mm=0.0,
                       deck_support_nodes=[f"N{n_el}"]),
    )


@pytest.fixture(scope="module")
def material():
    return load_material("popsicle_birch")


def test_pinned_column_matches_euler(material):
    """Pinned-pinned stick: P_cr = pi^2 E I / L^2 = pi^2 (10000)(6.667) / 115^2 = 49.75 N.

    Base pinned (DX, DY, DZ) with torsion held; top guided laterally (DX, DZ free in
    rotation). With P_ref = 1 N the critical load factor equals P_cr in newtons.
    """
    col = _column(Supports(pinned=["N0"], extra_restraints={"N0": ["RY"], "N8": ["DX", "DZ"]}))
    r = analyze(col, material)
    euler = math.pi**2 * E * I_MIN / L**2
    assert euler == pytest.approx(49.75, abs=0.01)
    assert r.buckling.lambda_cr == pytest.approx(euler, rel=5e-3)
    assert r.Fu_buckling_N == pytest.approx(euler, rel=5e-3)


def test_cantilever_column_matches_euler(material):
    """Fixed-free stick (flagpole): P_cr = pi^2 E I / (2L)^2 = 12.44 N."""
    col = _column(Supports(pinned=["N0"], extra_restraints={"N0": ["RX", "RY", "RZ"]}))
    r = analyze(col, material)
    euler = math.pi**2 * E * I_MIN / (2 * L) ** 2
    assert r.buckling.lambda_cr == pytest.approx(euler, rel=5e-3)


def test_hanging_rod_has_no_buckling_mode(material):
    """A rod hanging from a support is in tension everywhere: no buckling (F_cr = inf)."""
    nodes = [Node(id="A", x_mm=0, y_mm=200, z_mm=0), Node(id="B", x_mm=0, y_mm=100, z_mm=0)]
    rod = Bridge(
        nodes=nodes, members=[Member(id="M", i="A", j="B", section="s")],
        sections=[Section(id="s", b_mm=B, d_mm=D)],
        deck=Deck(x_start_mm=-1, x_end_mm=1, top_elevation_mm=100, clear_width_mm=10),
        supports=Supports(pinned=["A"], extra_restraints={"A": ["RX", "RY", "RZ"]}),
        load=PlateLoad(P_ref_N=1.0, plate_length_mm=1e-3, x_center_mm=0.0,
                       deck_support_nodes=["B"]),
    )
    r = analyze(rod, material)
    assert math.isinf(r.Fu_buckling_N)
    assert r.governing_mode != "global_buckling"


def test_default_bridge_buckling_is_finite_and_not_governing(material):
    """The default design is stiff enough that strength governs before global buckling."""
    r = analyze(generate_warren(), material)
    assert 1.0 < r.buckling.lambda_cr < math.inf
    assert r.Fu_buckling_N > r.Fu_strength_N
    assert r.buckling.key_member is not None
    assert max(r.buckling.member_energy.values()) == pytest.approx(1.0)


def test_removing_top_bracing_lowers_buckling_load(material):
    """Without top bracing the top chord can buckle sideways, so F_cr must drop."""
    braced = analyze(generate_warren(), material)
    unbraced = analyze(generate_warren(WarrenParams(top_bracing="none")), material)
    assert unbraced.Fu_buckling_N < braced.Fu_buckling_N


def test_slender_piers_govern_by_sway(material):
    """Thin legs (2 sticks) sway lengthwise long before the members fail: the predicted load
    must then be governed by global buckling, not hidden behind member checks."""
    p = WarrenParams()
    p.sections["pier"].sticks = 2
    r = analyze(generate_warren(p), material)
    assert r.governing_mode == "global_buckling"
    assert r.Fu_pred_N == pytest.approx(r.Fu_buckling_N)


def test_single_member_column_is_split_for_buckling(material):
    """A column drawn as ONE member is split into two elements internally, so the global
    eigen-solver is within ~1 % of Euler (one cubic element alone gives 12EI/L^2, +22 %)."""
    col = _column(Supports(pinned=["N0"], extra_restraints={"N0": ["RY"], "N1": ["DX", "DZ"]}),
                  n_el=1)
    r = analyze(col, material)
    euler = math.pi**2 * E * I_MIN / L**2
    assert r.buckling.lambda_cr == pytest.approx(euler, rel=0.01)
    assert set(r.displacements_mm) == {"N0", "N1"}  # interior nodes stay internal
    assert set(r.buckling.mode_mm) == {"N0", "N1"}


def test_crossing_braces_are_not_joined_by_the_split(material):
    """X-braces cross at their midpoints; the interior split points must avoid that."""
    r = analyze(generate_warren(), material)
    for m in r.bridge.members:
        assert len(r.fe_model.members[m.id].sub_members) == 2, m.id
