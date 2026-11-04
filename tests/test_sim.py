"""Crush simulation: element validation against textbook results, kernels, CAD import,
rules from CAD, the end-to-end crush of a small synthetic bridge and the AI interface."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from bridgesim.cad import load_cad
from bridgesim.cad.synthetic import box, simple_beam_bridge, write_stl
from bridgesim.schema import Material
from bridgesim.sim import lab
from bridgesim.sim import solver as S
from bridgesim.sim.materials import Ply, SimParams, laminate, parse_layup
from bridgesim.sim.shell import dkt_B

ROOT = Path(__file__).resolve().parents[1]
MAT = Material.from_yaml(ROOT / "materials" / "popsicle_birch.yaml")
SP = SimParams.load()


# ------------------------------------------------------------------ materials


def test_layup_parsing():
    assert parse_layup("Top Chord (10-ply UD 20.9x47.5) - Far").angles == [0.0] * 10
    assert sorted(set(parse_layup("Deck Plate (2-ply 0-90) - R").angles)) == [0.0, 90.0]
    assert sorted(set(parse_layup("Hub Cheek (4-ply +-45)").angles)) == [-45.0, 45.0]
    assert parse_layup("Floss Tie A (waxed floss)").kind == "floss"
    assert parse_layup("Floss Toggle (round toothpick dia 2.2 x 25)").kind == "toothpick"
    assert parse_layup("Deck Cleat (block 33x19x24.5)").kind == "block"
    assert parse_layup("Hub Spreader (6-ply UD along the span)").grain == "span"
    assert parse_layup("Body 12") is None


def test_classical_lamination_theory():
    ply = Ply.from_material(MAT, SP)
    ud = laminate(ply, [0.0] * 4, 8.0)
    assert ud.E_x == pytest.approx(ply.E_L, rel=1e-9)  # uniaxial modulus of UD = E_L
    assert ud.f_t == pytest.approx(ply.f_t, rel=1e-6)
    cross = laminate(ply, [0.0, 90.0, 90.0, 0.0], 8.0)
    assert cross.E_x == pytest.approx(0.5 * (ply.E_L + ply.E_T), rel=0.02)
    pm45 = laminate(ply, [45.0, -45.0, -45.0, 45.0], 8.0)
    assert pm45.G_xy > 3 * ud.G_xy  # +-45 plies resist in-plane shear
    assert pm45.E_x < 0.4 * ud.E_x


# -------------------------------------------------------------------- elements


def test_dkt_patch_test():
    """Constant curvature fields are reproduced exactly at any point of the element."""
    xy = np.array([[0.3, 0.1], [10.2, 1.5], [3.1, 7.7]])

    def dofs(f):
        out = []
        for x, y in xy:
            w, wx, wy = f(x, y)
            out += [w, wy, -wx]
        return np.array(out)

    for f, want in ((lambda x, y: (0.5 * x * x, x, 0), [-1, 0, 0]),
                    (lambda x, y: (0.5 * y * y, 0, y), [0, -1, 0]),
                    (lambda x, y: (x * y, y, x), [0, 0, -2]),
                    (lambda x, y: (1 + 2 * x + 3 * y, 2, 3), [0, 0, 0])):
        for s, e in ((1 / 3, 1 / 3), (0.2, 0.6), (0.5, 0.0)):
            B, _ = dkt_B(xy, s, e)
            assert B @ dofs(f) == pytest.approx(want, abs=1e-9)


def test_shell_element_has_only_rigid_body_zero_modes():
    """6 zero-energy modes (rigid body) and no more: an element whose nodes can spin
    about its normal for free lets glued plates drift (a creep that depends on the test
    speed)."""
    from bridgesim.sim.materials import Ply
    from bridgesim.sim.shell import element_data

    ply = Ply.from_material(MAT, SP)
    rng = np.random.default_rng(3)
    for _ in range(5):
        xy = rng.uniform(-10, 10, (3, 2))
        ed = element_data(xy, [0.0, 90.0], 3.6, ply, 1.0, 1.0, (3.0, 0.3, 1.0),
                          ply.E_L * 3.6)
        ev = np.linalg.eigvalsh(ed["K"])
        assert (ev < 1e-8 * ev.max()).sum() == 6


def test_cantilever_small_deflection():
    v, _u, _ = lab.cantilever(0.01)
    assert v == pytest.approx(0.01 / 3, rel=0.01)  # P L^3 / 3 E I


def test_cantilever_elastica():
    """Large deflection: Bisshopp & Drucker (1945), P L^2 / E I = 1."""
    v, u, _ = lab.cantilever(1.0)
    assert v == pytest.approx(0.3017, rel=0.01)
    assert u == pytest.approx(0.0564, rel=0.03)


def test_euler_buckling_of_a_stick():
    r = lab.euler_column()
    pcr = math.pi**2 * 10000 * 10 * 2**3 / 12 / 115**2
    assert 0.9 * pcr < r.peak < 1.02 * pcr


def test_three_point_bending():
    r = lab.three_point_bending(10000, 1e9, 1e9, rate=20)
    I = 10 * 2**3 / 12
    assert r.stiffness == pytest.approx(48 * 10000 * I / 100**3, rel=0.03)
    # with real strengths: wood yields in compression first, so the modulus of rupture is
    # above the compressive strength and below the tensile one
    r2 = lab.three_point_bending(10000, 40, 30)
    mor = r2.peak * 100 / 4 / (10 * 4 / 6)
    assert 30 < mor < 55


def test_lap_shear_reproduces_tau_g():
    """With the calibrated peak/average factor, the simulated 20 x 10 mm single-lap test
    gives the material's tau_g (what the bridge model assumes for every joint), whether
    its glue points hang on 9 nodes per side or on 1, 2, 3 or 5 (as in a coarse bridge
    mesh): the factor the bridge model uses for that node count gives tau_g within 5 %,
    so a joint's strength does not jump when the mesh changes."""
    import math

    P = SP["glue.tau_peak_over_avg"]
    assert lab.lap_shear(2.0 * P).peak / 200 == pytest.approx(2.0, rel=0.02)
    for n_ov, one, nodes in ((2, True, 1), (1, False, 2), (2, False, 3), (4, False, 5)):
        f = 1.0 + (P - 1.0) * (1.0 - math.exp(-2.2 * (nodes - 1)))
        r = lab.lap_shear(2.0 * f, n_ov=n_ov, one_node=one)
        assert r.peak / 200 == pytest.approx(2.0, rel=0.05), nodes


def test_energy_is_conserved_without_damping():
    """Free vibration of the synthetic bridge (beams, layered shells, glue) without any
    damping: kinetic + strain energy stays constant while energy goes back and forth
    between them. Strain energy = u.F(u) / 2 (small amplitudes); velocities at whole steps
    (mean of the half steps of the central-difference scheme)."""
    from bridgesim.sim.crush import stable_dt
    from bridgesim.sim.idealize import build_model

    m = build_model(simple_beam_bridge(), MAT, SP, seed=None)
    m.table_y, m.plate_node, m.q_att = -1e6, -1, m.q_att[:0]  # free-floating
    s = S.Settings(dt=1e-6, beta_frac=0.0)
    s.dt, m.mass, m.rot_inertia, _ = stable_dt(m, s, 5e-6, 0.0)
    st = S.initial_state(m, 1e12)
    x = m.X[:, 0]
    st.v[:, 1] = 2.0 * np.sin(np.pi * (x - x.min()) / np.ptp(x))  # a smooth bending motion
    st.v -= (m.mass[:, None] * st.v).sum(0) / m.mass.sum()
    F, Mo = np.zeros((m.n_nodes, 3)), np.zeros((m.n_nodes, 3))
    E, U_max = [], 0.0
    for k in range(20000):
        v0, w0 = st.v.copy(), st.w.copy()
        S.step(m, st, s, F, Mo, grav_scale=0.0, alpha=0.0)
        if k % 1000 == 0:
            S.internal_and_contact(m, st, s, F, Mo, update=False)
            U = 0.5 * float((F * (st.x - m.X)).sum() + (Mo * S.logvec(st.R)).sum())
            va, wa = 0.5 * (v0 + st.v), 0.5 * (w0 + st.w)
            KE = 0.5 * float((m.mass * (va**2).sum(1)).sum()
                             + (m.rot_inertia * (wa**2).sum(1)).sum())
            E.append(KE + U)
            U_max = max(U_max, U)
    E = np.array(E)
    assert U_max > 0.05 * E.mean()  # energy really changes form
    assert np.ptp(E) < 0.005 * E.mean()


def test_compiled_kernels_match_numpy():
    from bridgesim.sim.idealize import build_model

    pytest.importorskip("numba")
    m = build_model(simple_beam_bridge(), MAT, SP, seed=0)
    s = S.Settings(dt=3e-6)
    rng = np.random.default_rng(1)
    st = S.initial_state(m, 1e4)
    st.x += rng.normal(0, 0.01, st.x.shape)
    st.R = S.expmap(rng.normal(0, 0.002, (m.n_nodes, 3)))
    st.v = rng.normal(0, 50, st.x.shape)
    st.w = rng.normal(0, 1, st.x.shape)
    st2 = S.initial_state(m, 1e4)
    for k in ("x", "R", "v", "w"):
        setattr(st2, k, getattr(st, k).copy())
    m.table_y += 0.5  # some platform contacts active
    F, Mo = np.zeros((m.n_nodes, 3)), np.zeros((m.n_nodes, 3))
    F2, Mo2 = F.copy(), Mo.copy()
    S.internal_and_contact(m, st, s, F, Mo)
    S.internal_and_contact_numpy(m, st2, s, F2, Mo2)
    assert np.abs(F - F2).max() <= 1e-9 * max(1.0, np.abs(F2).max())
    assert np.abs(Mo - Mo2).max() <= 1e-9 * max(1.0, np.abs(Mo2).max())
    assert np.abs(st.b_gam).max() > 0  # Timoshenko shear state, same in both
    np.testing.assert_allclose(st.b_gam, st2.b_gam, rtol=1e-8, atol=1e-14)
    assert (m.params["p_lift"] > 0).any() and m.params["seating_mm"] > 0


def test_compiled_shell_damage_matches_numpy():
    """Large strains: the layered shell damage histories evolve identically."""
    from bridgesim.sim.idealize import build_model

    pytest.importorskip("numba")
    m = build_model(simple_beam_bridge(), MAT, SP, seed=0)
    s = S.Settings(dt=3e-6)
    rng = np.random.default_rng(2)
    st = S.initial_state(m, 1e4)
    st2 = S.initial_state(m, 1e4)
    F, Mo = np.zeros((m.n_nodes, 3)), np.zeros((m.n_nodes, 3))
    F2, Mo2 = F.copy(), Mo.copy()
    for _ in range(3):
        st.x = m.X + rng.normal(0, 0.15, st.x.shape)
        st.R = S.expmap(rng.normal(0, 0.01, (m.n_nodes, 3)))
        st2.x, st2.R = st.x.copy(), st.R.copy()
        S.internal_and_contact(m, st, s, F, Mo)
        S.internal_and_contact_numpy(m, st2, s, F2, Mo2)
    assert st.s_hist.max() > 1.0  # some points are past their strength
    assert (st.s_dmg > 0).any()
    np.testing.assert_allclose(st.s_hist, st2.s_hist, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(st.s_dmg, st2.s_dmg, rtol=1e-8, atol=1e-10)
    assert (st.s_gone == st2.s_gone).all()
    assert np.abs(F - F2).max() <= 1e-8 * max(1.0, np.abs(F2).max())
    assert np.abs(Mo - Mo2).max() <= 1e-8 * max(1.0, np.abs(Mo2).max())


def test_glue_line_crushing_densifies():
    """Wood pressed at a glue line crushes at f_c (permanent set), hardens as it compacts
    and locks up once crushed by the densification strain of its depth: a block cannot
    push through a 4 mm plate. Compiled kernel = NumPy reference."""
    from bridgesim.sim.lab import blank_model

    pytest.importorskip("numba")
    out = {}
    for impl in (S.internal_and_contact, S.internal_and_contact_numpy):
        m = blank_model(np.zeros((2, 3)))
        m.mass[:], m.rot_inertia[:] = 1.0, 1.0
        m.a_node, m.a_r0 = np.array([0, 1]), np.zeros((2, 3))
        m.c_a, m.c_b, m.c_area = np.array([0]), np.array([1]), np.array([1.0])
        m.c_n0, m.c_Kn, m.c_Ks = np.array([[0.0, 1.0, 0.0]]), np.array([150.0]), np.array([60.0])
        m.c_sig, m.c_tau = np.array([1.0]), np.array([2.0])
        m.c_GIc, m.c_GIIc, m.c_mu = np.array([0.25]), np.array([0.6]), np.array([0.4])
        m.c_patch, m.c_breakable = np.array([0]), np.array([True])
        m.params.update(glue_fc=np.array([5.0]), glue_cdep=np.array([4.0]),
                        densification_strain=0.45, densified_E=500.0)
        st = S.initial_state(m, 1e12)
        F, Mo = np.zeros((2, 3)), np.zeros((2, 3))
        rec = []
        for push in np.linspace(0, 3.0, 301):
            st.x[1, 1] = -push
            impl(m, st, S.Settings(dt=1e-6, beta_frac=0.0), F, Mo)
            rec.append((abs(F[1, 1]), -st.c_crush[0]))
        out[impl.__name__] = np.array(rec)
    a, b = out.values()
    np.testing.assert_allclose(a, b, atol=1e-9)
    stress, crush = a[:, 0], a[:, 1]
    assert stress[10] == pytest.approx(5.0 * (1 + crush[10] / 4), rel=0.02)  # crushing
    assert 5.0 < stress[100] < 7.0  # 1 mm: hardening as it compacts
    assert stress[-1] > 50 and crush[-1] < 2.6  # densified: no pushing through


def test_shell_fracture_energy_is_mesh_independent():
    """A plate strip pulled apart: same strength and same work of fracture (G_f W t) with
    5 mm and 2.5 mm elements (crack-band regularised shell damage)."""
    pytest.importorskip("numba")
    W, t = 10.0, 2.0
    ply = __import__("bridgesim.sim.materials", fromlist=["Ply"]).Ply.from_material(MAT, SP)
    for h in (5.0, 2.5):
        r = lab.shell_strip_tension(h, W=W, t=t)
        work = float(np.trapezoid(r.force, r.disp))
        assert r.peak == pytest.approx(0.95 * ply.f_t * W * t, rel=0.04), h
        assert work == pytest.approx(SP["wood.G_f_N_per_mm"] * W * t, rel=0.1), h
        assert r.force[-1] < 0.02 * r.peak  # completely broken


def test_self_contact_stops_a_falling_member():
    """A stick dropped across another (not touching at the start) comes to rest on it,
    at the sum of their contact radii, instead of passing through."""
    from bridgesim.sim import contact
    from bridgesim.sim.crush import stable_dt
    from bridgesim.sim.model import BodyInfo

    pytest.importorskip("numba")
    n = 9
    s_ = np.linspace(-100, 100, n)
    XA = np.c_[s_, np.zeros(n), np.zeros(n)]
    XB = np.c_[np.zeros(n), np.full(n, 15.0), s_]  # crossing above, 3.7 mm clear
    m = lab.blank_model(np.vstack([XA, XB]))
    spec = lab.BeamSpec(b=10.0, d=10.0, E=10000.0)
    lab.add_beam_line(m, list(range(n)), spec, up=(0, 1, 0))
    lab.add_beam_line(m, list(range(n, 2 * n)), spec, up=(0, 1, 0))
    m.b_body[n - 1:] = 1
    m.node_body[n:] = 1
    m.bodies = [BodyInfo(nm, "bar", "", "", 0.0, 0.0, section_mm=(10.0, 10.0))
                for nm in ("A", "B")]
    s = S.Settings(dt=1e-6)
    s.dt, m.mass, m.rot_inertia, _ = stable_dt(m, s, 5e-6, 0.0)
    st = S.initial_state(m, 1e12)
    st.sc = contact.setup(m, m.X, m.mass, s.dt)
    assert st.sc.n_active == 0
    st.v[n:, 1] = -300.0  # B thrown down onto A
    F, Mo = np.zeros((m.n_nodes, 3)), np.zeros((m.n_nodes, 3))
    lowest = 1e9
    for _ in range(int(0.1 / s.dt)):
        S.step(m, st, s, F, Mo, grav_scale=1.0, alpha=20.0)
        st.x[[0, n - 1]] = m.X[[0, n - 1]]  # A held at its ends
        st.v[[0, n - 1]] = 0.0
        lowest = min(lowest, st.x[n + n // 2, 1])
    reach = 2 * math.sqrt(100.0 / math.pi)  # two capsules of the stick's area
    assert lowest > reach - 1.5  # it did not go through
    assert st.x[n + n // 2, 1] == pytest.approx(reach + st.x[n // 2, 1], abs=1.0)


# --------------------------------------------------------------------- CAD + rules


def test_synthetic_box_is_a_closed_solid():
    b = box("b", (0, 0, 0), (10, 20, 30))
    assert b.mesh.volume == pytest.approx(6000)
    assert b.mesh.centroid == pytest.approx([5, 10, 15])


def test_stl_import_splits_bodies(tmp_path):
    bodies = simple_beam_bridge()
    write_stl(bodies, tmp_path / "b.stl")
    back = load_cad(tmp_path / "b.stl")
    assert len(back) == len(bodies)
    assert sum(b.mesh.volume for b in back) == pytest.approx(
        sum(b.mesh.volume for b in bodies), rel=1e-4)


def test_rules_from_cad_solids():
    from bridgesim.cad.rules_cad import measure_cad
    from bridgesim.rules import RuleSet, evaluate_values

    rs = RuleSet.load()
    meas = measure_cad(simple_beam_bridge(span=1000), 650, 0.1, rs.constants)
    v = meas.values
    assert v["total_length_mm"] == pytest.approx(1100)
    assert v["clear_span_mm"] == pytest.approx(1000)
    assert v["deck_width_mm"] == pytest.approx(150)
    assert v["deck_height_mm"] == pytest.approx(173.1)
    rep = evaluate_values(meas, rs)
    assert {r.section for r in rep.results if r.passed is False} <= {"8.9"}


def test_idealisation_of_the_synthetic_bridge():
    from bridgesim.sim.idealize import build_model

    m = build_model(simple_beam_bridge(), MAT, SP, seed=0)
    kinds = {b.name: b.kind for b in m.bodies}
    assert kinds["Pier L (block)"] == "block"
    assert kinds["Stringer Near (3-ply UD)"] == "bar"
    assert kinds["Deck (2-ply 0-90)"] == "shell"
    assert len(m.patches) == 6  # 2 stringers on 2 piers + deck on 2 stringers
    total = sum(p.area_mm2 for p in m.patches)
    assert total == pytest.approx(4 * 50 * 6 + 2 * 1100 * 6, rel=0.01)
    # glue area is fully distributed over the glue points
    assert m.c_area.sum() == pytest.approx(total, rel=1e-6)


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def synthetic_crush(tmp_path_factory):
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    d = tmp_path_factory.mktemp("crush")
    write_stl(simple_beam_bridge(), d / "bridge.stl")
    rep, res = study.crush(d / "bridge.stl", seed=0, out_dir=d / "out",
                           options=CrushOptions(speed_mm_s=150, max_travel_mm=30))
    return d, rep, res


def _parse_cli(argv: list[str]) -> None:
    """Parse a bridgesim command line exactly as the CLI would, without running it: an
    unknown option or a missing argument raises the CLI's usage error."""
    import typer

    from bridgesim.cli import app

    group = typer.main.get_command(app)
    ctx = group.make_context("bridgesim", list(argv), resilient_parsing=False)
    sub = group.get_command(ctx, argv[0])
    assert sub is not None, argv[0]
    sub.make_context(argv[0], list(argv[1:]))


def test_next_steps_and_job_commands_parse(synthetic_crush, tmp_path):
    """The commands the reports suggest (next_steps) and the ones the MCP server starts in
    the background are valid command lines of this CLI (a wrong option would only show
    when an agent ran them)."""
    import asyncio
    import shlex

    from bridgesim.sim import mcp_server as M
    from bridgesim.sim import study

    d, rep, _ = synthetic_crush
    cad = str(d / "bridge.stl")
    ins = study.inspect(cad)
    steps = ins.next_steps + rep.next_steps
    assert {s.action for s in rep.next_steps} >= {"strengthen", "predict", "compare"}
    assert [s.action for s in ins.next_steps].count("crush") == 1
    for s in steps:
        if s.cli:
            argv = shlex.split(s.cli.replace("<changed design>", cad))
            assert argv[0] == "bridgesim"
            _parse_cli(argv[1:])
    strengthen = next(s for s in rep.next_steps if s.action == "strengthen")
    assert strengthen.why and (not strengthen.parts or strengthen.where_mm)

    pytest.importorskip("mcp")
    cmds = []
    real = M._start
    M._start = lambda cmd, out: cmds.append(cmd) or "{}"
    try:
        srv = M.build_server()
        for tool, args in (
                ("start_crush", {"rate_check": True, "strict": True}),
                ("start_montecarlo", {"n": 2}),
                ("start_sensitivity", {"rel": 0.1}),
                ("start_compare_designs", {"cad_a": cad, "cad_b": cad, "n": 2})):
            a = {"out_dir": str(tmp_path / tool), "overrides_yaml": "params: {}\n", **args}
            if "cad_a" not in a:
                a["cad_path"] = cad
            asyncio.run(srv.call_tool(tool, a))
    finally:
        M._start = real
    assert len(cmds) == 4
    for cmd in cmds:
        assert cmd[1:3] == ["-m", "bridgesim.cli"]
        _parse_cli(cmd[3:])


def test_section_scale_what_if(tmp_path):
    """A what-if without the CAD: section_scale thickens a member (area x s^2) or a plate
    (thickness x s), adds its mass to the bridge and the mass rule, and is refused on a
    rigid block; bad values are reported."""
    from bridgesim.sim import study

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    r0 = study.inspect(tmp_path / "b.stl")
    bars = [b for b in r0.bodies if b.kind == "bar"]
    shells = [b for b in r0.bodies if b.kind == "shell"]
    ov = {"bodies": {bars[0].name: {"section_scale": 1.5},
                     shells[0].name: {"section_scale": 2.0}}}
    r1 = study.inspect(tmp_path / "b.stl", ov)
    b1 = {b.name: b for b in r1.bodies}
    assert b1[bars[0].name].mass_g == pytest.approx(bars[0].mass_g * 2.25, rel=1e-3)
    assert b1[shells[0].name].thickness_mm == pytest.approx(2 * shells[0].thickness_mm)
    added = (b1[bars[0].name].mass_g - bars[0].mass_g + b1[shells[0].name].mass_g
             - shells[0].mass_g) / 1000
    assert r1.mass_kg == pytest.approx(r0.mass_kg + added, abs=2e-3)
    assert any(w.startswith("what-if:") for w in r1.warnings)
    bad = study.inspect(tmp_path / "b.stl", {"bodies": {bars[0].name: {"section_scale": -1}}})
    assert any("section_scale" in w and "above 0" in w for w in bad.warnings)


def test_compare_against_a_montecarlo_range(tmp_path):
    """compare places a real test inside or outside a montecarlo spread."""
    from bridgesim.sim import study
    from bridgesim.sim.report import MonteCarloReport

    mc = MonteCarloReport(file="b.stl", n=5, Fu_kgf=[90, 95, 100, 105, 110], p05_kgf=91,
                          p50_kgf=100, p95_kgf=109, failure_modes={"collapse": 5},
                          recommended_prediction_kgf=98)
    (tmp_path / "montecarlo.json").write_text(mc.model_dump_json())
    for load, inside in ((103, True), (140, False)):
        (tmp_path / "t.csv").write_text("test,load_kgf,event,notes\n"
                                        f"T1,{load},failure,top chord\n")
        gap = study.compare(tmp_path / "montecarlo.json", tmp_path / "t.csv")
        assert ("inside" in gap.metrics[0].meaning) == inside
        assert gap.competition_prediction_error == pytest.approx(abs(load - 98) / load,
                                                                 abs=1e-3)
    assert "too strong" not in gap.advice[0] and "too weak" in gap.advice[0]


def test_bridge_calibration_recovers_known_factors(tmp_path):
    """A 'test' made by crushing a laminated beam on two piers with its stiffness x0.8:
    calibrate-bridge finds the stiffness factor, leaves the strength alone (the beam
    breaks at its bending strength either way) and matches F_u and the stiffness."""
    import csv as _csv

    from bridgesim.cad.synthetic import box
    from bridgesim.sim import study

    pytest.importorskip("numba")
    write_stl([box("Pier L", (-550, 0, -40), (-540, 150, 40)),
               box("Pier R", (540, 0, -40), (550, 150, 40)),
               box("Beam", (-550, 150, -25), (550, 179.4, 25))], tmp_path / "beam.stl")
    ov = {"bodies": {"Pier*": {"kind": "block"},
                     "Beam": {"layup": "14-ply UD", "butt_stagger_mm": 0}}}
    true = json.loads(json.dumps(ov))
    true["params"] = {k: v * 0.8 for k, v in (
        ("material.E_MPa", MAT.E_MPa.value), ("material.G_MPa", MAT.G_MPa.value),
        ("wood.E_T_MPa", SP["wood.E_T_MPa"]), ("glue.K_n_N_per_mm3", SP["glue.K_n_N_per_mm3"]),
        ("glue.K_s_N_per_mm3", SP["glue.K_s_N_per_mm3"]))}
    rep, _ = study.crush(tmp_path / "beam.stl", true, seed=None, fidelity="quick",
                         replay=False, options=study._screening("quick"))
    L, D = rep.curve.load_kgf, rep.curve.deflection_mm
    k = int(np.argmax(np.array(L) >= 0.999 * rep.Fu_kgf))
    with (tmp_path / "test.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["test", "load_kgf", "midspan_deflection_mm", "event", "notes"])
        for i in range(k + 1):
            w.writerow(["T1", L[i], D[i], "failure" if i == k else "", "Beam"])
    cal = study.calibrate_bridge(tmp_path / "beam.stl", tmp_path / "test.csv", ov,
                                 fidelity="quick", seed=None, iterations=4)
    assert cal.converged, cal.steps
    assert cal.stiffness_factor == pytest.approx(0.8, rel=0.05)
    assert cal.strength_factor == pytest.approx(1.0, rel=0.05)
    assert "material.E_MPa" in cal.overrides["params"] and "bodies" in cal.overrides


def test_synthetic_bridge_crush(synthetic_crush):
    d, rep, res = synthetic_crush
    # two 6.2 x 19 stringers + a 4.1 mm deck over 1 m: tens of kgf, not hundreds
    assert 5 < rep.Fu_kgf < 80
    assert rep.failure_mode in ("collapse", "deflection", "contact")
    assert rep.quality.added_mass_fraction < 0.05
    assert rep.failure_sequence
    assert Path(rep.artifacts["replay_html"]).stat().st_size > 10_000
    data = json.loads(Path(rep.artifacts["report_json"]).read_text())
    assert data["kind"] == "crush" and data["Fu_kgf"] == rep.Fu_kgf


def test_brief_crush_report(synthetic_crush):
    from bridgesim.sim import study

    _, rep, _ = synthetic_crush
    b = study.brief(rep)
    assert b["kind"] == "crush_brief" and "curve" not in b and "events" not in b
    assert b["Fu_kgf"] == rep.Fu_kgf
    assert b["peak"]["load_kgf"] == pytest.approx(rep.Fu_kgf, rel=0.01)
    assert len(json.dumps(b)) < len(rep.model_dump_json())
    sev = [q["severity"] for q in b["questions"]]
    assert sev == sorted(sev, key=["important", "check", "info"].index)


def test_reality_gap(synthetic_crush, tmp_path):
    from bridgesim.sim import study

    d, rep, _ = synthetic_crush
    c = rep.curve
    k = len(c.load_kgf) // 2
    rows = ["test,bridge_file,load_kgf,midspan_deflection_mm,event,notes"]
    for i in range(0, k, max(1, k // 10)):
        rows.append(f"T1,bridge.stl,{c.load_kgf[i] * 1.1:.3f},{c.deflection_mm[i]:.3f},,")
    (tmp_path / "test.csv").write_text("\n".join(rows))
    gap = study.compare(Path(rep.artifacts["report_json"]), tmp_path / "test.csv")
    assert 0 <= gap.score_0_100 <= 100
    assert {m.name for m in gap.metrics} >= {"ultimate load", "initial stiffness"}


def test_inspect_json_cli(tmp_path):
    from typer.testing import CliRunner

    from bridgesim.cli import app

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    r = CliRunner().invoke(app, ["inspect", str(tmp_path / "b.stl"), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["kind"] == "inspect" and data["n_bodies"] == 5
    # unnamed STL bodies -> questions about their material
    assert any(q["topic"] == "material" for q in data["questions"])
    r = CliRunner().invoke(app, ["schema", "crush"])
    assert r.exit_code == 0 and "Fu_kgf" in r.output


# ----------------------------------------------------------- the user's SolidWorks file

SLDPRT = Path(os.environ.get("BRIDGESIM_TEST_SLDPRT",
                             ROOT / "examples" / "cad" / "Bridge_v6.SLDPRT"))


@pytest.mark.skipif(not SLDPRT.exists(), reason="no SolidWorks test part available")
def test_read_solidworks_part():
    bodies = load_cad(SLDPRT)
    assert len(bodies) == 100
    assert all(b.mesh.volume > 0 for b in bodies)
    for b in bodies:  # every body is a closed surface
        T = b.mesh.triangles
        e = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)
        _, c = np.unique(e, axis=0, return_counts=True)
        assert (c == 2).all(), b.name
    names = {b.name for b in bodies}
    assert any(n.startswith("Top Chord (10-ply UD") for n in names)


def test_mcp_server_exposes_the_tools():
    pytest.importorskip("mcp")
    import asyncio

    from bridgesim.sim.mcp_server import build_server

    tools = asyncio.run(build_server().list_tools())
    names = {getattr(t, "name", None) for t in getattr(tools, "tools", tools)}
    assert {"inspect_bridge", "crush_bridge", "start_crush", "job_status", "start_montecarlo",
            "start_compare_designs", "start_sensitivity",
            "compare_with_test", "lab_tests", "calibrate_material", "get_schema"} <= names


def test_mcp_tools_return_structured_json():
    """Each tool publishes an output schema, a call returns the report as structured
    content, and a bad call is flagged as an error carrying the JSON error object."""
    pytest.importorskip("mcp")
    import asyncio

    from bridgesim.sim.mcp_server import build_server

    srv = build_server()

    async def go():
        tools = await srv.list_tools()
        tools = getattr(tools, "tools", tools)
        assert all((getattr(t, "output_schema", None) or getattr(t, "outputSchema", None))
                   for t in tools)
        ok = await srv.call_tool("get_schema", {"name": "question"})
        bad = None
        try:
            await srv.call_tool("get_schema", {"name": "nope"})
        except Exception as exc:  # noqa: BLE001 - the SDK turns this into isError
            bad = str(exc)
        return ok, bad

    ok, bad = asyncio.run(go())
    structured = ok[1] if isinstance(ok, tuple) else getattr(ok, "structured_content", ok)
    assert structured["title"] == "Question"
    assert bad is not None and json.loads(bad[bad.index("{"):])["kind"] == "bad_input"

    async def missing():
        try:
            await srv.call_tool("inspect_bridge", {"cad_path": "no/such/bridge.SLDPRT"})
        except Exception as exc:  # noqa: BLE001
            return str(exc)

    msg = asyncio.run(missing())  # a relative path says where it was looked for
    assert "absolute path" in msg and str(Path.cwd()) in msg


# ------------------------------------------------------------- agent-facing contract


def test_cli_error_contract(tmp_path):
    """Bad input: exit 2 and one JSON error object on stderr, nothing on stdout."""
    from typer.testing import CliRunner

    from bridgesim.cli import app

    r = CliRunner().invoke(app, ["inspect", str(tmp_path / "missing.SLDPRT"), "--json"])
    assert r.exit_code == 2
    assert r.stdout == ""
    err = json.loads(r.stderr.strip().splitlines()[-1])
    assert err["kind"] == "bad_input" and "missing.SLDPRT" in err["error"]


def test_overrides_are_validated(tmp_path):
    """Typos in overrides are reported (with a suggestion), and are errors with strict."""
    from bridgesim.sim import study

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    ov = {"bodies": {"Stringr*": {"layup": "3-ply UD"}}}
    rep = study.inspect(tmp_path / "b.stl", overrides=ov)
    assert any("Stringr*" in w for w in rep.warnings)
    with pytest.raises(ValueError, match="Stringr"):
        study.inspect(tmp_path / "b.stl", overrides=ov, strict=True)
    with pytest.raises(ValueError, match="no_such_key"):
        study.inspect(tmp_path / "b.stl", overrides={"params": {"no_such_key": 1}})


def test_orientation_and_units(tmp_path):
    """A bridge drawn with Z up (and in metres) is flagged, and the axis/unit options fix
    it so the same model is built."""
    from bridgesim.sim import study

    bodies = simple_beam_bridge()
    write_stl(bodies, tmp_path / "y_up.stl")
    for b in bodies:  # file (x, y, z) = (X, -Z, Y): Z up
        V = b.mesh.vertices
        b.mesh.vertices = np.c_[V[:, 0], -V[:, 2], V[:, 1]] / 1000.0
    write_stl(bodies, tmp_path / "z_up_m.stl")
    ref = study.inspect(tmp_path / "y_up.stl")
    with pytest.raises(ValueError, match="--units m"):
        study.inspect(tmp_path / "z_up_m.stl")
    with pytest.raises(ValueError, match="--up-axis z"):  # still lying on its side
        study.inspect(tmp_path / "z_up_m.stl", units="m")
    good = study.inspect(tmp_path / "z_up_m.stl", units="m", up="z", span="x")
    assert not any(q.id == "geometry:orientation" for q in good.questions)
    for k in ("glued_patches", "beam_elements", "floss_cables"):
        assert good.model[k] == ref.model[k], k
    assert good.model["nodes"] == pytest.approx(ref.model["nodes"], rel=0.1)
    assert good.mass_kg == pytest.approx(ref.mass_kg, rel=1e-3)


def test_compare_ignores_the_post_peak_branch(synthetic_crush, tmp_path):
    """The stiffness of the reality gap uses the rising branch only, so a test curve that
    drops after the peak still gives a positive stiffness close to the simulated one."""
    from bridgesim.sim import study

    _, rep, _ = synthetic_crush
    c = rep.curve
    L = np.array(c.load_kgf)
    ipk = int(np.argmax(L >= 0.999 * rep.Fu_kgf))  # the failure
    end = ipk + int(np.argmax(L[ipk:] < 0.5 * rep.Fu_kgf))  # the record ends after the drop
    rows = ["test,bridge_file,load_kgf,midspan_deflection_mm,event,notes"]
    idx = sorted(set(range(0, end + 1, max(1, end // 60))) | {ipk, end})
    for i in idx:  # the whole record, the drop after the peak included
        ev = "failure" if i == ipk else ""
        rows.append(f"T1,b.stl,{c.load_kgf[i]:.3f},{c.deflection_mm[i]:.3f},{ev},")
    (tmp_path / "t.csv").write_text("\n".join(rows))
    gap = study.compare(Path(rep.artifacts["report_json"]), tmp_path / "t.csv")
    k = {m.name: m for m in gap.metrics}["initial stiffness"]
    assert k.simulated > 0 and k.measured > 0
    assert k.measured == pytest.approx(k.simulated, rel=0.25)
    assert gap.score_0_100 > 60


def test_load_prediction_minimises_the_competition_error():
    """§12.7 scores E = |F_u - F_p| / F_u: the best prediction is the 1/F-weighted median,
    which beats the plain median on a skewed spread."""
    from bridgesim.sim.study import best_prediction

    F = np.array([800.0, 900, 1000, 1100, 2400])
    p = best_prediction(F)

    def err(q):
        return float(np.mean(np.abs(F - q) / F))

    grid = np.linspace(700, 2500, 1801)
    assert err(p) == pytest.approx(min(err(q) for q in grid), rel=1e-6)
    assert err(p) <= err(float(np.median(F)))


def test_design_comparison_statistics():
    from bridgesim.sim.study import summarize_pairs

    def run(seed, F):
        return {"seed": seed, "Fu_kgf": F, "mass_kg": 2.0, "efficiency_kgf_per_kg": F / 2,
                "first_failure": "x", "trustworthy": True}

    a = [run(s, F) for s, F in enumerate([1000.0, 1200, 900, 1100], 1)]
    b = [run(s, F * 1.2) for s, F in enumerate([1000.0, 1200, 900, 1100], 1)]
    rep = summarize_pairs(a, b, "A", "B", "quick")
    assert rep.Fu_change == pytest.approx(0.2) and rep.significant
    assert "stronger" in rep.verdict and "4 of 4" in rep.verdict
    noisy = [run(s, F) for s, F in enumerate([1100.0, 1000, 1000, 1000], 1)]
    rep = summarize_pairs(a, noisy, "A", "B", "quick")
    assert not rep.significant and "not proven" in rep.verdict
    with pytest.raises(ValueError, match="same seeds"):
        summarize_pairs(a, b[:2], "A", "B", "quick")


def test_ab_cli_rejects_missing_designs(tmp_path):
    from typer.testing import CliRunner

    from bridgesim.cli import app

    write_stl(simple_beam_bridge(), tmp_path / "a.stl")
    r = CliRunner().invoke(app, ["ab", str(tmp_path / "a.stl"), str(tmp_path / "nope.stl"),
                                 "--json"])
    assert r.exit_code == 2 and json.loads(r.stderr.strip())["kind"] == "bad_input"


def test_sensitivity_rejects_unknown_parameters(tmp_path):
    from bridgesim.sim import study

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    with pytest.raises(ValueError, match="unknown parameter"):
        study.sensitivity(tmp_path / "b.stl", params=["material.E_GPa"])
    with pytest.raises(ValueError, match="rel"):
        study.sensitivity(tmp_path / "b.stl", rel=1.5)


def test_calibration_closes_the_loop(tmp_path):
    """Coupon data made with known values -> calibrate -> the simulated coupons
    (bridgesim lab) reproduce the data."""
    from bridgesim.sim import study

    E, ft, fc = 8000.0, 50.0, 36.0
    r = lab.three_point_bending(E, ft, fc, span=100, b=10, t=2, Gf=SP["wood.G_f_N_per_mm"])
    k = 48 * E * (10 * 2**3 / 12) / 100**3
    (tmp_path / "stick_bending_template.csv").write_text(
        "specimen,span_L_mm,width_b_mm,thickness_t_mm,load_F_N,deflection_mm,failure_load_N\n"
        f"S1,100,10,2,{k:.4f},1.0,{r.peak:.3f}\n")
    mat, log, params = study.calibrate(tmp_path)
    assert mat.E_MPa.value == pytest.approx(E, rel=1e-6) and mat.E_MPa.source == "measured"
    assert mat.G_MPa.source == "assumed" and "ratio" in mat.f_t_MPa.note
    gaps = study.lab(mat, None, tmp_path).gaps
    errs = [m.error for g in gaps for m in g.metrics]
    assert errs and max(abs(e) for e in errs) < 0.01


def test_simply_supported_beam_matches_beam_theory(tmp_path):
    """End to end (CAD -> idealisation -> crushing test): one laminated beam on two
    narrow piers under the rigid plate. The plate bears on the bending beam at its two
    edges, so the stiffness is that of two loads P/2 100 mm either side of mid-span
    (Euler-Bernoulli), and the beam breaks when the section reaches the stick's bending
    strength (between f_c, first yield, and f_b)."""
    from bridgesim.cad.synthetic import box
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    pytest.importorskip("numba")
    T = 29.4
    write_stl([box("Pier L", (-550, 0, -40), (-540, 150, 40)),
               box("Pier R", (540, 0, -40), (550, 150, 40)),
               box("Beam", (-550, 150, -25), (550, 150 + T, 25))], tmp_path / "beam.stl")
    kinds = {b.name: b.kind for b in study.inspect(tmp_path / "beam.stl").bodies}
    ov = {"bodies": {n: ({"layup": "14-ply UD", "butt_stagger_mm": 0} if k == "bar"
                         else {"kind": "block"}) for n, k in kinds.items()}}
    rep, _ = study.crush(tmp_path / "beam.stl", overrides=ov, seed=None, replay=False,
                         options=CrushOptions(speed_mm_s=100, max_travel_mm=60))
    L, D = np.array(rep.curve.load_kgf), np.array(rep.curve.deflection_mm)
    i = int(np.argmax(L >= 0.999 * rep.Fu_kgf))
    sel = (L[:i] > 0.1 * rep.Fu_kgf) & (L[:i] < 0.5 * rep.Fu_kgf)
    k_sim = np.polyfit(D[:i][sel], L[:i][sel], 1)[0]
    E, I, Ls = MAT.E_MPa.value, 50 * T**3 / 12, 1090.0  # supports at the pier centres
    a = Ls / 2 - 100
    x = np.linspace(-100, 100, 201) + Ls / 2  # deflection averaged under the plate

    def defl(x):  # two loads of 1/2 N at a and Ls - a
        out = np.zeros_like(x)
        for xp in (a, Ls - a):
            bb = Ls - xp
            out += np.where(x <= xp, 0.5 * bb * x * (Ls**2 - bb**2 - x**2) / (6 * E * I * Ls),
                            0.5 * xp * (Ls - x) * (Ls**2 - xp**2 - (Ls - x)**2)
                            / (6 * E * I * Ls))
        return out

    shear = 0.5 * a / (5 / 6 * MAT.G_MPa.value * 50 * T)  # Timoshenko: constant under the plate
    k_theory = 1 / (defl(x).mean() + shear) / 9.80665
    assert k_sim == pytest.approx(k_theory, rel=0.05)
    mor = rep.Fu_kgf * 9.80665 * 0.5 * a / (50 * T**2 / 6)
    assert MAT.f_c_MPa.value < mor < 1.05 * MAT.f_b_MPa.value


def test_deep_beam_includes_shear_deformation():
    """Wood shears easily (E/G ~ 16): a beam 50 mm deep over 200 mm deflects 2.2 times
    more than Euler-Bernoulli says. The Timoshenko fibre beam is exact for any mesh, and
    its bending strength is that of the fibres (shear flexibility adds no capacity)."""
    from bridgesim.sim.lab import BeamSpec, add_beam_line, blank_model, drive

    pytest.importorskip("numba")
    E, G, span, b, d = 10000.0, 625.0, 200.0, 20.0, 50.0
    I, A = b * d**3 / 12, b * d
    k_t = 1 / (span**3 / (48 * E * I) + span / (4 * 5 / 6 * G * A))

    def run(n_el, shear, **kw):
        X = np.c_[np.linspace(0, span, n_el + 1), np.zeros((n_el + 1, 2))]
        m = blank_model(X)
        add_beam_line(m, list(range(n_el + 1)), BeamSpec(b=b, d=d, E=E, G=G, shear=shear,
                                                          **kw), up=(0, 1, 0))
        return drive(m, {0: "xyz", n_el: "yz"}, n_el // 2, 1, 20.0, sign=-1.0,
                     max_disp=kw and 4.0 or 0.2, stop_drop=0.6 if kw else -1e9,
                     slow_disp=0.05, slow_rate=5.0)

    for n_el in (4, 16):
        r = run(n_el, True)
        sel = r.disp > 0.05
        assert np.polyfit(r.disp[sel], r.force[sel], 1)[0] == pytest.approx(k_t, rel=0.01)
    strong = {"ft": 80.0, "fc": 40.0, "Gf": 150.0}
    assert run(8, True, **strong).peak == pytest.approx(run(8, False, **strong).peak, rel=0.02)


def test_stick_bending_strength_matches_f_b():
    """The fibre section (tension softening, compression yielding) gives a single stick
    in 3-point bending the material's bending strength f_b (within 10 %)."""
    r = lab.three_point_bending(MAT.E_MPa.value, MAT.f_t_MPa.value, MAT.f_c_MPa.value,
                                Gf=SP["wood.G_f_N_per_mm"])
    mor = 3 * r.peak * 100 / (2 * 10 * 2**2)
    assert mor == pytest.approx(MAT.f_b_MPa.value, rel=0.1)


def test_background_jobs_survive_a_server_restart(tmp_path):
    """job_status reads a job's state from its files: a job started by one MCP server is
    'done' (with its report) when polled by another, and a killed one is 'failed'."""
    import signal
    import sys
    import time

    from bridgesim.sim import mcp_server as M

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    M._start([sys.executable, "-m", "bridgesim.cli", "inspect", str(tmp_path / "b.stl"),
              "--json"], tmp_path / "j1")
    M._start([sys.executable, "-c", "import time; time.sleep(120)"], tmp_path / "j2")
    M._JOBS.clear()  # a new server knows nothing of them
    for _ in range(120):
        st = json.loads(M.job_status_impl(str(tmp_path / "j1")))
        if st["status"] != "running":
            break
        time.sleep(0.5)
    assert st["status"] == "done" and st["report"]["kind"] == "inspect"
    assert json.loads(M.job_status_impl(str(tmp_path / "j2")))["status"] == "running"
    pid = json.loads((tmp_path / "j2" / "job.json").read_text())["pid"]
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    time.sleep(0.5)
    st = json.loads(M.job_status_impl(str(tmp_path / "j2")))
    assert st["status"] == "failed" and "died" in st["error"]["error"]
    assert json.loads(M.job_status_impl(str(tmp_path / "nothing")))["status"] == "unknown"


def test_compare_with_competition_style_records(synthetic_crush, tmp_path):
    """Records shaped like the competition's (synthetic data: no physical test is in the
    repository): a peak-only record is compared; a much softer, weaker test curve scores
    low; the failure location respects the side."""
    from bridgesim.sim import study
    from bridgesim.sim.study import _part_match

    _, rep, _ = synthetic_crush
    js = Path(rep.artifacts["report_json"])
    (tmp_path / "peak.csv").write_text(
        "test,bridge_file,load_kgf,midspan_deflection_mm,event,notes\n"
        f"T,b.stl,{rep.Fu_kgf * 1.1:.2f},,failure,something\n")
    gap = study.compare(js, tmp_path / "peak.csv")
    assert {m.name for m in gap.metrics} >= {"ultimate load"}
    assert gap.competition_prediction_error == pytest.approx(1 - 1 / 1.1, abs=0.01)
    c = rep.curve
    L = np.array(c.load_kgf)
    ipk = int(np.argmax(L >= 0.999 * rep.Fu_kgf))
    rows = ["test,bridge_file,load_kgf,midspan_deflection_mm,event,notes"]
    for i in range(0, ipk + 1, max(1, ipk // 40)):  # 3x softer, 31 % weaker
        rows.append(f"T,b.stl,{c.load_kgf[i] * 0.69:.3f},{c.deflection_mm[i] * 3 / 0.69:.3f},,")
    (tmp_path / "soft.csv").write_text("\n".join(rows))
    assert study.compare(js, tmp_path / "soft.csv").score_0_100 <= 40
    assert _part_match("top chord far side, left", "Top Chord (10-ply UD) - L Far") == 1.0
    assert _part_match("top chord far side", "Top Chord (10-ply UD) - Near") == 0.5
    assert _part_match("deck stringer", "Top Chord (10-ply UD) - Near") == 0.0


def test_agent_docs_match_the_code():
    """The question ids AGENTS.md tells agents about exist in the code, and the schema
    names the MCP server documents are the real ones."""
    import re

    from bridgesim.sim import mcp_server
    from bridgesim.sim.report import SCHEMAS

    doc = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    src = "".join((ROOT / "src" / "bridgesim" / "sim" / f).read_text(encoding="utf-8")
                  for f in ("idealize.py", "study.py"))
    i = doc.index("## Questions")
    section = doc[i: doc.index("\n## ", i + 5)]
    ids = set(re.findall(r"`([a-z_]+):?(?:<[^>]*>)?`", section)) - {"important", "check",
                                                                        "info", "acknowledge",
                                                                        "acknowledged_questions",
                                                                        "answer_with", "id",
                                                                        "topic", "severity",
                                                                        "question", "assumed",
                                                                        "material", "connection",
                                                                        "geometry", "rules",
                                                                        "design", "bodies",
                                                                        "glue", "params", "cad",
                                                                        "cli", "centroid_mm",
                                                                        "bbox_min_mm",
                                                                        "bbox_max_mm", "axis",
                                                                        "where_mm"}
    assert ids, "no question ids found in AGENTS.md"
    for qid in ids:
        assert f'"{qid}' in src or f"'{qid}" in src or f'f"{qid}' in src, qid
    for name in SCHEMAS:
        assert name in doc, name
    src_mcp = Path(mcp_server.__file__).read_text(encoding="utf-8")
    j = src_mcp.index("def get_schema")
    doc_schema = src_mcp[j: src_mcp.index("if name not in SCHEMAS", j)]
    for name in SCHEMAS:
        assert name in doc_schema, name
    # every command line in the AGENTS.md table parses, with and without its options
    rows = re.findall(r"^\| `(bridgesim [^`]+)`", doc, flags=re.M)
    assert len(rows) >= 9
    for row in rows:
        cmd = row.replace("\\|", "|").replace("]...", "]")  # "..." marks a repeatable option
        for variant in (re.sub(r"\[([^\[\]]*)\]", r"\1", cmd), re.sub(r"\[[^\[\]]*\]", "", cmd)):
            argv = [{"N": "0"}.get(t, t).split("|")[0] for t in variant.split()]
            _parse_cli(argv[1:])


def test_questions_can_be_acknowledged(tmp_path):
    """A designer's 'as drawn' closes a question; a typo in acknowledge is reported."""
    from bridgesim.sim import study

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    first = study.inspect(tmp_path / "b.stl")
    layup_q = [q.id for q in first.questions if q.id.startswith("layup:")]
    assert layup_q
    rep = study.inspect(tmp_path / "b.stl",
                        overrides={"acknowledge": ["layup:*", "no_such_question"],
                                   "cad": "x"})
    assert not any(q.id.startswith("layup:") for q in rep.questions)
    assert set(layup_q) <= set(rep.acknowledged_questions)
    assert any("no_such_question" in w for w in rep.warnings)
    assert any("'cad'" in w for w in rep.warnings)
    b = study.inspect_brief(rep)
    assert b["kind"] == "inspect_brief" and "bodies" not in b and b["n_bodies"] == 5


def test_glue_bearing_follows_the_grain():
    """A member pressed on its end grain bears at f_c, across its grain at f_c90 (and in
    between by Hankinson); a plate pressed on its face bears at f_c90."""
    from bridgesim.sim.idealize import analyse_shape, grain_bearing, hankinson

    fc, fc90 = 30.0, 5.0
    assert hankinson(fc, fc90, 1.0) == pytest.approx(fc)
    assert hankinson(fc, fc90, 0.0) == pytest.approx(fc90)
    assert fc90 < hankinson(fc, fc90, 0.7) < fc
    bar = analyse_shape(box("Post (6-ply UD)", (0, 0, 0), (200, 12, 19)),
                        parse_layup("Post (6-ply UD)"), None)
    assert bar.kind == "bar"
    f_end, e_end = grain_bearing(bar, np.array([1.0, 0, 0]), fc, fc90)
    f_side, e_side = grain_bearing(bar, np.array([0.0, 1, 0]), fc, fc90)
    assert (f_end, e_end) == (pytest.approx(fc), 1.0)
    assert (f_side, e_side) == (pytest.approx(fc90), 0.0)
    plate = analyse_shape(box("Gusset (2-ply 0-90)", (0, 0, 0), (120, 80, 4)),
                          parse_layup("Gusset (2-ply 0-90)"), None)
    assert plate.kind == "shell"
    f_face, _ = grain_bearing(plate, np.array([0.0, 0, 1]), fc, fc90)
    f_edge, e_edge = grain_bearing(plate, np.array([1.0, 0, 0]), fc, fc90)
    assert f_face == pytest.approx(fc90)
    assert f_edge == pytest.approx((fc + fc90) / 2) and e_edge == 0.5
