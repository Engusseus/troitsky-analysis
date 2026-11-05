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
# the stick modulus the crush and the screen use: the craft-stick placeholder while the
# material's E is assumed (study._inputs)
E_SIM = SP["wood.E_placeholder_MPa"] if MAT.E_MPa.source == "assumed" else MAT.E_MPa.value
G_SIM = E_SIM / 16 if MAT.E_MPa.source == "assumed" else MAT.G_MPa.value


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


def test_a_seed_is_the_same_bridge_at_every_fidelity():
    """The random glue strength of each joint is drawn from a stream keyed on the CAD
    (the two bodies and their overlap), not on the mesh: a seed is the same imperfect
    bridge at quick, standard and fine fidelity."""
    from bridgesim.sim.idealize import build_model

    out = []
    for scale in (1.6, 1.0, 0.75):
        m = build_model(simple_beam_bridge(), MAT, SP, seed=3, mesh_scale=scale)
        out.append({tuple(sorted((m.bodies[p.body_a].name, m.bodies[p.body_b].name))):
                    float(m.c_GIc[(m.c_patch == k) & m.c_breakable].mean())
                    for k, p in enumerate(m.patches)
                    if ((m.c_patch == k) & m.c_breakable).any()})
    assert out[0] and out[0].keys() == out[1].keys() == out[2].keys()
    for o in out[1:]:
        assert o == pytest.approx(out[0], rel=1e-12)


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
    M._start = lambda cmd, out: cmds.append(cmd) or json.dumps(
        {"status": "running", "out_dir": str(out)})
    try:
        srv = M.build_server()
        for tool, args in (
                ("start_crush", {"rate_check": True, "strict": True}),
                ("start_montecarlo", {"n": 2}),
                ("start_sensitivity", {"rel": 0.1}),
                ("start_diagnose", {"n": 1, "candidates": 2}),
                ("start_compare_designs", {"cad_a": cad, "cad_b": cad, "n": 2})):
            a = {"out_dir": str(tmp_path / tool), "overrides_yaml": "params: {}\n", **args}
            if "cad_a" not in a:
                a["cad_path"] = cad
            asyncio.run(srv.call_tool(tool, a))
    finally:
        M._start = real
    assert len(cmds) == 5
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
    # a glue what-if on every copy of a joint at once: glob patterns, either order
    from bridgesim.sim.idealize import build_model

    base = build_model(simple_beam_bridge(), MAT, SP, seed=None)
    from bridgesim.sim.idealize import Overrides

    ov = Overrides.from_dict({"glue": {"Stringer* | Pier*": 2.0}})
    m2 = build_model(simple_beam_bridge(), MAT, SP, ov, seed=None)
    hit = [k for k, p in enumerate(base.patches)
           if {base.bodies[p.body_a].name[:4], base.bodies[p.body_b].name[:4]} == {"Stri", "Pier"}]
    assert len(hit) == 4 and not [w for w in m2.notes if "glue joint" in w]
    for k in hit:
        sel = base.c_patch == k
        assert m2.c_GIc[sel].mean() == pytest.approx(2 * base.c_GIc[sel].mean())


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
    # the 10 GPa of the material file (the craft-stick placeholder off): at 7.1 GPa x 0.8
    # this beam reaches the 50 mm deflection limit about when it breaks, and a fit of
    # the strength would have nothing to go by
    ov = {"bodies": {"Pier*": {"kind": "block"},
                     "Beam": {"layup": "14-ply UD", "butt_stagger_mm": 0}},
          "params": {"wood.E_placeholder_MPa": 0}}
    true = json.loads(json.dumps(ov))
    true["params"] = {"wood.E_placeholder_MPa": 0} | {k: v * 0.8 for k, v in (
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
                                 fidelity="quick", seed=None, seeds=1, iterations=4)
    assert cal.converged, cal.steps
    assert cal.stiffness_factor == pytest.approx(0.8, rel=0.05)
    assert cal.strength_factor == pytest.approx(1.0, rel=0.05)
    assert "material.E_MPa" in cal.overrides["params"] and "bodies" in cal.overrides


def test_agent_loop_through_mcp(tmp_path):
    """The loop an agent drives, through the MCP tools only: inspect, answer the questions
    in overrides, crush, take the strengthen step's what-if, test it with a background A/B
    comparison on the same seed, and see the design get stronger."""
    import asyncio
    import time

    import yaml

    from bridgesim.cad.synthetic import box
    from bridgesim.sim.mcp_server import build_server

    pytest.importorskip("mcp")
    pytest.importorskip("numba")
    cad = tmp_path / "beam.stl"
    write_stl([box("Pier L", (-550, 0, -40), (-540, 150, 40)),
               box("Pier R", (540, 0, -40), (550, 150, 40)),
               box("Beam", (-550, 150, -25), (550, 179.4, 25))], cad)
    srv = build_server()

    def call(tool, **kw):
        out = asyncio.run(srv.call_tool(tool, kw))
        out = out[1] if isinstance(out, tuple) else getattr(out, "structured_content", out)
        return out

    ins = call("inspect_bridge", cad_path=str(cad), brief=False)
    assert ins["questions"] and ins["next_steps"]
    # the designer's answers: what each body is (an STL has no names: the agent reads
    # them, and where each body is, from the full inspect report)
    bodies = {b["name"]: ({"layup": "14-ply UD", "butt_stagger_mm": 0} if b["kind"] == "bar"
                          else {"kind": "block"}) for b in ins["bodies"]}
    ov = {"bodies": bodies, "acknowledge": ["material:*", "rigid_*", "floating:*", "rule:*",
                                           "weak_joints"]}
    ov_yaml = yaml.safe_dump(ov)
    ins2 = call("inspect_bridge", cad_path=str(cad), overrides_yaml=ov_yaml, strict=True)
    assert not [q for q in ins2["questions"] if q["severity"] == "important"], \
        ins2["questions"]
    scr = call("screen_bridge", cad_path=str(cad), overrides_yaml=ov_yaml)  # seconds
    assert scr["kind"] == "screen" and scr["limits"] and scr["stiffness_kgf_per_mm"] > 0
    rep = call("crush_bridge", cad_path=str(cad), overrides_yaml=ov_yaml, fidelity="quick",
               seed=1, out_dir=str(tmp_path / "A"))
    assert rep["kind"] == "crush_brief" and rep["Fu_kgf"] > 50
    step = next(s for s in rep["next_steps"] if s["action"] == "strengthen")
    assert step["overrides"], step  # a what-if to try before touching the CAD
    # the what-if with the designer's answers, saved next to the report for ab
    whatif = Path(rep["artifacts"]["whatif_yaml"]).read_text(encoding="utf-8")
    wb = yaml.safe_load(whatif)["bodies"]
    assert all(wb[n].get("layup") == v.get("layup") for n, v in bodies.items())
    call("start_compare_designs", cad_a=str(cad), cad_b=str(cad), out_dir=str(tmp_path / "ab"),
         n=1, overrides_a_yaml=ov_yaml, overrides_b_yaml=whatif, fidelity="quick")
    for _ in range(600):
        st = call("job_status", out_dir=str(tmp_path / "ab"))
        if st["status"] != "running":
            break
        time.sleep(1)
    assert st["status"] == "done", st
    assert st["report"]["Fu_change"] > 0.1  # the what-if made it stronger


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


def test_overrides_never_read_an_arbitrary_file(tmp_path):
    """Overrides typed as text (MCP, the app) must be a mapping: a bare string is refused,
    never taken as a file path. A file is read only with a .yaml/.yml name, and must hold
    a mapping; its content is not echoed in the error."""
    import asyncio

    from bridgesim.sim import study

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    with pytest.raises(ValueError, match="yaml"):
        study.load_overrides(str(secret))
    listy = tmp_path / "list.yaml"
    listy.write_text("- TOP-SECRET-CONTENT\n")
    with pytest.raises(ValueError, match="mapping") as exc:
        study.load_overrides(listy)
    assert "TOP-SECRET" not in str(exc.value)
    pytest.importorskip("mcp")
    from bridgesim.sim.mcp_server import build_server

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    try:
        asyncio.run(build_server().call_tool(
            "inspect_bridge", {"cad_path": str(tmp_path / "b.stl"),
                               "overrides_yaml": str(secret)}))
        raise AssertionError("a bare string was accepted as overrides")
    except Exception as e:  # noqa: BLE001 - the SDK's ToolError
        assert "mapping" in str(e) and "TOP-SECRET" not in str(e)


def test_unfilled_answers_are_refused(tmp_path):
    """Feeding the questions' answer_with back without filling them in is an error that
    names the key (never read as '1-ply UD'); a layup override must follow the strict
    form, and a non-number in params names its key."""
    from bridgesim.sim import study
    from bridgesim.sim.materials import parse_override_layup

    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    rep = study.inspect(tmp_path / "b.stl")
    merged: dict = {}
    for q in rep.questions:
        for sec in ("bodies", "glue", "params"):
            for k, v in (q.answer_with or {}).get(sec, {}).items():
                merged.setdefault(sec, {})[k] = v
    assert merged, "the synthetic bridge asks questions with answer_with"
    with pytest.raises(ValueError, match="placeholders") as exc:
        study.inspect(tmp_path / "b.stl", merged)
    assert "<" in str(exc.value)
    assert parse_override_layup("10-ply UD").angles == [0.0] * 10
    assert parse_override_layup("4-ply +-45 along the span").grain == "span"
    for bad in ("<N>-ply <UD | 0-90 | +-45>, or block", "10-ply <UD | 0-90 | +-45>",
                "10 plies", "UD"):
        assert parse_override_layup(bad) is None, bad
    with pytest.raises(ValueError, match="params.floss.strands"):
        study.inspect(tmp_path / "b.stl", {"params": {"floss.strands": "six"}})


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


def test_direct_coupon_tests_calibrate_their_values(tmp_path):
    """Tension, short-stack compression, a block pressed flat, end-to-end joints and a
    lap load-slip record each set their value; the templates as shipped set nothing."""
    import shutil

    from bridgesim.sim import study

    for f in (ROOT / "validation" / "physical").glob("*_template.csv"):
        shutil.copy(f, tmp_path / f.name)
    with pytest.raises(ValueError, match="no coupon test results"):
        study.calibrate(tmp_path)  # the templates as shipped hold no results
    (tmp_path / "stick_tension_template.csv").write_text(
        "specimen,width_mm,thickness_mm,failure_load_N\nT1,10,2,1600\nT2,10,2,1700\n")
    (tmp_path / "stick_compression_template.csv").write_text(
        "specimen,length_mm,width_mm,thickness_mm,failure_load_N\n"
        "C1,30,10,10,4000\nC2,200,10,10,1000\n")  # C2 is a column: skipped
    (tmp_path / "crush_across_template.csv").write_text(
        "specimen,contact_length_mm,width_mm,plateau_load_N\nP1,30,10,1800\n")
    (tmp_path / "end_grain_template.csv").write_text(
        "specimen,width_mm,thickness_mm,failure_load_N\nE1,10,2,20\n")
    slip = np.linspace(0, 0.4, 9)
    load = np.where(slip <= 0.2, 2000 * slip, 400 - 2000 * (slip - 0.2))
    (tmp_path / "lap_slip_template.csv").write_text(
        "specimen,overlap_mm,width_mm,load_N,slip_mm\n"
        + "".join(f"L1,20,10,{f_:.3f},{x:.3f}\n" for f_, x in zip(load, slip, strict=True)))
    mat, log, params = study.calibrate(tmp_path)
    assert mat.f_t_MPa.value == pytest.approx(82.5) and mat.f_t_MPa.source == "measured"
    assert mat.f_c_MPa.value == pytest.approx(40.0)
    assert params["wood.f_c90_MPa"] == pytest.approx(6.0)
    tau = MAT.glue.tau_g_MPa.value
    assert params["glue.end_grain_factor"] == pytest.approx(1.0 / (SP["glue.peel_to_shear"]
                                                                   * tau), rel=1e-6) or \
        params["glue.end_grain_factor"] == 1.5
    assert params["glue.K_s_N_per_mm3"] == pytest.approx(2000 / 200, rel=1e-6)
    assert params["glue.G_IIc_N_per_mm"] == pytest.approx(0.5 * 400 * 0.4 / 200, rel=1e-6)
    assert any("skipped" in ln for ln in log)


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
    E, I, Ls = E_SIM, 50 * T**3 / 12, 1090.0  # supports at the pier centres
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

    shear = 0.5 * a / (5 / 6 * G_SIM * 50 * T)  # Timoshenko: constant under the plate
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


def test_deep_laminated_beam_fails_in_shear():
    """A deep beam (span 2.5x its depth, the shape of a floor beam under the plate) that
    is strong in bending splits in shear at P = 4/3 f_v A (tau_max = 1.5 V / A), on a
    coarse and a fine mesh."""
    from bridgesim.sim.lab import BeamSpec, add_beam_line, blank_model, drive

    pytest.importorskip("numba")
    span, b, d, fv = 150.0, 20.0, 60.0, 3.0
    for n_el in (6, 12):
        X = np.c_[np.linspace(0, span, n_el + 1), np.zeros((n_el + 1, 2))]
        m = blank_model(X)
        add_beam_line(m, list(range(n_el + 1)), BeamSpec(b=b, d=d, E=10000.0, G=625.0,
                                                          ft=80.0, fc=60.0, Gf=50.0, fv=fv),
                      up=(0, 1, 0))
        r = drive(m, {0: "xyz", n_el: "yz"}, n_el // 2, 1, 20.0, sign=-1.0, max_disp=1.5,
                  stop_drop=0.5, slow_disp=0.05, slow_rate=5.0)
        assert r.peak == pytest.approx(4 / 3 * fv * b * d, rel=0.02)
        assert r.state.b_shear.any() and not r.state.b_fractured.any()


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


def test_background_jobs_queue_and_cancel(tmp_path, monkeypatch):
    """With one job slot, a second background job waits as 'queued' instead of fighting
    the first for the cores, and cancel_job stops it."""
    import sys
    import time

    from bridgesim.sim import mcp_server as M

    monkeypatch.setenv("BRIDGESIM_JOB_SLOTS", "1")
    monkeypatch.setenv("BRIDGESIM_JOB_SLOT_DIR", str(tmp_path / "slots"))
    M._start([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "j1")
    time.sleep(3.0)  # j1 takes the only slot
    M._start([sys.executable, "-c", "print('{}')"], tmp_path / "j2")
    for _ in range(60):
        st2 = json.loads(M.job_status_impl(str(tmp_path / "j2")))
        if st2["status"] == "queued":
            break
        time.sleep(0.25)
    assert st2["status"] == "queued", st2
    assert json.loads(M.job_status_impl(str(tmp_path / "j1")))["status"] == "running"
    assert json.loads(M.cancel_job_impl(str(tmp_path / "j1")))["status"] == "cancelled"
    assert json.loads(M.job_status_impl(str(tmp_path / "j1")))["status"] == "cancelled"
    for _ in range(80):  # the queued job gets the slot and finishes
        st2 = json.loads(M.job_status_impl(str(tmp_path / "j2")))
        if st2["status"] not in ("queued", "running"):
            break
        time.sleep(0.25)
    assert st2["status"] == "done", st2


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


# ------------------------------------------------------------- round 8: the agent loop


def test_simulation_params_are_range_checked(tmp_path):
    """A params override outside its physical range is an input error naming the key;
    the defaults are all inside their ranges."""
    from bridgesim.sim import study
    from bridgesim.sim.materials import param_range_error

    assert [k for k, v in SP.values.items() if param_range_error(k, v)] == []
    for key, val in (("glue.K_n_N_per_mm3", -5), ("contact.friction_platform", 7),
                     ("scatter.glue_cov", -1), ("wood.nu_LT", "soft")):
        with pytest.raises(ValueError, match=key.replace(".", r"\.")):
            SP.updated({key: val})
    assert SP.updated({"floss.strands": 6})["floss.strands"] == 6
    stl = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), stl)
    with pytest.raises(ValueError, match="physical range"):
        study.inspect(stl, {"params": {"glue.G_Ic_N_per_mm": 0}})


def test_whatif_is_written_with_the_users_overrides(synthetic_crush):
    """The strengthen step's what-if is saved next to the report, merged with the user's
    overrides, and every command of next_steps names files by absolute path."""
    import shlex

    import yaml

    from bridgesim.sim.study import _copies, merge_whatif

    d, rep, _ = synthetic_crush
    st = next(s for s in rep.next_steps if s.action == "strengthen")
    if st.overrides:
        wy = Path(rep.artifacts["whatif_yaml"])
        assert wy.parent == (d / "out").resolve()
        assert yaml.safe_load(wy.read_text()) == merge_whatif({}, st.overrides)
        assert str(wy) in st.cli
    for s in rep.next_steps:
        if s.cli and s.action in ("crush", "rerun", "confirm", "predict", "strengthen"):
            argv = shlex.split(s.cli)
            assert Path(argv[2]).is_absolute(), s.cli
    user = {"bodies": {"Stringer*": {"section_scale": 1.2, "grain": "span"}},
            "params": {"floss.strands": 6}}
    wi = {"bodies": {"Stringer*": {"section_scale": 1.25}}, "glue": {"A* | B*": 1.5}}
    m = merge_whatif(user, wi)
    assert m["bodies"]["Stringer*"] == {"section_scale": 1.5, "grain": "span"}
    assert m["glue"] == {"A* | B*": 1.5} and m["params"] == {"floss.strands": 6}
    assert user["bodies"]["Stringer*"]["section_scale"] == 1.2  # not modified
    # a what-if is made on every copy of a part, and only on its copies
    assert _copies("Top Chord (10-ply UD) - Far") == "Top Chord (10-ply UD)*"
    assert _copies("Body 4") == "Body 4"


def test_inspect_selects_bodies_and_joints(tmp_path):
    """inspect --body/--joint: only the parts asked for, in full, with their questions."""
    from typer.testing import CliRunner

    from bridgesim.cli import app
    from bridgesim.sim import study

    stl = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), stl)
    rep = study.inspect(stl)  # an STL has no body names: "Body 1" ...
    bars = sorted(b.name for b in rep.bodies if b.kind == "bar")
    s = study.select(rep, body=[n.upper() for n in bars])  # case ignored
    assert sorted(b.name for b in s.bodies) == bars
    # with their joints (each stringer: two piers and the deck)
    assert s.joints and all(j.a in bars or j.b in bars for j in s.joints)
    assert len(s.joints) == len([j for j in rep.joints if j.a in bars or j.b in bars])
    s = study.select(rep, joint=bars[0])  # a stringer: on two piers, under the deck
    assert s.bodies == [] and len(s.joints) == 3
    assert all(bars[0] in (j.a, j.b) for j in s.joints)
    names = {p for j in s.joints for p in (j.a, j.b)}
    assert all(any(n in q.id or n in q.question for n in names) for q in s.questions)
    deck = next(b.name for b in rep.bodies if b.kind == "shell")
    r = CliRunner().invoke(app, ["inspect", str(stl), "--body", deck, "--json"])
    assert r.exit_code == 0, r.stderr
    assert [b["name"] for b in json.loads(r.stdout)["bodies"]] == [deck]


def test_crush_brief_keeps_only_important_questions_in_full(synthetic_crush):
    """The brief carries the important questions in full and counts the others (their
    ids are in the ask_designer step), without empty fields in its next steps."""
    from bridgesim.sim import study

    _, rep, _ = synthetic_crush
    b = study.brief(rep)
    assert all(q["severity"] == "important" and "answer_with" in q for q in b["questions"])
    assert sum(b["n_questions"].values()) == len(rep.questions)
    check = [q.id for q in rep.questions if q.severity == "check"]
    asked = [i for s in b["next_steps"] if s["action"] == "ask_designer" for i in s["ids"]]
    assert set(check) <= set(asked)
    assert all(v not in (None, [], {}) for s in b["next_steps"] for v in s.values())
    assert len(b["failures"]) <= 10
    assert not any(k in f for f in b["failures"] for k in ("t_s", "load_N", "parts"))


def test_calibrate_without_data_is_an_input_error(tmp_path):
    from typer.testing import CliRunner

    from bridgesim.cli import app
    from bridgesim.sim import study

    with pytest.raises(ValueError, match="no coupon test results"):
        study.calibrate(tmp_path)
    r = CliRunner().invoke(app, ["calibrate", str(tmp_path), "--out",
                                 str(tmp_path / "m.yaml"), "--json"])
    assert r.exit_code == 2 and r.stdout == ""
    assert json.loads(r.stderr.strip().splitlines()[-1])["kind"] == "bad_input"
    assert not (tmp_path / "m.yaml").exists()


def test_failure_location_counts_the_first_failure_fully(synthetic_crush, tmp_path):
    """A test that broke the part the simulation broke first scores 0 error; one that
    broke the part it broke second near the peak scores 0.5."""
    from bridgesim.sim import study
    from bridgesim.sim.report import EventOut

    _, rep, _ = synthetic_crush
    f1 = EventOut(t_s=0.1, jack_travel_mm=5.0, load_N=rep.Fu_N, load_kgf=rep.Fu_kgf,
                  kind="member_fracture", what="Top Chord - Far", parts=["Top Chord - Far"])
    f2 = f1.model_copy(update={"what": "Deck Stringer - Near",
                               "parts": ["Deck Stringer - Near"]})
    r2 = rep.model_copy(update={"failures": [f1, f2]})
    (tmp_path / "c.json").write_text(r2.model_dump_json())
    err = {}
    for part in ("top chord far", "deck stringer near", "end post"):
        (tmp_path / "t.csv").write_text("test,load_kgf,event,notes\n"
                                        f"T,{rep.Fu_kgf:.1f},failure,{part}\n")
        gap = study.compare(tmp_path / "c.json", tmp_path / "t.csv")
        err[part] = next(m.error for m in gap.metrics if m.name == "failure location")
    assert err == {"top chord far": 0.0, "deck stringer near": 0.5, "end post": 1.0}


def test_montecarlo_can_sample_the_placeholders():
    """--sample-placeholders draws each placeholder value per run (the same for the same
    seed), keeping the user's overrides; measured values are not drawn."""
    from bridgesim.sim.study import PLACEHOLDER_COV, draw_placeholders, placeholders

    ph = placeholders(MAT, SP)
    assert "material.glue.tau_g_MPa" in ph and "material.E_MPa" in ph
    user = {"params": {"floss.strands": 6}, "bodies": {"X*": {"glue_factor": 0.8}}}
    a, b = draw_placeholders(user, ph, 1), draw_placeholders(user, ph, 1)
    assert a == b and a != draw_placeholders(user, ph, 2)
    assert a["params"]["floss.strands"] == 6 and a["bodies"] == user["bodies"]
    assert set(a["params"]) == set(ph) | {"floss.strands"}
    logs = [math.log(draw_placeholders({}, ph, s)["params"]["material.E_MPa"]
                     / ph["material.E_MPa"]) for s in range(200)]
    assert abs(np.mean(logs)) < 0.05 and np.std(logs) == pytest.approx(PLACEHOLDER_COV,
                                                                      rel=0.15)
    measured = MAT.model_copy(update={"E_MPa": MAT.E_MPa.model_copy(
        update={"source": "measured"})})
    assert "material.E_MPa" not in placeholders(measured, SP)


def _x_plate(name: str, half: float, arm: float, y0: float, y1: float):
    """An X of two arms ``arm`` mm wide, 2 x ``half`` mm across, as one flat body."""
    from bridgesim.cad.synthetic import prism

    w = arm / 2
    P = []
    for ang in (45, 135, 225, 315):
        a = math.radians(ang)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        an = math.radians(ang + 45)
        P += [half * u - w * v, half * u + w * v,
              np.array([math.cos(an), math.sin(an)]) * w * math.sqrt(2)]
    return prism(name, P, y0, y1)


def test_geometry_the_model_cannot_represent_well_is_asked_about():
    """A plate made of narrow arms (an X) becomes a frame of four members meeting at its
    centre (frame_plates says so); kept a plate, it is asked about (narrow_plates). Faces
    read only approximately (a freeform face flattened) raise a question too."""
    from bridgesim.cad import CadBody
    from bridgesim.cad.mesh import tessellate
    from bridgesim.cad.parasolid import Body
    from bridgesim.sim.idealize import Overrides, build_model

    x = _x_plate("X Brace (6-ply UD)", 100.0, 12.0, 173.1, 179.1)
    assert x.mesh.volume == pytest.approx(6 * (4 * 100 * 12 - 12**2), rel=1e-6)
    bodies = simple_beam_bridge()
    pier = bodies[0]
    faces = list(pier.faces)
    top = int(np.argmax([f.normal[1] for f in faces]))  # its top face, read as freeform
    faces[top] = type(faces[top])("other", faces[top].loops)
    bodies[0] = CadBody(pier.name, tessellate(Body(0, pier.name, faces)), faces, 0)
    m = build_model(bodies + [x], MAT, SP, seed=0)
    xb = next(b for b in m.bodies if b.name == "X Brace (6-ply UD)")
    assert xb.kind == "frame" and xb.section_mm[1] == pytest.approx(6.0)
    # four arms of ~100 mm from the centre, all meeting at one node
    ends = np.concatenate([[m.b_i[e], m.b_j[e]] for e in xb.beams])
    centre = np.bincount(ends).argmax()
    assert np.bincount(ends)[centre] == 4
    assert float(m.b_L0[xb.beams].sum()) == pytest.approx(4 * (100 - 6), rel=0.08)
    qs = {q["id"]: q for q in m.questions}
    assert "X Brace (6-ply UD)" in qs["frame_plates"]["question"]
    assert "narrow_plates" not in qs
    m2 = build_model(bodies + [x], MAT, SP, Overrides.from_dict(
        {"bodies": {"X Brace*": {"kind": "shell"}}}), seed=0)
    assert {b.name: b.kind for b in m2.bodies}["X Brace (6-ply UD)"] == "shell"
    assert "X Brace (6-ply UD)" in {q["id"]: q for q in m2.questions}[
        "narrow_plates"]["question"]
    q = qs["cad_approximated:Pier L (block)"]
    assert q["severity"] == "important" and "10.0%" in q["question"]  # 50x150 of 75000
    assert "cad_approximated:Pier R (block)" not in qs


def test_linear_screen_matches_beam_theory(tmp_path):
    """screen (seconds, linear): a laminated beam on two piers under the plate has the
    stiffness of beam theory with the plate bearing at its two edges (the contacts are
    compression only), and its extreme fibre reaches f_c, then f_t, at the loads beam
    theory gives; the CLI prints the report as JSON."""
    from typer.testing import CliRunner

    from bridgesim.cad.synthetic import box
    from bridgesim.cli import app
    from bridgesim.sim import study

    T = 29.4
    cad = tmp_path / "beam.stl"
    write_stl([box("Pier L", (-550, 0, -40), (-540, 150, 40)),
               box("Pier R", (540, 0, -40), (550, 150, 40)),
               box("Beam", (-550, 150, -25), (550, 150 + T, 25))], cad)
    kinds = {b.name: b.kind for b in study.inspect(cad).bodies}
    ov = {"bodies": {n: ({"layup": "14-ply UD", "butt_stagger_mm": 0} if k == "bar"
                         else {"kind": "block"}) for n, k in kinds.items()}}
    rep = study.screen(cad, overrides=ov)
    E, I, Ls, S = E_SIM, 50 * T**3 / 12, 1090.0, 50 * T**2 / 6
    a = Ls / 2 - 100  # the plate's edges, 100 mm either side of mid-span
    # deflection under each of two loads P/2 at a and Ls - a (unit P), plus shear
    d = 0.5 * a**2 * (3 * Ls - 4 * a) / (6 * E * I) * 2 / 2 \
        + 0.5 * a / (5 / 6 * G_SIM * 50 * T)
    assert rep.stiffness_kgf_per_mm == pytest.approx(1 / d / 9.80665, rel=0.03)
    lim = {lm.mode: lm.load_kgf for lm in rep.limits}
    for mode, f in (("compression", MAT.f_c_MPa.value), ("tension", MAT.f_t_MPa.value)):
        assert lim[mode] == pytest.approx(f * S / (0.5 * a) / 9.80665, rel=0.03)
    assert rep.first_limit.startswith("compression") and rep.wall_s < 30
    assert [lm.utilisation for lm in rep.limits] == sorted(
        (lm.utilisation for lm in rep.limits), reverse=True)
    import yaml

    (tmp_path / "ov.yaml").write_text(yaml.safe_dump(ov))
    r = CliRunner().invoke(app, ["screen", str(cad), "-o", str(tmp_path / "ov.yaml"),
                                 "--top", "3", "--json"])
    assert r.exit_code == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["kind"] == "screen" and len(out["limits"]) == 3


def test_interlaminar_strength_follows_the_plies():
    """The glue lines between plies split at the bond line's strength or at what the
    weakest ply takes there: f_v along its grain (a UD stack), the rolling shear across
    it (0-90), in between at an angle (+-45, a quadratic interaction)."""
    from bridgesim.sim.idealize import interlaminar_strength

    fv, fr = 6.0, 2.0
    assert interlaminar_strength([0, 0, 0], fv, fr, 10.0) == pytest.approx(6.0)
    assert interlaminar_strength([0, 0, 0], fv, fr, 4.0) == pytest.approx(4.0)  # the bond
    assert interlaminar_strength([0, 90], fv, fr, 10.0) == pytest.approx(2.0)
    a45 = interlaminar_strength([45, -45], fv, fr, 10.0)
    assert a45 == pytest.approx(1 / math.hypot(math.sqrt(0.5) / fv, math.sqrt(0.5) / fr))
    assert fr < a45 < fv


def test_members_split_on_their_shear_over_a_section_depth():
    """A member splits where its shear, averaged over a section depth along it, reaches
    the strength: a peak on one element next to a joint (finer the mesh, sharper the
    peak) does not split it, the same shear over the whole member does."""
    from bridgesim.cad.synthetic import box
    from bridgesim.sim import solver as S_
    from bridgesim.sim.idealize import Overrides, build_model

    bodies = [box("Pier L", (-550, 0, -40), (-540, 150, 40)),
              box("Pier R", (540, 0, -40), (550, 150, 40)),
              box("Beam (14-ply UD)", (-550, 150, -25), (550, 179.4, 25))]
    m = build_model(bodies, MAT, SP, Overrides.from_dict(
        {"bodies": {"Pier*": {"kind": "block"}}}), seed=None, mesh_scale=0.5)
    st = S_.initial_state(m, 0.0)
    bar = next(b for b in m.bodies if b.kind == "bar")
    els = np.asarray(bar.beams)
    mid = els[len(els) // 2]
    assert max(bar.section_mm) > 2 * m.b_L0[mid]  # several elements within a depth
    st.b_rat3[:] = 0.0
    st.b_rat3[els] = 0.6
    st.b_rat3[mid] = 1.5  # one element over its strength
    S_.split_members(m, st)
    assert not st.b_shear.any()
    st.b_rat3[els] = 1.2  # the whole member
    S_.split_members(m, st)
    assert st.b_shear[els].all() and (st.b_s3[els] < 1).all()


def test_frame_plate_bends_like_a_beam(tmp_path):
    """A strip of plate standing on its edge on two piers (in-plane bending, ~2.5 shell
    elements across) becomes a frame member and has beam theory's stiffness and first
    fibre limit; the same strip kept as constant-strain shells comes out ~70 % too stiff.
    A curved rib becomes one member chain following its curve."""
    from shapely.geometry import Polygon

    from bridgesim.cad.synthetic import box, prism
    from bridgesim.sim import study
    from bridgesim.sim.skeleton import medial_chains

    H, T = 40.0, 8.0
    cad = tmp_path / "strip.stl"
    write_stl([box("Pier L", (-550, 0, -40), (-540, 150, 40)),
               box("Pier R", (540, 0, -40), (550, 150, 40)),
               box("Strip", (-550, 150, -T / 2), (550, 150 + H, T / 2))], cad)
    names = {b.name: b.kind for b in study.inspect(cad).bodies}
    strip = next(n for n, k in names.items() if k == "frame")  # found without asking
    E, G, I = E_SIM, G_SIM, T * H**3 / 12
    Ls = 1090.0
    a = Ls / 2 - 100  # the plate bears at its two edges
    d = 0.5 * a**2 * (3 * Ls - 4 * a) / (6 * E * I) + 0.5 * a / (5 / 6 * G * T * H)
    k = {}
    for kind in ("frame", "shell"):
        ov = {"bodies": {n: ({"layup": "14-ply UD", "butt_stagger_mm": 0, "kind": kind}
                             if n == strip else {"kind": "block"}) for n in names}}
        rep = study.screen(cad, overrides=ov)
        k[kind] = rep.stiffness_kgf_per_mm * 9.80665 * d
        if kind == "frame":
            lim = {lm.mode: lm.load_kgf for lm in rep.limits if lm.part == strip}
            S = T * H**2 / 6
            assert lim["compression"] == pytest.approx(
                MAT.f_c_MPa.value * S / (0.5 * a) / 9.80665, rel=0.03)
    assert k["frame"] == pytest.approx(1.0, rel=0.03)
    assert k["shell"] > 1.4  # in-plane bending locks with ~2.5 elements across
    # a semicircular rib 30 mm wide: one chain along its mid-line
    th = np.linspace(0, np.pi, 60)
    outer = np.c_[500 * np.cos(th), 300 * np.sin(th)]
    inner = np.c_[470 * np.cos(th[::-1]), 270 * np.sin(th[::-1])]
    ch = medial_chains(Polygon(np.vstack([outer, inner])))
    assert len(ch) == 1 and ch[0].width.mean() == pytest.approx(30, rel=0.15)
    ellipse = np.pi * (3 * (485 + 285) - np.sqrt((3 * 485 + 285) * (485 + 3 * 285))) / 2
    assert ch[0].length == pytest.approx(ellipse - 30, rel=0.05)
    from bridgesim.sim.idealize import _frame_chains, analyse_shape
    from bridgesim.sim.materials import parse_layup

    rib = prism("Rib (6-ply UD)", list(np.vstack([outer, inner])), 0.0, 8.0)
    sh = analyse_shape(rib, parse_layup(rib.name), None)
    assert sh.kind == "shell" and len(_frame_chains(sh)) == 1  # built as a frame


def test_the_built_bridge_leans_without_stress():
    """The global imperfection: a seeded bridge leans out of its plane by up to
    scatter.lean_rad (height x angle), the same at every node and glue point, so every
    glued pair still coincides (no initial stress); the perfect bridge stands upright."""
    from bridgesim.sim.idealize import build_model

    sp = SP.updated({"scatter.lean_rad": 0.01, "scatter.bow_L_over": 1e6,
                     "scatter.support_level_mm": 0.0, "scatter.placement_mm": 0.0})
    bodies = simple_beam_bridge()
    m0 = build_model(bodies, MAT, sp, seed=None)
    leans = []
    for seed in (1, 2, 3):
        m = build_model(bodies, MAT, sp, seed=seed)
        dz = m.X[:, 2] - m0.X[:, 2]
        y = m0.X[:, 1] - min(float(b.mesh.vertices[:, 1].min()) for b in bodies)
        sel = y > 10
        lean = float(np.median(dz[sel] / y[sel]))
        assert np.allclose(dz, lean * y, atol=2e-3) and abs(lean) <= 0.01  # bows ~1e-3
        assert np.allclose(m.X[:, :2], m0.X[:, :2], atol=2e-3)
        pos = m.X[m.a_node] + m.a_r0
        assert np.abs(pos[m.c_a] - pos[m.c_b]).max() < 1e-9  # glue pairs still together
        leans.append(lean)
    assert len(set(np.round(leans, 6))) == 3 and max(abs(x) for x in leans) > 1e-3


def test_mcp_outputs_match_their_published_schemas(tmp_path):
    """Every tool but get_schema publishes a typed output schema, and what the tools
    return (the briefs by default, the full reports on request, job states) validates
    against it; `bridgesim schema crush_brief` prints the brief's schema."""
    pytest.importorskip("mcp")
    jsonschema = pytest.importorskip("jsonschema")
    import asyncio

    from typer.testing import CliRunner

    from bridgesim.cli import app
    from bridgesim.sim.mcp_server import build_server

    srv = build_server()
    tools = asyncio.run(srv.list_tools())
    tools = {t.name: (getattr(t, "output_schema", None) or getattr(t, "outputSchema", None))
             for t in getattr(tools, "tools", tools)}
    for name, schema in tools.items():
        if name != "get_schema":
            assert len(schema.get("properties", {})) > 2, name

    def call(tool, **kw):
        out = asyncio.run(srv.call_tool(tool, kw))
        out = out[1] if isinstance(out, tuple) else getattr(out, "structured_content", out)
        jsonschema.validate(out, tools[tool])
        return out

    stl = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), stl)
    assert call("inspect_bridge", cad_path=str(stl))["kind"] == "inspect_brief"
    assert call("inspect_bridge", cad_path=str(stl), brief=False)["kind"] == "inspect"
    assert call("screen_bridge", cad_path=str(stl))["kind"] == "screen"
    assert call("job_status", out_dir=str(tmp_path / "nothing"))["status"] == "unknown"
    r = CliRunner().invoke(app, ["schema", "crush_brief"])
    assert r.exit_code == 0 and json.loads(r.stdout)["title"] == "CrushBrief"


# ------------------------------------------------------------- round 9: exact, measured


def test_suggested_commands_keep_the_run_context(tmp_path):
    """Following a next_steps command never changes the model: a Z-up STL inspected with
    a material file gets commands carrying --material and --up-axis (and absolute output
    folders), which run as they are; a crush's commands also keep its seed, and ab keeps
    its fidelity."""
    import shlex
    import shutil

    from typer.testing import CliRunner

    from bridgesim.cli import app
    from bridgesim.sim import study
    from bridgesim.sim.report import NextStep
    from bridgesim.sim.study import _context, _with_context

    bodies = simple_beam_bridge()
    for b in bodies:  # stand it on its side: Z up
        b.mesh.vertices[:] = b.mesh.vertices[:, [0, 2, 1]] * np.array([1, -1, 1])
    cad = tmp_path / "bz.stl"
    write_stl(bodies, cad)
    mat = tmp_path / "my_material.yaml"
    shutil.copy(ROOT / "materials" / "popsicle_birch.yaml", mat)
    rep = study.inspect(cad, material=str(mat), up="z")
    clis = [s.cli for s in rep.next_steps if s.cli]
    assert clis and all(f"--material {mat}" in c and "--up-axis z" in c for c in clis)
    for c in clis:
        argv = shlex.split(c)
        if "--out" in argv:
            assert Path(argv[argv.index("--out") + 1]).is_absolute()
        _parse_cli(argv[1:])
    screen = next(c for c in clis if c.startswith("bridgesim screen"))
    r = CliRunner().invoke(app, shlex.split(screen)[1:])
    assert r.exit_code == 0, r.stderr
    steps = _with_context([NextStep(action="rerun", why="", cli="bridgesim crush a.stl --json"),
                           NextStep(action="strengthen", why="",
                                    cli="bridgesim ab a.stl a.stl -n 3 --json")],
                          _context(str(mat), up="z", seed=3, fidelity="quick"))
    assert "--seed 3" in steps[0].cli and "--fidelity quick" in steps[1].cli
    assert "--seed" not in steps[1].cli  # ab runs its own seeds


def test_reported_load_has_no_lag():
    """The reported load (centred smoothing) follows a rising load without the online
    filter's lag and removes the ringing: a softening rise to 2000 N at 5 mm of jack
    travel with a 270 Hz ring, then a sudden 40 % drop. Its peak is the true one within
    2 %, at the right place; the online reading lags the rise."""
    from bridgesim.sim.crush import LOAD_CELL_MM, reported_load

    v = 100.0  # mm/s
    t = np.arange(0, 0.06, 3e-5)
    x = v * t  # jack travel
    true = np.where(x < 5.0, 2000.0 * (1 - (1 - x / 5.0) ** 2), 1200.0)
    raw = true + 150.0 * np.sin(2 * np.pi * 270 * t)
    rep = reported_load(t, raw, v)
    on = np.zeros_like(raw)  # the online one-pole filter
    for i in range(1, len(t)):
        a = min(1.0, (t[i] - t[i - 1]) * v / LOAD_CELL_MM)
        on[i] = on[i - 1] + a * (raw[i] - on[i - 1])
    rise = (x > 0.5) & (x < 4.0)
    assert np.abs(rep[rise] - true[rise]).max() < 0.02 * 2000  # no lag, ring removed
    assert abs(rep[0]) < 0.01 * 2000  # no bias where the record starts
    assert (true[rise] - on[rise]).mean() > 0.05 * 2000  # the online reading lags
    i = int(np.argmax(rep))
    assert rep[i] == pytest.approx(2000.0, rel=0.02) and abs(x[i] - 5.0) < 0.5
    # holds (the jack slowed to 2 % for 20 ms, the speed ramping over 2 ms): smoothed
    # along the jack travel, the load neither keeps rising at constant travel (beyond
    # the jack's creep) nor jumps; smoothed along time it would catch up in the hold
    th = np.clip(np.minimum(t - 0.03, 0.05 - t) / 0.002, 0, 1)  # 0 -> 1 into the hold
    v2 = v * (1 - 0.98 * th)
    x2 = np.cumsum(v2) * np.r_[0, np.diff(t)]
    raw2 = 400.0 * x2 + 25.0 * np.sin(2 * np.pi * 270 * t)  # ringing ~2 % (Bridge_v6)
    sel = th > 0.99
    creep = (400 * x2[sel].max() - 400 * x2[sel][0]) / (400 * x2[sel][0])
    rt = reported_load(t, raw2, v)
    assert (rt[sel].max() - rt[sel][0]) / rt[sel][0] - creep > 0.01  # time: catches up
    r2 = reported_load(t, raw2, v, x2)
    assert (r2[sel].max() - r2[sel][0]) / r2[sel][0] - creep < 0.003
    k = 400.0  # N/mm: no jump larger than the stiffness x 0.01 mm + 1 % within 0.01 mm
    for i in np.where(x2 > 0.5)[0]:
        j = np.searchsorted(x2, x2[i] + 0.01)
        if j < len(x2):
            assert abs(r2[j] - r2[i]) <= k * 0.01 + 0.01 * r2[i] + 1e-6


def test_governing_is_the_break_that_started_the_fall():
    """Bridge_v6 seed 1 at quick fidelity: a deck-stringer joint let go at the peak (the
    load fell 5 % and came back), then the End Post joints went at 96 % of the peak and
    the load fell 40 %. Governing: the End Post joint; the deck joint competes. A member
    split in the same instant as a member fracture is put down to the fracture."""
    from bridgesim.sim.crush import KGF, Event, attribute

    def ev(t, kgf, kind, what):
        return Event(t, 100 * t, kgf * KGF, kind, what)

    seq = [ev(0.0944, 2313, "member_shear", "Deck Stringer - R Far"),
           ev(0.1124, 2399, "joint_failure", "Body 21531 | Deck Stringer - L Near"),
           ev(0.1142, 2256, "joint_cracking", "End Post - L Far | T1 Gusset - L Far"),
           ev(0.1142, 2256, "joint_failure", "End Post - L Far | T1 Gusset - L Far"),
           ev(0.1149, 2016, "member_shear", "Top Strut Inner - R")]
    gov, comp = attribute(seq, t_c=0.1139, peak=2405 * KGF, peak_travel=11.18,
                          speed_mm_s=200.0)
    assert gov.what == "End Post - L Far | T1 Gusset - L Far" and gov.kind == "joint_failure"
    assert "joint failure: Body 21531 | Deck Stringer - L Near" in comp
    assert not any("Top Strut" in c for c in comp)  # the cascade afterwards
    # nothing broke from t_c: the last survived break near the peak
    gov, _ = attribute(seq[:2], t_c=0.1139, peak=2405 * KGF, peak_travel=11.18,
                       speed_mm_s=200.0)
    assert gov.what == "Body 21531 | Deck Stringer - L Near"
    split = [ev(0.2, 2000, "member_shear", "Chord - L"),
             ev(0.2005, 1990, "member_fracture", "Chord - R")]
    gov, _ = attribute(split, t_c=0.199, peak=2000 * KGF, peak_travel=20.0,
                       speed_mm_s=100.0)
    assert gov.kind == "member_fracture"


def test_ab_says_what_governed_and_where_the_failure_moved():
    from bridgesim.sim.study import summarize_pairs

    def run(seed, F, gov):
        return {"seed": seed, "Fu_kgf": F, "mass_kg": 2.0, "efficiency_kgf_per_kg": F / 2,
                "first_failure": "joint failure: Hub Tie Strap - Far | Hub Spreader - R",
                "governing": gov, "trustworthy": True}

    a = [run(1, 2000, "member shear: Deck Stringer (3-ply) - R Far"),
         run(2, 1800, "joint failure: End Post - L Near | T1 Cover - L Near")]
    b = [run(1, 2050, "member shear: Deck Stringer (3-ply) - L Near"),
         run(2, 2100, "member shear: Deck Stringer (3-ply) - R Near")]
    rep = summarize_pairs(a, b, "A", "B", "quick")
    assert rep.governing_b == "member shear: Deck Stringer (3-ply)"  # by part with copies
    assert rep.runs_a[1].governing.startswith("joint failure: End Post")
    assert len(rep.failure_moved) == 1 and rep.failure_moved[0].startswith("seed 2")
    assert "moved" in rep.verdict


def test_diagnose_ranks_what_limits_the_bridge(tmp_path, monkeypatch):
    """diagnose strengthens each part that broke near the peak in turn on the same seeds
    and ranks them by the rise of F_u (the crushes are stubbed here: the ranking, the
    what-ifs and the verdict are what is tested)."""
    import yaml

    from bridgesim.sim import study

    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), cad)
    joint = {"kind": "joint_failure", "what": "End Post - L | T1 Cover - L",
             "parts": ["End Post - L", "T1 Cover - L"]}
    shear = {"kind": "member_shear", "what": "Stringer - R", "parts": ["Stringer - R"]}
    gains = {"base": 1.0, "c0": 1.0, "c1": 1.15}  # strengthening the joint (c1) helps

    def fake_pool(args, n_runs, processes, progress, label):
        out = []
        for tag, a in args:
            ov = a[1] or {}
            first = "c0" if "bodies" in ov else "c1" if "glue" in ov else "base"
            out.append({"tag": tag, "seed": a[5], "Fu_kgf": 1000.0 * gains[first] + a[5],
                        "first_failure": "x", "governing": "member shear: Stringer - R",
                        "trustworthy": True, "mass_kg": 2.0, "efficiency_kgf_per_kg": 500.0,
                        "at_peak": [shear, joint] if a[5] == 1 else [shear]})
        return out

    monkeypatch.setattr(study, "_pool_runs", fake_pool)
    rep = study.diagnose(cad, n=2, candidates=2, out_dir=tmp_path / "d")
    assert [c.kind for c in rep.candidates] == ["joint failure", "member shear"]
    assert rep.candidates[0].Fu_change == pytest.approx(0.15, abs=0.001)
    assert rep.candidates[1].seen == 2 and rep.candidates[1].Fu_change == pytest.approx(0)
    assert rep.limiting == "End Post | T1 Cover" and "Strengthen End Post" in rep.verdict
    wi = yaml.safe_load((tmp_path / "d" / "whatif.yaml").read_text())
    assert wi["glue"] == {"End Post* | T1 Cover*": 1.5}
    assert str((tmp_path / "d" / "whatif.yaml").resolve()) in rep.next_steps[0].cli


def test_bridge_calibration_fits_the_median_over_seeds(tmp_path, monkeypatch):
    """calibrate-bridge crushes several seeds per step and fits their median, so the
    strength factor does not absorb one seed's imperfections (crushes stubbed: F_u is
    proportional to the strength factor, the seeds 10 % weak, nominal, 30 % strong)."""
    from bridgesim.sim import study

    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), cad)
    (tmp_path / "test.csv").write_text("test,load_kgf,event,notes\nT1,1200,failure,"
                                       "deck\n", encoding="utf-8")
    f_b = float(MAT.f_b_MPa.value)
    seen = []

    def fake_pool(args, n_runs, processes, progress, label):
        out = []
        for tag, a in args:
            fS = a[1]["params"]["material.f_b_MPa"] / f_b
            seen.append(a[5])
            out.append({"tag": tag, "seed": a[5], "governing": "joint failure: A | B",
                        "Fu_kgf": 1000.0 * fS * {1: 0.9, 2: 1.0, 3: 1.3}[a[5]],
                        "secant_deflection": float("nan"), "secant_travel": float("nan")})
        return out

    monkeypatch.setattr(study, "_pool_runs", fake_pool)
    cal = study.calibrate_bridge(cad, tmp_path / "test.csv", fidelity="quick")
    assert cal.seeds == [1, 2, 3] and cal.seed is None and set(seen) == {1, 2, 3}
    assert cal.steps[0].Fu_kgf == 1000.0 and cal.steps[0].Fu_kgf_runs == [900, 1000, 1300]
    assert cal.converged and cal.strength_factor == pytest.approx(1.2, rel=0.01)
    assert cal.steps[-1].first_failure == "joint failure: A | B"


def test_coupon_files_are_found_by_name_or_columns(tmp_path):
    from bridgesim.sim.study import coupon_files

    (tmp_path / "tension_feb.csv").write_text(
        "specimen,width_mm,thickness_mm,failure_load_N\nT1,10,2,1600\n")
    (tmp_path / "pulls.csv").write_text(
        "specimen,width_mm,thickness_mm,failure_load_N\nT1,10,2,1600\n")
    (tmp_path / "bend.csv").write_text(
        "specimen,span_L_mm,width_b_mm,thickness_t_mm,load_F_N,deflection_mm,"
        "failure_load_N\nB1,100,10,2,1,0.5,20\n")
    (tmp_path / "laps.csv").write_text("specimen,overlap_mm,width_mm,faces,failure_load_N\n"
                                       "L1,20,10,1,400\n")
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n")
    (tmp_path / "bridge.csv").write_text("test,load_kgf\nT,100\n")
    found, warn = coupon_files(tmp_path)
    assert [f.name for f in found["stick_tension"]] == ["tension_feb.csv"]
    assert [f.name for f in found["stick_bending"]] == ["bend.csv"]
    assert [f.name for f in found["glue_joint"]] == ["laps.csv"]
    assert any("pulls.csv" in w and "tension" in w for w in warn)
    assert any("notes.csv" in w for w in warn) and not any("bridge.csv" in w for w in warn)
    # a short-beam shear file, found by its columns, sets the interlaminar strength
    import shutil

    from bridgesim.sim.study import calibrate

    sb = tmp_path / "sbs"
    sb.mkdir()
    (sb / "laminate_tests.csv").write_text(
        "specimen,plies,span_mm,width_mm,thickness_mm,peak_load_N\n"
        "S1,4,40,10,8,640\nS2,4,40,10,8,600\n")
    assert [f.name for f in coupon_files(sb)[0]["short_beam"]] == ["laminate_tests.csv"]
    _, _, params = calibrate(sb)
    assert params["glue.interlaminar_MPa"] == pytest.approx(0.75 * 620 / 80)
    shutil.rmtree(sb)


def test_torsion_shears_each_plane_by_saint_venant():
    """A twisted rectangle: the peak shear (Roark) on the planes normal to its short side,
    0.742-1 of it on the others (Timoshenko & Goodier: 0.795 at 2:1). A flat laminate's
    glue lines (normal to its thickness) take the short-side value, not the peak."""
    from bridgesim.sim.materials import rect_torsion

    J, W2, W3 = rect_torsion(10.0, 10.0)  # square: tau = T / (0.208 a^3), J = 0.1406 a^4
    assert 1 / W2 == pytest.approx(1 / (0.208 * 1000), rel=0.01) and W2 == W3
    assert J == pytest.approx(0.1406 * 1e4, rel=0.01)
    J, W2, W3 = rect_torsion(20.0, 10.0)  # 2:1, long side along e2
    assert 1 / W2 == pytest.approx(1 / (0.246 * 20 * 100), rel=0.01)  # alpha = 0.246
    assert W2 / W3 == pytest.approx(0.795, rel=0.01)
    assert rect_torsion(10.0, 20.0)[1:] == pytest.approx((W3, W2))  # stacked the other way
    J, W2, W3 = rect_torsion(100.0, 2.0)  # a thin strip: tau = 3 T / (b t^2)
    assert 1 / W2 == pytest.approx(3 / (100 * 4), rel=0.02)
    assert W2 / W3 == pytest.approx(0.742, rel=0.01)


def test_plies_resist_a_split_across_the_width_with_their_fibres():
    """A member splits across its width (along it, through every ply) at f_v when its
    plies run along or across it, at the fibres' strength (netting: min(f_t, f_c) / 2)
    when they cross the plane at 45 degrees, in a quadratic interaction between."""
    from bridgesim.sim.materials import inplane_split_strength

    assert inplane_split_strength([0, 0, 0], 6.0, 15.0) == pytest.approx(6.0)
    assert inplane_split_strength([0, 90], 6.0, 15.0) == pytest.approx(6.0)
    assert inplane_split_strength([45, -45], 6.0, 15.0) == pytest.approx(15.0)
    assert 6.0 < inplane_split_strength([30, -30], 6.0, 15.0) < 15.0


def test_frame_plies_follow_the_sheet_or_the_arm():
    """A cross-ply or angle-ply frame is a sheet cut to shape: its plies keep the plate's
    axis, so an arm at 45 degrees to a +-45 plate is 0/90 (stiffer); a UD frame is built
    arm by arm, its sticks along each arm."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from bridgesim.cad.synthetic import prism
    from bridgesim.sim.idealize import Overrides, build_model

    outline = unary_union([LineString([(-400, 0), (400, 0)]).buffer(6, cap_style=2)] + [
        LineString([(0, 0), (sx * 60, sz * 60)]).buffer(6, cap_style=2)
        for sx in (-1, 1) for sz in (-1, 1)])  # symmetric: its long axis is x
    pts = np.asarray(outline.exterior.coords)[:-1]
    ply = Ply.from_material(MAT, SP)
    E, qs = {}, {}
    for lay in ("4-ply +-45", "4-ply UD", "4-ply +-45 arm"):
        name = f"Rib ({lay.replace(' arm', '')})"
        plate = prism(name, pts, 20.0, 28.0)
        ov = Overrides.from_dict({"bodies": {"Rib*": {"grain": "arm"}}} if "arm" in lay
                                 else {})
        m = build_model(simple_beam_bridge() + [plate], MAT, SP, ov, seed=None)
        rib = next(b for b in m.bodies if b.name.startswith("Rib"))
        assert rib.kind == "frame"
        qs[lay] = {q["id"] for q in m.questions}
        d = m.X[m.b_j[rib.beams]] - m.X[m.b_i[rib.beams]]
        diag = np.abs(d[:, 0]) < 0.9 * np.linalg.norm(d, axis=1)
        E[lay] = (float(np.median(m.b_E[rib.beams][~diag])),
                  float(np.median(m.b_E[rib.beams][diag])))
    e45 = laminate(ply, [45, -45, 45, -45], 8.0).E_x  # (build_model: the file's E)
    e090 = laminate(ply, [0, 90, 0, 90], 8.0).E_x
    assert E["4-ply +-45"][0] == pytest.approx(e45, rel=0.05)  # along the plate's axis
    assert E["4-ply +-45"][1] == pytest.approx(e090, rel=0.05)  # the 45-degree arm
    assert E["4-ply UD"][0] == pytest.approx(E["4-ply UD"][1], rel=0.01)  # along each arm
    # which: asked (the sheet's axis is assumed); laid along each arm, both arms +-45
    assert "frame_plies:Rib (4-ply +-45)" in qs["4-ply +-45"]
    assert not any(q.startswith("frame_plies") for q in qs["4-ply UD"] | qs["4-ply +-45 arm"])
    assert E["4-ply +-45 arm"] == pytest.approx((e45, e45), rel=0.05)


def test_stick_scatter_is_correlated_and_averaged_by_lamination():
    """Each body's stiffness and strength factors: the strength has the stated CoV and
    the stated correlation with the stiffness; a member of n plies averages its partly
    alike sticks (CoV x sqrt((1 + (n - 1) rho) / n)); the amplitude scales both."""
    from bridgesim.sim.idealize import stick_factors

    def stats(n, sp):
        f = np.log(np.array([stick_factors(0, f"Body {i}", n, sp) for i in range(4000)]))
        cov = np.sqrt(np.exp(f.var(0)) - 1)
        return cov, float(np.corrcoef(f.T)[0, 1])

    cov, r = stats(1, SP)
    assert cov == pytest.approx([SP["scatter.E_cov"], SP["scatter.strength_cov"]], rel=0.05)
    assert r == pytest.approx(SP["scatter.E_strength_corr"], abs=0.03)
    cov10, r10 = stats(10, SP)
    k = math.sqrt((1 + 9 * SP["scatter.ply_corr"]) / 10)
    assert cov10 == pytest.approx(cov * k, rel=0.03) and r10 == pytest.approx(r, abs=1e-9)
    cov_a, _ = stats(1, SP.updated({"scatter.amplitude": 0.1}))
    assert cov_a == pytest.approx(0.1 * cov, rel=0.05)
    assert stick_factors(None, "Body 1", 1, SP) == (1.0, 1.0)


def test_wood_softens_before_it_crushes():
    """A short stack in compression: straight to the proportional limit (0.65 f_c), then
    bending over to f_c at (2 - 0.65) f_c / E, as wood does; elastic-perfectly plastic
    with the limit at 1."""
    for kap in (0.65, 1.0):
        m = lab.blank_model(np.array([[0.0, 0, 0], [100.0, 0, 0]]))
        lab.add_beam_line(m, [0, 1], lab.BeamSpec(b=10, d=10, E=7100, ft=40, fc=30,
                                                   fv=1e9))
        m.params["fc_proportional"] = kap
        r = lab.drive(m, {0: "xyzr", 1: "yzr"}, 1, 0, 5.0, max_disp=1.0, sign=-1.0,
                      stop_drop=0.0)
        ec0, A = 30 / 7100, 100.0
        f = np.interp([0.5 * ec0 * 100, ec0 * 100], r.disp, r.force)
        assert f[0] == pytest.approx(0.5 * 30 * A, rel=0.02)  # linear below the limit
        assert r.peak == pytest.approx(30 * A, rel=0.02)
        u_pk = float(r.disp[np.argmax(r.force)])
        if kap < 1:
            # at f_c / E: on the curve, 1 - 0.35 (1 - 0.5)^2 = 0.9125 of f_c
            assert f[1] == pytest.approx(0.9125 * 30 * A, rel=0.03)
            assert u_pk == pytest.approx(1.35 * ec0 * 100, rel=0.15)
        else:
            assert f[1] == pytest.approx(30 * A, rel=0.02)
