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

    # the simulation's own glue and stick values (the crush's), not the coupon's
    # defaults: the factor has to be recomputed whenever they change
    from bridgesim.sim.study import coupon_K_s

    kw = {"Kn": SP["glue.K_n_N_per_mm3"],
          "Ks": coupon_K_s(SP, 2.0, SP["wood.E_placeholder_MPa"] / 16.0),
          "GIc": SP["glue.G_Ic_N_per_mm"], "GIIc": SP["glue.G_IIc_N_per_mm"],
          "E": SP["wood.E_placeholder_MPa"], "onset": SP["glue.prepeak_onset"],
          "peak_slip": SP["glue.peak_slip_ratio"]}

    def tau_avg(tp, **k):
        peel = min(SP["glue.peel_to_shear"] * tp, SP["wood.f_t90_MPa"]) / tp
        return lab.lap_shear(tp, peel, **kw, **k).peak / 200

    P = SP["glue.tau_peak_over_avg"]
    assert tau_avg(2.0 * P) == pytest.approx(2.0, rel=0.02)
    for n_ov, one, nodes in ((2, True, 1), (1, False, 2), (2, False, 3), (4, False, 5)):
        f = 1.0 + (P - 1.0) * (1.0 - math.exp(-2.2 * (nodes - 1)))
        assert tau_avg(2.0 * f, n_ov=n_ov, one_node=one) == pytest.approx(2.0, rel=0.05), \
            nodes


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
    # (since round 28 the piers are cells of wood: the planes between them are patches of
    # kind block, not joints)
    joints = [k for k, p in enumerate(m.patches) if p.kind != "block"]
    assert len(joints) == 6  # 2 stringers on 2 piers + deck on 2 stringers
    total = sum(m.patches[k].area_mm2 for k in joints)
    assert total == pytest.approx(4 * 50 * 6 + 2 * 1100 * 6, rel=0.01)
    # glue area is fully distributed over the glue points
    assert m.c_area[np.isin(m.c_patch, joints)].sum() == pytest.approx(total, rel=1e-6)


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def synthetic_crush(tmp_path_factory):
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    d = tmp_path_factory.mktemp("crush")
    write_stl(simple_beam_bridge(), d / "bridge.stl")
    # the reporting pipeline at a fixed jack speed (the bridge's own period would slow
    # this soft bridge 4x: test_the_jack_slows_for_a_bridge_that_rings_slowly)
    # (since round 19 its stringers' gapped butt joints let them crush first, a plateau,
    # and it collapses past ~30 mm of travel)
    rep, res = study.crush(d / "bridge.stl", seed=0, out_dir=d / "out",
                           options=CrushOptions(speed_mm_s=150, max_travel_mm=45,
                                                auto_speed=False))
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
               box("Beam", (-550, 150, -25), (550, 179.4, 25))], tmp_path / "beam.stl",
              names=True)  # named, so the overrides below apply (no butt joints)
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
               box("Beam", (-550, 150, -25), (550, 192, 25))], cad)  # breaks < 50 mm
    srv = build_server()

    def call(tool, **kw):
        out = asyncio.run(srv.call_tool(tool, kw))
        out = out[1] if isinstance(out, tuple) else getattr(out, "structured_content", out)
        return out

    ins = call("inspect_bridge", cad_path=str(cad), brief=False)
    assert ins["questions"] and ins["next_steps"]
    # the designer's answers: what each body is (an STL has no names: the agent reads
    # them, and where each body is, from the full inspect report)
    bodies = {b["name"]: ({"layup": "20-ply UD", "butt_stagger_mm": 0} if b["kind"] == "bar"
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
    # the real test reaches the peak after the jack's travel at its own pace (10 mm/min)
    pace = SP["crusher.test_rate_mm_per_min"] / 60.0
    assert rep.test_time_at_Fu_s == pytest.approx(b["peak"]["jack_travel_mm"] / pace,
                                                  rel=0.1, abs=0.5)
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
    # sticks glued end to end and pulled: 20 N on 20 mm^2, over the sticks' measured f_t
    assert params["glue.end_grain_tension_factor"] == pytest.approx(1.0 / 82.5, rel=1e-6)
    # the lap's slope over its area, the two sticks' shear through their thickness taken
    # out (the crush adds each joint's own back)
    k_lap = 2000 / 200
    t_st, G = MAT.stick.thickness_mm, MAT.G_MPa.value
    assert params["glue.K_s_N_per_mm3"] == pytest.approx(
        1 / (1 / k_lap - 2 * t_st / (3 * G)), rel=1e-6)
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
    # a team's own words for Bridge_v10's node M, and the body name from the part column
    gov = "Knee-Diagonal R Far (13-ply UD) | Stub Far (13-ply UD)"
    assert _part_match("diagonal foot at the centre node, far side, right", gov) == 1.0
    assert _part_match("right diagonal, far truss, at node M", gov) == 1.0
    assert _part_match("Knee-Diagonal R Far*", gov) == 1.0
    assert _part_match("Knee-Diagonal R Far* | Stub Far*", gov) == 1.0
    assert _part_match("Knee-Diagonal L Near*", gov) == 0.5  # the same part, other side


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
                                                                        "butt_zones_mm",
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
    from bridgesim.sim.study import (
        PLACEHOLDER_COV,
        PLACEHOLDER_PRIORS,
        draw_placeholders,
        placeholders,
    )

    ph = placeholders(MAT, SP)
    assert "material.glue.tau_g_MPa" in ph and "material.E_MPa" in ph
    user = {"params": {"floss.strands": 6}, "bodies": {"X*": {"glue_factor": 0.8}}}
    a, b = draw_placeholders(user, ph, 1), draw_placeholders(user, ph, 1)
    assert a == b and a != draw_placeholders(user, ph, 2)
    assert a["params"]["floss.strands"] == 6 and a["bodies"] == user["bodies"]
    assert set(a["params"]) == set(ph) | {"floss.strands"}
    # each placeholder is drawn with its own prior (the literature or craft-stick
    # range): the glue far wider than the sticks' modulus; others with PLACEHOLDER_COV
    for key in ("material.E_MPa", "material.glue.tau_g_MPa"):
        logs = [math.log(draw_placeholders({}, ph, s)["params"][key] / ph[key])
                for s in range(300)]
        assert abs(np.mean(logs)) < 0.06
        assert np.std(logs) == pytest.approx(PLACEHOLDER_PRIORS[key], rel=0.15)
    assert PLACEHOLDER_PRIORS["material.glue.tau_g_MPa"] > PLACEHOLDER_PRIORS[
        "material.E_MPa"] and 0 < PLACEHOLDER_COV < 1
    # the wood's strength is drawn with its stiffness (r ~ 0.7), the glue on its own
    lE, lft, ltg = (np.array([math.log(draw_placeholders({}, ph, s)["params"][k] / ph[k])
                              for s in range(300)])
                    for k in ("material.E_MPa", "material.f_t_MPa",
                              "material.glue.tau_g_MPa"))
    assert np.corrcoef(lE, lft)[0, 1] == pytest.approx(0.7, abs=0.1)
    assert abs(np.corrcoef(lE, ltg)[0, 1]) < 0.15
    measured = MAT.model_copy(update={"E_MPa": MAT.E_MPa.model_copy(
        update={"source": "measured"})})
    assert "material.E_MPa" not in placeholders(measured, SP)


def test_montecarlo_placeholder_draws_are_a_latin_hypercube():
    """With the number of runs given, each placeholder's draws over the runs fall one in
    each of n equal-probability strata of its prior (so 8 runs cannot all draw the glue
    weak), the same for the same seed and n; the wood's strengths still follow E."""
    from statistics import NormalDist

    from bridgesim.sim.study import PLACEHOLDER_PRIORS, draw_placeholders, placeholders

    ph = placeholders(MAT, SP)
    n = 8
    runs = [draw_placeholders({}, ph, s, n)["params"] for s in range(1, n + 1)]
    assert runs[0] == draw_placeholders({}, ph, 1, n)["params"]
    nd = NormalDist()
    for key in ("material.glue.tau_g_MPa", "material.E_MPa", "glue.weibull_m"):
        z = [math.log(r[key] / ph[key]) / PLACEHOLDER_PRIORS[key] for r in runs]
        strata = sorted(int(nd.cdf(v) * n) for v in z)
        assert strata == list(range(n)), key
        assert sum(r[key] < ph[key] for r in runs) == n // 2
    # f_t is drawn with E (r 0.7): not stratified itself, but still correlated with it
    lE = np.array([math.log(r["material.E_MPa"] / ph["material.E_MPa"]) for r in runs])
    lft = np.array([math.log(r["material.f_t_MPa"] / ph["material.f_t_MPa"])
                    for r in runs])
    assert np.corrcoef(lE, lft)[0, 1] > 0.3
    # without n (or a seed outside 1..n): the plain draw, as before
    assert draw_placeholders({}, ph, 9, n) == draw_placeholders({}, ph, 9)


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
    # knife-edge piers (2 mm), so the beam is pinned as beam theory has it: glued on a
    # wider pier standing on the table its ends are partly fixed (10 mm: 29 % stiffer)
    write_stl([box("Pier L", (-546, 0, -40), (-544, 150, 40)),
               box("Pier R", (544, 0, -40), (546, 150, 40)),
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
    # it splits along one glue line into halves (lever arms x 1/2), and again only while
    # the shear still reaches the strength, never past its separate plies (1/14)
    assert np.allclose(st.b_s3[els], 0.5)
    st.b_rat3[els] = 0.9
    S_.split_members(m, st)
    assert np.allclose(st.b_s3[els], 0.5)
    st.b_rat3[els] = 1.2
    for _ in range(6):
        S_.split_members(m, st)
    assert np.allclose(st.b_s3[els], 1.0 / 14)


def test_frame_plate_bends_like_a_beam(tmp_path):
    """A strip of plate standing on its edge on two knife-edge piers (in-plane bending,
    ~2.5 shell elements across) becomes a frame member and has beam theory's stiffness
    and first fibre limit; the same strip kept as constant-strain shells comes out ~20 %
    too stiff. (Glued on 10 mm piers its ends are partly fixed, frame and shell alike:
    the 70 % once reported here was mostly that.) A curved rib becomes one member chain
    following its curve."""
    from shapely.geometry import Polygon

    from bridgesim.cad.synthetic import box, prism
    from bridgesim.sim import study
    from bridgesim.sim.skeleton import medial_chains

    H, T = 40.0, 8.0
    cad = tmp_path / "strip.stl"
    write_stl([box("Pier L", (-546, 0, -40), (-544, 150, 40)),  # knife edges: pinned
               box("Pier R", (544, 0, -40), (546, 150, 40)),
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
    # in-plane bending locks with ~2.5 elements across: 1.09-1.2 (the knife edges' bearing
    # faces softer since round 25, Blass & Gorlacher's spread; three glue points a side
    # since round 26)
    assert k["shell"] > 1.08
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


def test_the_peak_before_a_break_is_read_where_it_was():
    """A load that rises straight to a break and falls 45 % there (a joint letting go at
    the peak, as on Bridge_v10): smoothed across the drop it read 9 % low and the break
    at a fraction of its load; cut at the break (the hold that starts there), the peak
    and the load at the break read within 1 %, and the ring is still removed."""
    from bridgesim.sim.crush import reported_load

    v = 100.0
    t = np.arange(0, 0.06, 3e-5)
    x = v * t
    tb = 0.037  # the break, at 3.7 mm
    true = np.where(t < tb, 500.0 * x, 0.55 * 500.0 * v * tb)
    raw = true + 20.0 * np.sin(2 * np.pi * 270 * t)
    across = reported_load(t, raw, v, x)
    cut = reported_load(t, raw, v, x, breaks=[tb])
    peak = 500.0 * v * tb
    assert across.max() < 0.93 * peak  # the window across the drop: ~9 % low
    assert cut.max() == pytest.approx(peak, rel=0.01)
    assert cut[np.searchsorted(t, tb) - 1] == pytest.approx(peak, rel=0.01)  # at it
    rise = (x > 0.5) & (x < 3.5)
    assert np.abs(cut[rise] - true[rise]).max() < 0.01 * peak


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
    # what the test can tell is measured from then on (montecarlo no longer redraws it):
    # a joint governed, so the glue's strength, not the members' (the adversarial review
    # of round 26); the model-form error stays until another test checks the fit
    assert "material.glue.tau_g_MPa" in cal.overrides["measured"]
    assert "material.f_b_MPa" not in cal.overrides["measured"]
    assert "material.f_c_MPa" not in cal.overrides["measured"]
    assert set(cal.overrides["measured"]) <= set(cal.overrides["params"])
    assert "uncertainty.model_cov" not in cal.overrides["params"]
    assert study._mechanism("member fracture: Chord") == "member"
    assert study._mechanism("bearing crushing: Post | Seat") == "bearing"


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


def test_a_fine_frame_keeps_its_elements():
    """Arm ends closer than a third of the arm's width are one junction, but the nodes
    along an arm never merge: a frame meshed finer than that keeps every element (it
    used to lose most of them below x0.6, at x0.5 on Bridge_v6's hub plates)."""
    from bridgesim.cad.synthetic import box
    from bridgesim.sim.idealize import Overrides, build_model

    bodies = [box("Pier L", (-550, 0, -40), (-540, 150, 40)),
              box("Pier R", (540, 0, -40), (550, 150, 40)),
              box("Strip", (-550, 150, -4), (550, 190, 4))]
    ov = Overrides.from_dict({"bodies": {"Strip": {"kind": "frame", "layup": "4-ply UD"},
                                         "Pier*": {"kind": "block"}}})
    for ms in (1.0, 0.3):
        m = build_model(bodies, MAT, SP, ov, seed=None, mesh_scale=ms)
        strip = next(b for b in m.bodies if b.name == "Strip")
        L = m.b_L0[strip.beams]
        h = 30.0 * ms
        assert strip.kind == "frame" and L.max() <= 1.1 * h, (ms, L.max())
        assert L.sum() == pytest.approx(1100 - 40, rel=0.05)  # the arm's centre line


def _one_glue_point(Ks=60.0, tau=1e6):
    """Two nodes, one glue point of 1 mm^2 between them (normal Y), for law checks."""
    m = lab.blank_model(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    m.mass[:] = 1.0
    m.rot_inertia[:] = 1.0
    m.a_node, m.a_r0 = np.array([0, 1]), np.zeros((2, 3))
    m.c_a, m.c_b, m.c_area = np.array([0]), np.array([1]), np.array([1.0])
    m.c_n0 = np.array([[0.0, 1.0, 0.0]])
    m.c_Kn, m.c_Ks = np.array([150.0]), np.array([Ks])
    m.c_tau, m.c_sig = np.array([tau]), np.array([tau])
    m.c_GIc, m.c_GIIc, m.c_mu = np.array([1e3]), np.array([1e3]), np.array([0.4])
    m.c_patch, m.c_breakable = np.array([0]), np.array([True])
    return m


@pytest.mark.parametrize("fast", [False, True])
def test_glue_creeps_in_test_time(fast):
    """A glue line held at a fixed slip relaxes like a Kelvin element in series with
    its spring: the shear force falls from K_s x slip toward K_s x slip / (1 + phi) with
    the time constant t_r / (1 + phi), in physical test time (the jack's travel at the
    real rate); with no test time passing (a hold) nothing creeps."""
    if fast:
        pytest.importorskip("numba")
    phi, tr, slip = 1.0, 600.0, 0.01
    m = _one_glue_point()
    m.params.update(glue_creep_factor=phi, glue_creep_time_s=tr)
    st = S.initial_state(m, jack_y=1e12)
    st.x[1, 0] = slip  # node 1 slid along X by the slip
    s = S.Settings(dt=1e-6, use_numba=fast)

    def force(dt_test, steps):
        F, Mo = np.zeros((2, 3)), np.zeros((2, 3))
        st.creep_dt = dt_test
        for _ in range(steps):
            F[:] = 0.0
            Mo[:] = 0.0
            S.internal_and_contact(m, st, s, F, Mo, update=True)
        return abs(F[1, 0])

    f0 = force(0.0, 3)
    assert f0 == pytest.approx(60.0 * slip, rel=1e-6)  # nothing creeps while the jack stands
    t = 300.0
    ft = force(t / 1000, 1000)  # 300 s of test time in 1000 steps
    expect = 60.0 * slip * (1 - phi / (1 + phi) * (1 - math.exp(-(1 + phi) * t / tr)))
    assert ft == pytest.approx(expect, rel=1e-3)
    fl = force(100.0, 200)  # long after: the final, crept stiffness K_s / (1 + phi)
    assert fl == pytest.approx(60.0 * slip / (1 + phi), rel=1e-3)
    m.params["glue_creep_factor"] = 0.0
    st = S.initial_state(m, jack_y=1e12)
    st.x[1, 0] = slip
    assert force(100.0, 50) == pytest.approx(60.0 * slip, rel=1e-6)  # no creep


@pytest.mark.parametrize("fast", [False, True])
def test_a_member_bent_past_a_radian_is_eroded_not_fractured(fast):
    """An element turned past 1 rad at an end is removed: reported as erosion, its
    fibres not counted as cracked through."""
    if fast:
        pytest.importorskip("numba")
    m = lab.blank_model(np.array([[0.0, 0, 0], [100.0, 0, 0]]))
    lab.add_beam_line(m, [0, 1], lab.BeamSpec(b=10, d=10, E=7100, ft=40, fc=30))
    st = S.initial_state(m, jack_y=1e12)
    c, sn = math.cos(1.2), math.sin(1.2)
    st.R[1] = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])  # 1.2 rad about Z
    F, Mo = np.zeros((2, 3)), np.zeros((2, 3))
    S.internal_and_contact(m, st, S.Settings(dt=1e-6, use_numba=fast), F, Mo, update=True)
    assert not st.b_alive[0] and not st.b_fractured[0]


def test_platform_bearing_follows_the_grain():
    """A part standing on the platform bears by the angle of its plies' grain to the
    vertical (Hankinson): a standing +-45 plate between a UD plate standing on its
    end grain and one lying across it."""
    from bridgesim.cad.synthetic import box
    from bridgesim.sim.idealize import build_model

    fm = {}
    for lay in ("4-ply UD", "4-ply +-45"):
        bodies = [box(f"Leg L ({lay})", (-550, 0, -40), (-540, 150, 40)),
                  box(f"Leg R ({lay})", (540, 0, -40), (550, 150, 40)),
                  box("Beam (14-ply UD)", (-550, 150, -25), (550, 179.4, 25))]
        m = build_model(bodies, MAT, SP, seed=None)
        sup = np.asarray(m.p_support, bool)
        fm[lay] = float(np.asarray(m.params["fmax_p"])[sup].sum())
    ply = Ply.from_material(MAT, SP)
    from bridgesim.sim.idealize import hankinson

    assert fm["4-ply +-45"] / fm["4-ply UD"] == pytest.approx(
        hankinson(ply.f_c, ply.f_c90, math.sqrt(0.5)) / ply.f_c, rel=0.05)


def _standing(body):
    """A flat prism turned upright (+90 degrees about X: (x, y, z) -> (x, -z, y))."""
    from bridgesim.cad import CadBody
    from bridgesim.cad.mesh import tessellate
    from bridgesim.cad.parasolid import Body

    Rm = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    faces = [type(f)(f.kind, [lp @ Rm.T for lp in f.loops], normal=Rm @ f.normal,
                     origin=Rm @ f.origin) for f in body.faces]
    return CadBody(body.name, tessellate(Body(0, body.name, faces)), faces, 0,
                   named_in_cad=True)


def test_a_frame_leg_bears_on_its_sticks_end_grain():
    """An L-shaped UD leg (a frame, its sticks along each arm) stands on the platform on
    the end grain of its vertical arm (f_c); the same leg with its grain along the span
    bears across it (f_c90)."""
    from bridgesim.cad.synthetic import box, prism
    from bridgesim.sim.idealize import Overrides, build_model

    def leg(name, x0, s):  # outline in (x, -y), turned upright: the leg stands at y = 0
        P = [(x0, 0), (x0 + 20 * s, 0), (x0 + 20 * s, -130), (x0 + 150 * s, -130),
             (x0 + 150 * s, -150), (x0, -150)]
        return _standing(prism(name, P, -4.0, 4.0))

    fm = {}
    for grain in ("long", "span"):
        bodies = [leg("Leg L (4-ply UD)", -550.0, 1), leg("Leg R (4-ply UD)", 550.0, -1),
                  box("Beam (14-ply UD)", (-550, 150, -25), (550, 179.4, 25))]
        ov = Overrides.from_dict({"bodies": {"Leg*": {"kind": "frame", "grain": grain}}})
        m = build_model(bodies, MAT, SP, ov, seed=None)
        assert {b.name: b.kind for b in m.bodies}["Leg L (4-ply UD)"] == "frame"
        sup = np.asarray(m.p_support, bool)
        fm[grain] = float(np.asarray(m.params["fmax_p"])[sup].sum())
    ply = Ply.from_material(MAT, SP)
    assert fm["long"] / fm["span"] == pytest.approx(ply.f_c / ply.f_c90, rel=0.02)


def test_the_jack_slows_for_a_bridge_that_rings_slowly(tmp_path):
    """The simulated jack moves no further per period of the bridge's lowest mode than
    for a stiff bridge: a bridge ringing with a period above PERIOD_REF_S gets a jack
    slower in proportion (at most MAX_SLOWDOWN times). The period is that of the plate
    plus half the bridge's mass on the bridge's linear stiffness under the plate."""
    from bridgesim.sim import screen as scr
    from bridgesim.sim import study
    from bridgesim.sim.crush import (
        MAX_SLOWDOWN,
        PERIOD_REF_S,
        TRAVEL_REF_MM,
        fundamental_period,
        jack_speed,
    )

    assert jack_speed(100.0, 0.5 * PERIOD_REF_S) == 100.0
    assert jack_speed(100.0, 0.5 * PERIOD_REF_S, 0.1) == 100.0
    assert jack_speed(100.0, 2 * PERIOD_REF_S) == pytest.approx(50.0)
    assert jack_speed(100.0, 100 * PERIOD_REF_S) == pytest.approx(100.0 / MAX_SLOWDOWN)
    # past TRAVEL_REF_MM the jack speeds up with the travel: as far per period against
    # the travel reached, back up to the fidelity's speed
    assert jack_speed(100.0, 2 * PERIOD_REF_S, 1.5 * TRAVEL_REF_MM) == pytest.approx(75.0)
    assert jack_speed(100.0, 2 * PERIOD_REF_S, 5 * TRAVEL_REF_MM) == 100.0
    cad = tmp_path / "bridge.stl"
    write_stl(simple_beam_bridge(), cad)
    _, bodies, ov, _, mat, sp, _ = study._inputs(cad, None, None, None, None, None, "y",
                                                 "x")
    m = study._build(bodies, mat, sp, ov, 0, 1.6, False)
    k = 1000.0 / scr.solve(m, 1000.0)["plate_dy"]  # N/mm
    m_eff = (sp["crusher.plate_mass_kg"] + 0.5 * m.bridge_mass_kg) / 1000.0  # t
    T1 = fundamental_period(m)
    assert T1 == pytest.approx(2 * math.pi * math.sqrt(m_eff / k), rel=0.02)
    assert 0.02 < T1 < 0.5  # two stringers over a metre: tens of ms, a slow jack


def test_the_plate_rests_on_the_deck_not_on_what_spans_above_it(tmp_path):
    """The loading plate slides in under the top bracing and bears on the deck: a strut
    across the top, over the plate's footprint, carries none of the plate's load, and
    the plate set off-centre (a seeded hand placement) still sits on the deck instead of
    tipping about the strut's line."""
    from bridgesim.sim import study

    base = simple_beam_bridge()
    zmax = max(float(b.mesh.vertices[:, 2].max()) for b in base)
    ytop = max(float(b.mesh.vertices[:, 1].max()) for b in base)
    strut = box("Top Strut", (-5, ytop + 60, -zmax), (5, ytop + 66, zmax))
    k = {}
    for name, bodies in (("open", base), ("strut", base + [strut])):
        cad = tmp_path / f"{name}.stl"
        write_stl(bodies, cad)
        for off in (0.0, 3.0):
            ov = {"params": {"scatter.placement_mm": off}}
            k[name, off] = study.screen(cad, overrides=ov).stiffness_kgf_per_mm
    assert k["strut", 0.0] == pytest.approx(k["open", 0.0], rel=0.02)
    assert k["strut", 3.0] == pytest.approx(k["open", 3.0], rel=0.02)
    assert k["strut", 3.0] == pytest.approx(k["strut", 0.0], rel=0.1)


def test_a_negligible_joint_neither_governs_nor_competes():
    """A glued joint that can carry under MINOR_SHARE of F_u (a toothpick's end glued to
    a chord) breaking near the peak is a symptom of the bridge deforming: the break of
    the part that carries the load governs."""
    from bridgesim.sim.crush import KGF, MINOR_SHARE, Event, attribute, major_breaks

    peak = 1000 * KGF
    seq = [Event(0.100, 10.0, 990 * KGF, "joint_failure", "Toothpick | Chord", "",
                 None, None, 0.5 * MINOR_SHARE * peak),
           Event(0.101, 10.1, 985 * KGF, "joint_failure", "End Post | Gusset", "",
                 None, None, 0.4 * peak),
           Event(0.300, 30.0, 200 * KGF, "joint_failure", "Late | Joint", "", None, None,
                 0.4 * peak)]
    kept = major_breaks(seq, t_call=0.2, peak=peak)
    assert [e.what for e in kept] == ["End Post | Gusset"]
    gov, comp = attribute(kept, t_c=0.0995, peak=peak, peak_travel=10.0, speed_mm_s=100.0)
    assert gov.what == "End Post | Gusset" and not any("Toothpick" in c for c in comp)


def test_the_total_range_adds_mesh_and_model_error():
    """montecarlo's total range widens the runs' spread by the mesh and model-form
    error (lognormal, median 1, as the placeholders are drawn): for runs all at one load
    it is that load times exp(+-1.645 sigma), and it is the same for the same runs."""
    from bridgesim.sim.study import total_range

    s = math.hypot(SP["uncertainty.mesh_cov"], SP["uncertainty.model_cov"])
    lo, hi = total_range([1000.0] * 10, s)
    assert lo == pytest.approx(1000 * math.exp(-1.645 * s), rel=0.03)
    assert hi == pytest.approx(1000 * math.exp(1.645 * s), rel=0.03)
    assert total_range([900.0, 1000.0, 1100.0], s) == total_range([900.0, 1000.0,
                                                                    1100.0], s)
    wide = total_range([800.0, 1000.0, 1200.0], s)
    assert wide[0] < lo and wide[1] > hi


def test_named_stl_solids_keep_their_names(tmp_path):
    """An ASCII STL with one named solid per body reads back as those bodies, with their
    names, even where two of them touch (a binary STL merges touching bodies)."""
    from bridgesim.cad.synthetic import layered_truss

    bodies = layered_truss()
    write_stl(bodies, tmp_path / "truss.stl", names=True)
    back = load_cad(tmp_path / "truss.stl")
    assert sorted(b.name for b in back) == sorted(b.name for b in bodies)
    assert all(b.mesh.volume > 0 for b in back)



def test_a_plate_that_does_not_fit_is_asked_about_not_exploded(tmp_path):
    """Two kerbs on the deck 86 mm apart, too tall for the plate to sit on, leave no
    room for the 90 mm plate between them: inspect asks (important, `plate_fit`, naming
    them), and the model takes them as clear of the plate (no plate contact point starts
    inside it) instead of throwing them off."""
    from bridgesim.sim import study

    deck_top = 150 + 23.1
    kerbs = [box(f"Kerb {s}", (-60, deck_top, z0), (60, deck_top + 70, z0 + 6))
             for s, z0 in (("Near", 43.0), ("Far", -49.0))]
    cad = tmp_path / "kerbs.stl"
    write_stl(simple_beam_bridge() + kerbs, cad, names=True)
    rep = study.inspect(cad)
    q = next(q for q in rep.questions if q.id == "plate_fit")
    assert q.severity == "important" and "Kerb Near" in q.question and "Kerb Far" in q.question
    _, bodies, ov, _, mat, sp, _ = study._inputs(cad, None, None, None, None, None, "y", "x")
    m = study._build(bodies, mat, sp, ov, None, 1.6, False)
    loc = m.X[m.a_node[m.q_att]] + m.a_r0[m.q_att] - m.X[m.plate_node]
    assert not (np.abs(loc) < m.plate_half - 0.05).all(axis=1).any()
    write_stl(simple_beam_bridge(), tmp_path / "open.stl")
    assert not [q for q in study.inspect(tmp_path / "open.stl").questions
                if q.id == "plate_fit"]


def test_a_soft_bridge_is_loaded_without_a_kick(tmp_path):
    """A bridge ringing slowly under the 4.2 kg plate (two stringers over a metre: a
    period of ~0.17 s) is started and sped up over whole periods, and the plate is
    damped only relative to the jack: its load against the deck's deflection is a
    straight line through zero at the bridge's stiffness. (A 20 ms start kicked the
    plate: at a fixed 600 mm/s the line started at 14 kgf, more than the bridge
    carries.)"""
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    pytest.importorskip("numba")
    write_stl(simple_beam_bridge(), tmp_path / "b.stl")
    k = study.screen(tmp_path / "b.stl").stiffness_kgf_per_mm
    rep, _ = study.crush(tmp_path / "b.stl", seed=None, replay=False,
                         options=CrushOptions(speed_mm_s=600, max_travel_mm=5))
    T1 = rep.quality.fundamental_period_ms / 1000
    assert rep.quality.speed_mm_s == pytest.approx(600 * 0.010 / T1, rel=0.01)
    L, D = np.array(rep.curve.load_kgf), np.array(rep.curve.deflection_mm)
    sel = (D > 1.0) & (D < 4.0)
    slope, icpt = np.polyfit(D[sel], L[sel], 1)
    assert 0.75 * k < slope < 1.05 * k  # the screen holds the supports fully
    assert abs(icpt) < 0.1 * slope * 4.0


def test_screen_takes_unglued_faces_as_bearing_only(tmp_path):
    """Faces that touch but are not glued (glue 0: a cut end left dry) bear in
    compression only, as in the crush: the screen does not list them as glue limits at
    ~0 kgf, and a block resting dry on the deck adds no stiffness in tension."""
    from bridgesim.sim import study

    deck_top = 150 + 23.1
    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge() + [box("Weight", (300, deck_top, -20),
                                          (340, deck_top + 20, 20))], cad, names=True)
    glued = study.screen(cad)
    dry = study.screen(cad, overrides={"glue": {"Deck* | Weight*": 0}})
    assert any("Weight" in lm.part for lm in glued.limits)
    assert not any("Weight" in lm.part for lm in dry.limits)
    assert dry.stiffness_kgf_per_mm == pytest.approx(glued.stiffness_kgf_per_mm, rel=0.05)


def test_dry_faces_are_no_joint_failures(tmp_path):
    """A part resting dry on another (glue 0) was never glued: the crush logs no joint
    separating there at 0 kgf, and the first joint failure is not put at zero load."""
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    deck_top = 150 + 23.1
    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge() + [box("Weight", (300, deck_top, -20),
                                          (340, deck_top + 20, 20))], cad, names=True)
    rep, res = study.crush(cad, overrides={"glue": {"Deck* | Weight*": 0}}, seed=None,
                           replay=False, options=CrushOptions(speed_mm_s=150,
                                                              max_travel_mm=1.0,
                                                              auto_speed=False))
    assert not any("Weight" in e.what for e in res.events)
    assert rep.curve.joints_broken[0] == 0
    assert rep.Fu_first_joint_failure_kgf is None or rep.Fu_first_joint_failure_kgf > 0


def test_glue_points_give_a_patch_its_turning_stiffness():
    """The glue points of a patch, equal shares of its area, have the rectangle's own
    second moment about both axes whatever their number (two per side at the Gauss
    points), so a small glued joint is neither a hinge (one point) nor 25 % too soft in
    turning (two points at the cell midpoints)."""
    from shapely.geometry import box as sbox

    from bridgesim.sim.idealize import _sample_polygon

    Lu, Lw = 26.0, 18.0
    poly = sbox(-Lu / 2, -Lw / 2, Lu / 2, Lw / 2)
    for spacing in (40.0, 9.0, 4.0):  # 2 x 2, 3 x 2, 6 x 4 points
        pts = _sample_polygon(poly, spacing, max_pts=400, min_per_side=2, exact_moment=True)
        A = Lu * Lw / len(pts)
        assert len(pts) >= 4
        assert (A * pts[:, 0] ** 2).sum() == pytest.approx(Lu**3 * Lw / 12, rel=1e-6)
        assert (A * pts[:, 1] ** 2).sum() == pytest.approx(Lw**3 * Lu / 12, rel=1e-6)
    mid = _sample_polygon(poly, 40.0, max_pts=400, min_per_side=2)
    assert (Lu * Lw / 4 * mid[:, 0] ** 2).sum() == pytest.approx(0.75 * Lu**3 * Lw / 12)


def test_stacked_plates_are_meshed_node_on_node(tmp_path):
    """Plates glued face to face (a deck laminated from separate bodies, its middle
    panel's plies grooved for the side panels' tongues, on a block under the plate, as
    on Bridge_v10) are meshed so that the glue joins nodes straight across: no glue
    point hangs on a node of either plate by an arm in its plane, and the deck's
    stiffness under the plate is the same at the quick and the fine mesh (hung on
    nodes off-centre, each thin layer bent between them: 9x softer at quick)."""
    from bridgesim.cad.synthetic import box
    from bridgesim.sim import screen as scr
    from bridgesim.sim import study

    y0 = 180.0
    B = [box("Pier L", (-550, 0, -80), (-500, 150, 80)),
         box("Pier R", (500, 0, -80), (550, 150, 80)),
         box("Stringer Near (13-ply UD)", (-550, 150, 50), (550, y0, 76)),
         box("Stringer Far (13-ply UD)", (-550, 150, -76), (550, y0, -50)),
         box("Floor Block", (-100, 0, -50), (100, y0, 50)),
         box("Deck Middle Bot (2-ply 0-90)", (-125, y0, -80), (125, y0 + 4, 80)),
         box("Deck Middle Mid (3-ply 0-90)", (-105, y0 + 4, -80), (105, y0 + 10, 80)),
         box("Deck Middle Top (2-ply 0-90)", (-125, y0 + 10, -80), (125, y0 + 14, 80))]
    for side, (a, b) in (("L", (-550, -125)), ("R", (125, 550))):
        B += [box(f"Deck {side} Bot (2-ply 0-90)", (a, y0, -80), (b, y0 + 4, 80)),
              box(f"Deck {side} Top (2-ply 0-90)", (a, y0 + 10, -80), (b, y0 + 14, 80)),
              box(f"Deck {side} Mid (3-ply 0-90)",  # its tongue runs 20 mm into the groove
                  (a if side == "L" else 105, y0 + 4, -80),
                  (-105 if side == "L" else b, y0 + 10, 80))]
    cad = tmp_path / "deck.stl"
    write_stl(B, cad, names=True)
    _, bodies, ov, _, mat, sp, _ = study._inputs(
        cad, {"bodies": {"Floor Block": {"kind": "block"}, "Pier*": {"kind": "block"}}},
        None, None, None, None, "y", "x")
    k = {}
    for scale in (1.6, 0.5):
        m = study._build(bodies, mat, sp, ov, None, scale, False)
        names = [b.name for b in m.bodies]
        for i in range(len(m.c_a)):
            pa = m.patches[m.c_patch[i]]
            if names[pa.body_a].startswith("Deck") and names[pa.body_b].startswith("Deck"):
                nrm = m.c_n0[i] / np.linalg.norm(m.c_n0[i])
                if abs(nrm[1]) < 0.99:
                    continue  # an end face butting another: not a stack
                for att in (m.c_a[i], m.c_b[i]):
                    r = m.a_r0[att]
                    assert np.linalg.norm(r - (r @ nrm) * nrm) < 1.0, scale
        k[scale] = 1000.0 / scr.solve(m, 1000.0)["plate_dy"]
    assert k[1.6] == pytest.approx(k[0.5], rel=0.15)  # was 9x softer at quick


def test_a_finished_run_is_reused_only_for_the_same_inputs(synthetic_crush, tmp_path):
    """montecarlo resumes from a run left in its folder only if everything it depends
    on is unchanged: the overrides as read now (a file passed by path too), the
    material, the simulation parameters, seed and fidelity. (An overrides file was
    compared by its path, so a run with other values was reused.)"""
    import yaml

    from bridgesim.sim.study import _finished

    d, rep, _ = synthetic_crush
    out, cad = d / "out", d / "bridge.stl"
    assert rep.inputs_sha256
    assert _finished(out, cad, 0, "standard", None) is not None
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({"params": {"material.glue.tau_g_MPa": 0.5}}))
    assert _finished(out, cad, 0, "standard", other) is None
    assert _finished(out, cad, 1, "standard", None) is None
    assert _finished(out, cad, 0, "quick", None) is None
    same = tmp_path / "same.yaml"
    same.write_text("{}\n")
    assert _finished(out, cad, 0, "standard", same) is not None


def test_a_stick_modulus_override_scales_the_shear_modulus(tmp_path):
    """Setting the stick modulus (by hand, or a montecarlo draw) keeps the wood's E/G:
    G follows, marked assumed, unless it is set too."""
    from bridgesim.sim import study

    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), cad)
    *_, mat0, _sp, _rs = study._inputs(cad, None, None, None, None)
    *_, mat, _sp, _rs = study._inputs(cad, {"params": {"material.E_MPa": 9000}}, None,
                                      None, None)
    assert mat.E_MPa.value / mat.G_MPa.value == pytest.approx(
        mat0.E_MPa.value / mat0.G_MPa.value)
    assert mat.G_MPa.source == "assumed"
    *_, mat2, _sp, _rs = study._inputs(
        cad, {"params": {"material.E_MPa": 9000, "material.G_MPa": 700}}, None, None, None)
    assert mat2.G_MPa.value == 700



def test_end_grain_glue_changes_smoothly_with_the_cut():
    """A glued face is the more end grain the closer its normal lies to the fibres,
    smoothly (like Hankinson's bearing), not side grain up to 60 degrees and end grain
    past it: a diagonal's foot cut at 55 degrees to its fibres (Bridge_v10) glues at
    0.60 of side grain with end grain at 0.5, not at 1."""
    from bridgesim.sim.idealize import end_grain_share

    ef = 0.5
    cs = np.cos(np.radians(np.linspace(0, 90, 91)))
    share = np.array([end_grain_share(c, ef) for c in cs])
    assert share[0] == pytest.approx(1.0) and share[-1] == pytest.approx(0.0)
    assert np.all(np.diff(share) < 0) and np.abs(np.diff(share)).max() < 0.05
    c = math.cos(math.radians(34.6))
    assert 1 - end_grain_share(c, ef) * (1 - ef) == pytest.approx(0.60, abs=0.01)


def test_cut_ends_stand_off_their_seats_by_a_seeded_gap(tmp_path):
    """A member's end glued on another part is not cut square: with a seed its face
    stands off its seat by 0 at the edge that touches up to a seeded gap (at most
    scatter.bearing_gap_mm) at the other. The glue filling the gap is weaker the wider it
    is (gap_strength: full up to glue.gap_fill_mm, none from ~1 mm), and where it is too
    wide the face is dry, its two sides held apart by the gap until it closes. The
    perfect bridge (no seed) has none, and side-grain laps are cut flat."""
    from bridgesim.sim import solver as S
    from bridgesim.sim import study
    from bridgesim.sim.idealize import gap_strength

    assert gap_strength(0.1) == 1.0 and gap_strength(0.25) == pytest.approx(0.9)
    assert gap_strength(0.5) == pytest.approx(0.6) and gap_strength(1.0) == 0.0
    deck_top = 150 + 23.1
    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge() + [box("Post (4-ply UD)", (300, deck_top, -4),
                                          (310, deck_top + 60, 4))], cad, names=True)
    _, bodies, ov, _, mat, sp, _ = study._inputs(cad, None, None, None, None, None, "y",
                                                 "x")
    sp = sp.updated({"scatter.bearing_gap_mm": 3.0, "glue.gap_fill_mm": 0.15,
                     "scatter.glue_cov": 0.0, "scatter.glue_bridge_cov": 0.0})

    def build(seed):
        m = study._build(bodies, mat, sp, ov, seed, 1.0, False)
        st = S.initial_state(m, 0.0)
        pa, _ = S._attach_pos(m, st, m.c_a)
        pb, _ = S._attach_pos(m, st, m.c_b)
        dn = ((pb - pa) * m.c_n0).sum(1)
        post = np.zeros(len(dn), bool)
        for pa_ in m.patches:
            if "Post" in m.bodies[pa_.body_a].name + m.bodies[pa_.body_b].name:
                post[np.asarray(pa_.springs, int)] = True
        return dn, np.asarray(m.params["c_d0"]), post, np.asarray(m.c_tau)

    dn, d0, post, tau0 = build(None)
    assert post.any() and np.abs(dn).max() < 1e-6 and not d0.any()
    n_dry = n_weak = 0
    for seed in range(6):
        dn, d0, post, tau = build(seed)
        assert np.abs(dn[~post]).max() < 1e-6 and not d0[~post].any()  # laps: flat
        assert np.allclose(tau[~post], tau0[~post])
        dry = d0 >= 1
        assert (dn[dry] > 1.0 - 1e-9).all() and (dn[post] <= 3.0 + 1e-9).all()
        wet = post & ~dry
        assert not wet.any() or np.abs(dn[wet]).max() < 1e-6  # the glue fills the gap
        ratio = tau[wet] / tau0[wet]
        assert ((ratio > 0) & (ratio <= 1 + 1e-9)).all()
        n_dry += int(dry.any())
        n_weak += int((ratio < 0.99).any())
    assert n_dry >= 2 and n_weak >= 3



def test_butt_joints_cut_the_plies_whose_joints_line_up():
    """The tension capacity of a laminated member with butt joints is its weakest path
    through the plies, by statics: in a brick pattern (each ply's joints half a stick
    from its neighbours') joints repeat every stick, so every other ply is cut at the
    same section and a 13-ply member keeps 6 plies there (not 12 of 13); joints all in
    line cut every ply; joints staggered a little each ply give the old staircase,
    (n - k) plies plus the glue sheared between k joints."""
    from bridgesim.sim.idealize import butt_path_capacity

    Ls, P = 115.0, 1.0
    strong = 100.0  # glue far stronger than a ply over any stagger: the net section
    assert butt_path_capacity(13, Ls / 2, Ls, P, strong) == pytest.approx(6.0)
    assert butt_path_capacity(3, Ls / 2, Ls, P, strong) == pytest.approx(1.0)
    assert butt_path_capacity(4, Ls / 2, Ls, P, strong) == pytest.approx(2.0)
    assert butt_path_capacity(5, Ls, Ls, P, strong) == pytest.approx(0.0)
    assert butt_path_capacity(5, Ls / 5, Ls, P, strong) == pytest.approx(4.0)
    # a little stagger, weak glue: the path runs from joint to joint
    tw = 0.01  # per mm: 10 mm of glue carry a tenth of a ply
    assert butt_path_capacity(3, 10.0, Ls, P, tw) == pytest.approx(2 * 0.1)
    old = min((3 - k) * P + (k - 1) * tw * 10.0 for k in range(1, 4))
    assert butt_path_capacity(3, 10.0, Ls, P, tw) == pytest.approx(old)


def test_laminated_members_and_stick_plates_lose_tension_strength_at_butt_joints():
    """A 13-ply member longer than a stick, brick pattern by default, keeps 6/13 of its
    tension strength by statics (the glue between plies at the laminate's interlaminar
    strength far stronger than a ply over half a stick), times
    wood.butt_tension_efficiency (0.8: in stick-thin plies the glue cracks from the
    joints' tips, Jung & Murphy 1983); a deck plate longer and wider than a stick keeps
    about 0.8 x half (its strands of sticks, rows and plies, in a brick pattern); with
    butt_stagger_mm 0 (no joints) both keep all of it."""
    from bridgesim.sim.idealize import Overrides, build_model

    bodies = simple_beam_bridge() + [box("Chord (13-ply UD)", (-300, 200, -13),
                                         (300, 210, 13))]
    m = build_model(bodies, MAT, SP, Overrides(), seed=None)
    info = {b.name: b for b in m.bodies}
    ft = MAT.f_t_MPa.value
    eff = SP["wood.butt_tension_efficiency"]
    assert eff == pytest.approx(0.8)
    assert info["Chord (13-ply UD)"].f_t_eff_MPa == pytest.approx(6 / 13 * eff * ft,
                                                                  rel=0.02)
    deck = info["Deck (2-ply 0-90)"]
    ov = Overrides.from_dict({"bodies": {"*": {"butt_stagger_mm": 0}}})
    m0 = build_model(bodies, MAT, SP, ov, seed=None)
    info0 = {b.name: b for b in m0.bodies}
    assert info0["Chord (13-ply UD)"].f_t_eff_MPa == pytest.approx(ft, rel=0.02)
    ratio = deck.f_t_eff_MPa / info0["Deck (2-ply 0-90)"].f_t_eff_MPa
    assert 0.4 * eff < ratio < 0.55 * eff
    qs = {q["id"]: q for q in m.questions}
    assert "butt_joints" in qs and "Deck" in qs["butt_joints"]["question"]


def test_the_load_at_a_break_and_in_a_hold_is_smoothed_not_raw():
    """At a break the reading is the trend's end, not the raw sample there: a load
    rising to a break with a 3 % ring at its crest reads the true peak within 1 %
    (odd reflection pinned it to the crest, +3 %). In the hold that follows (the jack
    still, the load falling in time as a joint lets go, ringing), each moment reads the
    load over the period before it (the ring cancelled, half a period late), the break
    itself the load it broke at; along the travel it read the hold's start throughout."""
    from bridgesim.sim.crush import reported_load

    v, T = 100.0, 1 / 270.0
    t = np.arange(0, 0.08, 3e-5)
    tb, th = 0.037 - 0.25 * T, 0.057  # the break at a crest of the ring; the hold
    hold = (t >= tb) & (t < th)
    x = np.where(t < tb, v * t, np.where(hold, v * tb, v * tb + v * (t - th)))
    peak = 500.0 * v * tb
    fall = np.clip((t - tb) / 0.015, 0, 1)  # the load falls 40 % over 15 ms
    true = np.where(t < tb, 500.0 * x, peak * (1 - 0.4 * fall))
    raw = true + 0.03 * peak * np.sin(2 * np.pi * (t - tb) / T + np.pi / 2)
    rep = reported_load(t, raw, v, x, breaks=[tb], hold=hold, period_s=T)
    ib = np.searchsorted(t, tb) - 1
    assert rep[ib] == pytest.approx(true[ib], rel=0.01)
    assert rep[:ib + 1].max() == pytest.approx(peak, rel=0.01)
    inh = np.where(hold)[0]
    # the load at the break plus the true change since, over the period before each
    # moment (a fall read half a period late)
    since = np.where(t < tb, 0.0, true - peak)
    late = np.array([peak + since[(t > ti - T) & (t <= ti)].mean() for ti in t[inh]])
    # (kept within the raw force's envelope over half a period either side, which a
    # reading half a period late can touch: within the ring's amplitude, 3 %)
    assert np.abs(rep[inh] - late).max() < 0.035 * peak
    assert rep[inh[0]] == pytest.approx(peak, rel=0.01)  # the break: the load it broke at
    flat = reported_load(t, raw, v, x, breaks=[tb])  # along the travel only
    assert np.abs(flat[inh] - true[inh]).max() > 0.1 * peak


def test_the_prediction_carries_the_model_error_and_its_own_interval():
    """The model-form and mesh error widen the true F_u around the runs: the §12.7-best
    prediction over the runs times that error is lower than over the runs alone (the
    1/F weight leans on the low side), and a bootstrap over the runs gives how far other
    runs could move it, wide for 8 runs in two clusters."""
    from bridgesim.sim.study import best_prediction, prediction_interval

    F = np.array([1323.0, 1450, 1600, 1700, 1762, 2502, 2600, 2674])  # Bridge_v10 mc8
    p0, p1 = best_prediction(F), best_prediction(F, 0.25)
    assert p1 < p0
    G = np.exp(np.random.default_rng(0).normal(0, 0.25, 200000) - 0.5 * 0.25**2)

    def err(q):  # the expected §12.7 error, the runs times the error
        T = (F[:, None] * G[None, :2000]).ravel()
        return float(np.mean(np.abs(T - q) / T))

    assert err(p1) < err(p0)
    lo, hi = prediction_interval(F, 0.25)
    assert lo < p1 < hi and (hi - lo) / p1 > 0.15
    lo1, hi1 = prediction_interval(np.full(8, 1500.0))
    assert lo1 == hi1 == 1500.0


def test_elements_past_the_crack_band_dissipate_the_fracture_energy():
    """An element larger than the crack band a mode allows (fibre tension: 2 E G_f /
    f_t^2): its strength in that mode is lowered until its steepest softening
    dissipates G_f per unit crack area (Bazant & Oh), 1/2 E e0^2 l_c rf = G_f; a small
    element keeps its strength. Until round 16 the large one kept its strength and
    dissipated G_f x 1.05 / rf_raw (~1.6x here)."""
    from bridgesim.sim import shell as shellmod
    from bridgesim.sim.materials import Ply

    ply = Ply.from_material(MAT, SP)
    G = (SP["wood.G_f_N_per_mm"], SP["wood.G_f90_N_per_mm"], SP["wood.G_v_N_per_mm"])
    h_max = 2 * ply.E_L * G[0] / ply.f_t**2  # the fibre-tension crack band
    for size, reduced in ((0.1 * h_max, False), (1.6 * h_max, True)):
        xy = np.array([[0.0, 0.0], [size, 0.0], [0.0, size]]) - [size / 3, size / 3]
        ed = shellmod.element_data(xy, [0.0], 2.0, ply, 1.0, 1.0, G, ply.E_L * 2.0)
        e0, rf = ed["lim"][0], ed["rf"][0]
        if reduced:
            assert e0 < ply.f_t / ply.E_L and ed["snapback"]
            assert 0.5 * ply.E_L * e0**2 * ed["lc"] * rf == pytest.approx(G[0], rel=1e-6)
        else:
            assert e0 == pytest.approx(ply.f_t / ply.E_L)
            assert 0.5 * ply.E_L * e0**2 * ed["lc"] * rf == pytest.approx(G[0], rel=1e-6)


V10 = Path(os.environ.get("BRIDGESIM_TEST_V10",
                          ROOT / "examples" / "cad" / "Bridge_v10.SLDPRT"))
V10_OVERRIDES = {  # as the team's design report (Rev N, section 10.1) models it
    "glue": {"BC * | Stub *": 0, "Deck ? Top* | Deck Middle Top*": 0,
             "Deck ? Bot* | Deck Middle Bot*": 0, "Deck ? Mid* | Deck Middle Mid*": 0,
             "BC * | Heel Seat *": 10, "Floor Beam M (block) | Floor Beam M Tenon *": 10}}


@pytest.mark.skipif(not V10.exists(), reason="Bridge_v10.SLDPRT (a team's design, not "
                                              "in the repository) not available")
def test_bridge_v10_regression():
    """A team's real design (Rev N, 69 bodies; set BRIDGESIM_TEST_V10 to its file): what
    the model is made of, and the screen's stiffness and first limits, against the values
    of round 18 (validation/README.md section 3.10): 227 glued joints, 7.9 kg, ~1010
    kgf/mm under the plate with sticks of 10.5 GPa and butt joints that cost stiffness
    (~1130 in round 18, ~820 at 7.1 GPa in rounds 16-17), the
    End Post to knee joints the first glue limits at ~430 kgf (~265 in round 24, ~180
    since round 25), and their faces or the outer floor beams' on the pier legs the first
    to bear past the wood's crushing strength (~2000-2400 kgf if linear; a deck's face on
    a heel gusset at ~2900 since round 25); end-grain and butt-joint
    questions asked; its lowest tap-test frequency ~48 Hz (~39 since round 24)."""
    from bridgesim.sim import study

    rep = study.inspect(V10, overrides_data=V10_OVERRIDES)
    assert rep.n_bodies == 69 and len(rep.joints) == 227
    assert rep.mass_kg == pytest.approx(7.90, abs=0.05)
    ids = {q.id for q in rep.questions}
    assert {"end_grain_joints", "butt_joints"} <= ids
    sc = study.screen(V10, overrides_data=V10_OVERRIDES, load_kgf=1000.0, n_limits=400)
    # ~1010 in rounds 21-23 (butt joints cost stiffness: ~0.88 of E for 13 plies); ~750
    # since round 24 (the wood under a face pressed across the grain is compliant)
    assert 680 < sc.stiffness_kgf_per_mm < 830
    glue = [lm for lm in sc.limits if lm.mode == "glue"]
    # node A (the End Post on the knee) or node M (the diagonal feet on the stub): the
    # size effect weakens node A's large laps more (round 19)
    assert ("End Post" in glue[0].part or "Stub" in glue[0].part) \
        and "Knee-Diagonal" in glue[0].part
    # ~265 in round 24 (~420 before): the knee face, softer across the grain, passes
    # more of the End Post's load through its glue in shear; ~180 since round 25 (the
    # glue lines' shear stiffness from the parts' own shear: the knee's lap stiffer)
    assert 150 < glue[0].load_kgf < 500
    # no face bears past the wood's crushing strength below ~F_u: the knee's or the floor
    # beams' on the pier legs (~2000-2400 kgf if linear) until round 24; since round 25 a
    # short face crushes later (prEN 1995's sqrt(l_ef / l)), a deck's face on a heel
    # gusset first, at ~2900
    bear = [lm for lm in sc.limits if lm.mode == "bearing"]
    assert ("End Post" in bear[0].part and "Knee-Diagonal" in bear[0].part) or (
        "Floor Beam" in bear[0].part and "Pier Leg" in bear[0].part) or (
        "Deck" in bear[0].part and "gus_heel" in bear[0].part)
    assert 1700 < bear[0].load_kgf < 3500
    assert 34 < sc.modes[0]["frequency_hz"] < 56


def test_long_laps_lose_strength_by_shear_lag_as_generalised_volkersen():
    """Verification of the glue model against closed-form fracture mechanics: a lap of
    two 2 mm sticks carries tau_f b l tanh(w/2) / (w/2) with w^2 = tau_f^2 l^2 / (G_IIc
    E t) (generalised Volkersen, Gustafsson 1987, as in Serrano 2000): the longer the lap
    and the lower the glue's toughness, the less of tau_f x area. The simulated single
    lap (glue points hung on the sticks' fibre beams) follows it within 12 % from 60 to
    110 mm, for G_IIc 0.6 and 1.2 N/mm; at 20 mm it is ~9 % below pure shear (a single
    lap also peels), which tau_peak_over_avg calibrates."""
    pytest.importorskip("numba")
    tau, E, t, b = 2.31, 7100.0, 2.0, 10.0
    for G in (0.6, 1.2):
        ratio = {}
        for L in (20.0, 60.0, 110.0):
            n_ov = max(8, int(L / 2.5))
            r = lab.lap_shear(tau, 0.66, 150.0, 60.0, 0.25, G, overlap=L, width=b, t=t,
                              E=E, n_ov=n_ov, n_glue=2 * n_ov + 1)
            ratio[L] = r.peak / (tau * b * L)
            w = math.sqrt(tau**2 * L**2 / (G * E * t))
            volk = math.tanh(w / 2) / (w / 2)
            if L > 20:
                assert ratio[L] == pytest.approx(volk, rel=0.12), (G, L)
        assert ratio[110.0] < 0.8 * ratio[20.0]


def test_the_reading_stays_within_the_force_the_jack_applied():
    """A force that rounds over in the last 0.3 mm before a break (the bridge softening
    as it fails) read 4-5 % high from the straight line fitted to the end of the
    stretch; a break that the load survives, followed 8 ms later by the collapse within
    the same stretch, read the collapse back at the break, while the raw force still
    rose. The reading is kept within the raw force's envelope over a centred ring period:
    the peak within the ring of the true one, never a load the jack did not apply."""
    from bridgesim.sim.crush import reported_load

    v, T = 100.0, 1 / 150.0
    t = np.arange(0, 0.07, 3e-5)
    x = v * t
    tb = 0.040
    xb = v * tb
    # rises at 500 N/mm, rounding over in the last 0.3 mm to a flat top, then falls
    true = np.where(x < xb - 0.3, 500.0 * x,
                    500.0 * (xb - 0.3) + 500.0 * 0.3 * (1 - ((xb - x) / 0.3 - 1) ** 2) / 2
                    * np.where(x < xb, 1.0, 0.0))
    true = np.where(x >= xb, 0.5 * 500.0 * (xb - 0.15), true)
    raw = true + 5.0 * np.sin(2 * np.pi * t / T)
    rep = reported_load(t, raw, v, x, breaks=[tb], period_s=T)
    pre = t < tb
    assert rep[pre].max() <= raw[pre].max() + 1e-9
    assert rep[pre].max() == pytest.approx(true[pre].max(), rel=0.01)
    # a survived break at tb, the collapse 8 ms later in the same stretch
    tc = tb + 0.008
    true2 = np.where(t < tc, 500.0 * x, 500.0 * x * np.clip(1 - (t - tc) / 0.002, 0, 1))
    raw2 = true2 + 5.0 * np.sin(2 * np.pi * t / T)
    rep2 = reported_load(t, raw2, v, x, breaks=[tb], period_s=T)
    rising = (t >= tb) & (t < tc - T)
    assert (rep2[rising] >= true2[rising] - 0.01 * true2[rising]).all()


def _stub_result(m, governing=""):
    from types import SimpleNamespace

    return SimpleNamespace(kinetic_ratio_max=0.0, added_mass_frac=0.0, failure_mode="collapse",
                           notes=[], governing=governing, model=m, dt=1e-6, steps=1,
                           wall_s=1.0, settle_sag_mm=0.0, speed_mm_s=100.0, holds=0,
                           hold_time_s=0.0)


def test_a_light_bridge_gets_short_elements_and_a_slow_jack_and_needs_fine():
    """On a light bridge (members 6 mm deep) elements longer than its members' depth do
    not converge (the light truss in round 18, elements 22 mm at standard: quick 17.9,
    standard 34.5, fine 44.6 kgf). Since round 25 (the realism review of round 24: quick
    still read half of fine) a member shallower than 10 mm gets elements at most 1.33x
    its depth at quick and standard alike (finer at fine), and a bridge that rings slowly
    starts from standard's jack speed at quick too (quick's 200 mm/s left the light truss
    dynamic, kinetic/work 0.10). That is not enough: it read 18.5 / 37.8 / 44.8 kgf at
    quick / standard / fine, its plates and glued joints setting it, so quick and
    standard stay flagged on such a bridge, fine is not, nor quick on deep members."""
    from bridgesim.cad.synthetic import layered_truss
    from bridgesim.sim import study
    from bridgesim.sim.crush import FIDELITY
    from bridgesim.sim.idealize import Overrides, build_model, element_depth_cap

    for scale, ok in ((1.6, False), (0.75, False), (0.5, True)):
        m = build_model(layered_truss(), MAT, SP, Overrides(), seed=None, mesh_scale=scale)
        q = study._quality(_stub_result(m), MAT, SP)
        assert q.trustworthy == ok and q.min_fidelity == "fine", scale
        assert ok or any("too coarse" in r for r in q.reasons)
        bars = [b for b in m.bodies if b.kind == "bar" and b.section_mm]
        for b in bars:
            d = min(b.section_mm)
            L = np.linalg.norm(m.X[m.b_j[b.beams]] - m.X[m.b_i[b.beams]], axis=1)
            assert L.max() <= element_depth_cap(d, scale) * 1.001 + 1e-9 or \
                L.max() <= 30.0 * scale, (b.name, scale)
    assert element_depth_cap(6.0, 1.6) == element_depth_cap(6.0, 0.75) == pytest.approx(
        1.33 * 6.0)
    assert element_depth_cap(26.0, 1.6) == pytest.approx(52.0)
    for fid in ("quick", "standard", "fine"):
        o = study._screening(fid)
        assert o.slow_bridge_speed_mm_s == min(FIDELITY[fid][1], FIDELITY["standard"][1])
    deep = [box("Pier L (block)", (-550, 0, -40), (-500, 150, 40)),
            box("Pier R (block)", (500, 0, -40), (550, 150, 40)),
            box("Beam (13-ply UD)", (-550, 150, -13), (550, 180, 13))]
    m = build_model(deep, MAT, SP, Overrides(), seed=None, mesh_scale=1.6)
    assert study._quality(_stub_result(m), MAT, SP).trustworthy


def test_a_crack_is_flagged_only_past_the_crack_band_of_its_own_mode():
    """A plate whose elements are past the crack band in shear only: a crack that formed
    in shear there starts where the mesh decides (flagged); one that formed in fibre
    tension, whose crack band the elements are within, is not."""
    from bridgesim.sim import study
    from bridgesim.sim.idealize import Overrides, build_model

    m = build_model(simple_beam_bridge(), MAT, SP, Overrides(), seed=None)
    deck = [b.name for b in m.bodies].index("Deck (2-ply 0-90)")
    sel = m.s_body == deck
    snap3 = np.zeros((len(m.s_body), 3), bool)
    snap3[sel, 2] = True  # past the crack band in shear only
    m.params["s_snap3"] = snap3
    m.params["s_snap"] = snap3.any(1)
    mode = np.full(len(m.s_body), -1)
    mode[np.where(sel)[0][:5]] = 2
    m.params["s_crack_mode"] = mode
    q = study._quality(_stub_result(m, "plate crack: Deck (2-ply 0-90)"), MAT, SP)
    assert not q.trustworthy and any("crack band" in r and "shear" in r for r in q.reasons)
    mode[np.where(sel)[0][:5]] = 0  # it cracked in fibre tension
    q = study._quality(_stub_result(m, "plate crack: Deck (2-ply 0-90)"), MAT, SP)
    assert not any("crack band" in r for r in q.reasons)


def test_screen_gives_the_tap_test_frequencies(tmp_path):
    """The screen's natural frequencies (what a tap test of the built bridge measures):
    a square laminated beam on knife edges with the 4.2 kg loading plate on it rings
    in vertical bending, the plate moving most, at sqrt(k / (M_plate + 0.49 m_beam)) / 2 pi
    with k the screen's own stiffness under the plate (Rayleigh, the beam's mass at
    mid-span), within 5 % (below it, the beam rolls on its 2 mm knife edges)."""
    from bridgesim.sim import study

    cad = tmp_path / "beam.stl"
    write_stl([box("Pier L", (-546, 0, -40), (-544, 150, 40)),
               box("Pier R", (544, 0, -40), (546, 150, 40)),
               box("Beam", (-550, 150, -15), (550, 180, 15))], cad, names=True)
    ov = {"bodies": {"Beam": {"layup": "15-ply UD", "butt_stagger_mm": 0},
                     "Pier*": {"kind": "block"}}}
    rep = study.screen(cad, overrides=ov)
    k = rep.stiffness_kgf_per_mm * 9.80665  # N/mm
    m_beam = MAT.density_kg_m3.value * 1100 * 30 * 30 * 1e-12  # t
    m_plate = SP["crusher.plate_mass_kg"] * 1e-3
    f = math.sqrt(k / (m_plate + 0.49 * m_beam)) / (2 * math.pi)
    plate = [m for m in rep.modes if m["moves"][0].startswith("loading plate")]
    assert plate and plate[0]["frequency_hz"] == pytest.approx(f, rel=0.05)
    assert all(m["frequency_hz"] > 0 for m in rep.modes)


def test_a_solid_joint_breaks_at_the_wood_strength():
    """Two bodies drawn apart but one piece of wood (a floor beam and its tenon): the
    glue override 'solid' gives the face between them the wood's own strength instead of
    a glue line's (or the 10x glue that made it unbreakable): shear f_v / 1.5 averaged,
    as the code check of a notched end, tension along the fibres that cross it (a block
    with no grain known), no fit-up gap and no glue scatter; it is listed as solid."""
    from bridgesim.sim import study
    from bridgesim.sim.idealize import Overrides, build_model

    bodies = simple_beam_bridge() + [
        box("Beam M (block)", (-100, 173.1, -60), (100, 203.1, 60)),
        box("Beam M Tenon (block)", (-80, 173.1, 60), (80, 193.1, 75))]
    ov = Overrides.from_dict({"glue": {"Beam M (block) | Beam M Tenon*": "solid"}})
    assert ov.solid == ["Beam M (block) | Beam M Tenon*"] and not ov.glue
    m = build_model(bodies, MAT, SP, ov, seed=1)
    names = [b.name for b in m.bodies]
    sol = [p for p in m.patches if {names[p.body_a], names[p.body_b]}
           == {"Beam M (block)", "Beam M Tenon (block)"}]
    assert len(sol) == 1 and sol[0].kind == "solid"
    k = np.asarray(sol[0].springs)
    fv = MAT.f_v_MPa.value / 1.5
    assert (m.c_tau[k] >= fv - 1e-9).all() and (m.c_tau[k] <= fv * SP["glue.tau_peak_over_avg"]
                                                + 1e-9).all()
    assert np.allclose(m.c_sig[k], MAT.f_t_MPa.value)  # fibres across a tenon root
    assert not np.asarray(m.params["c_d0"])[k].any()
    rep = study._joint_reports(m, MAT)
    j = next(r for r in rep if r.kind == "solid")
    assert j.shear_capacity_N == pytest.approx(sol[0].area_mm2 * fv, rel=1e-3)


def test_butt_joints_with_a_gap_are_out_in_compression_too():
    """A butt joint whose gap closes before the member is damaged bears: up to 0.5 mm
    (Berard et al. 2011: 0.25 and 0.5 mm gaps closed early after first yield in 2 mm
    veneer laminates) the member keeps 0.95 of its crushing strength (Dansoh et al.
    2003), sanded tight (0) or not, as the perfect bridge does; from 1 mm (rounded stick
    ends: ~2) the gap never closes: the open joints' net section, the plies beside them
    weakened as Freas & Selbo's (USDA TB 1069) fig. 61 has it (the literature review of
    round 23: 0.42 for a 13-ply brick pattern, 0.89 with joints 40 mm apart ply to ply);
    linear between."""
    from bridgesim.sim.idealize import Overrides, build_model, butt_open_compression

    assert butt_open_compression(13, 57.5, 115.0, 2.0) == pytest.approx(0.42, abs=0.005)
    assert butt_open_compression(13, 40.0, 115.0, 2.0) == pytest.approx(0.89, abs=0.005)
    assert butt_open_compression(1, 57.5, 115.0, 2.0) == 1.0
    bodies = simple_beam_bridge() + [box("Chord (13-ply UD)", (-300, 200, -13),
                                         (300, 210, 13))]
    fc = MAT.f_c_MPa.value
    kt = 6 / 13 * SP["wood.butt_tension_efficiency"]
    kc = butt_open_compression(13, MAT.stick.length_mm / 2, MAT.stick.length_mm,
                               MAT.stick.thickness_mm)
    eff = SP["wood.butt_bearing_efficiency"]
    for gap, seed, factor in ((0.3, 1, eff), (0.0, 1, eff), (None, None, eff),
                              (0.75, 1, (eff + kc) / 2), (2.0, 1, kc)):
        o = {"butt_gap_mm": gap} if gap is not None else {}
        m = build_model(bodies, MAT, SP, Overrides.from_dict({"bodies": {"Chord*": o}}),
                        seed=seed)
        info = {b.name: b for b in m.bodies}["Chord (13-ply UD)"]
        assert info.f_c_eff_MPa == pytest.approx(factor * fc, rel=0.02), gap
        assert info.f_t_eff_MPa == pytest.approx(kt * MAT.f_t_MPa.value, rel=0.02)


def test_open_butt_joints_sit_at_their_places_as_net_sections():
    """A butt joint that stays open (rounded stick ends) is a short element of the net
    section at its place, its cut plies' fibres removed and the rest at their centroid,
    over the gap and the shear-lag length 2 / lambda (USDA TB 1069; the literature review
    of round 26): ply k's joints at a seeded offset plus k x the stagger, every stick
    length, or where the designer puts them (butt_at); the rest of the member keeps its
    full section and strength, the zones its butt-joint factors over their net share.
    Joints that bear keep the factors over the whole member."""
    from bridgesim.sim.idealize import Overrides, build_model, butt_zones, zone_layers

    z = butt_zones(13, 600.0, 115.0, 57.5, 57.5, 13.0)
    assert [round(x, 1) for x, _c in z] == [57.5, 115.0, 172.5, 230.0, 287.5, 345.0,
                                             402.5, 460.0, 517.5, 575.0][: len(z)]
    assert z[0][1] == frozenset(range(0, 13, 2)) and z[1][1] == frozenset(range(1, 13, 2))
    at = butt_zones(5, 260.0, 115.0, 57.5, 0.0, 13.0, [(0.5, [0, 4])])
    assert at == [(130.0, frozenset({0, 4}))]
    keep, shift = zone_layers(5, frozenset({0, 4}), 5, 10.0, 1.0)
    assert keep.tolist() == pytest.approx([0, 1, 1, 1, 0]) and abs(shift).max() < 1e-9
    keep, shift = zone_layers(5, frozenset({0}), 5, 10.0, 0.5)
    assert keep[0] == pytest.approx(0.5) and keep[1:].tolist() == pytest.approx([1] * 4)
    keep, shift = zone_layers(13, frozenset(range(0, 13, 2)), 5, 26.0, 1.0)
    assert keep.mean() == pytest.approx(6 / 13, abs=0.02)
    bodies = simple_beam_bridge() + [box("Chord (13-ply UD)", (-300, 200, -13),
                                         (300, 210, 13))]
    for gap, n_zones in ((2.0, 5), (0.3, 0)):
        m = build_model(bodies, MAT, SP, Overrides.from_dict(
            {"bodies": {"Chord*": {"butt_gap_mm": gap}}}), seed=1)
        info = {b.name: b for b in m.bodies}["Chord (13-ply UD)"]
        assert len(info.butt_zones) >= n_zones and (n_zones or not info.butt_zones)
        A = np.array([m.b_fA[e].sum() for e in info.beams])
        ft = np.array([m.b_ft[e] for e in info.beams])
        if n_zones:
            assert A.min() < 0.6 * A.max()  # a net section at each joint
            assert ft.max() == pytest.approx(MAT.f_t_MPa.value, rel=0.02)  # full outside
            assert ((A < 0.99 * A.max()) == (ft < 0.99 * ft.max())).all()
        else:
            assert A.min() == pytest.approx(A.max()) and ft.max() < MAT.f_t_MPa.value
    m = build_model(bodies, MAT, SP, Overrides.from_dict({"bodies": {"Chord*": {
        "butt_gap_mm": 2.0, "butt_at": [[0.5, [0, 12]]]}}}), seed=1)
    info = {b.name: b for b in m.bodies}["Chord (13-ply UD)"]
    assert info.butt_zones == [pytest.approx(300.0, abs=0.5)]


def test_faces_drawn_a_little_apart_are_glued_and_asked_about():
    """Two parts that face each other over an overlap but are drawn a few tenths apart (a
    member drawn thinner than the gap it sits in) or one a little inside the other are
    glued as if they touched, up to NEAR_GLUE_MM, and asked about pair by pair
    (near_miss, important); further apart, up to NEAR_ASK_MM, only asked about; not when
    the overrides already leave them unglued, nor when a third part fills the gap (the
    adversarial review of round 26: a 0.4 mm gap silently lost a diagonal's laps and read
    37-49 % low)."""
    from bridgesim.sim.idealize import Overrides, build_model

    def model(gap: float, ov=None):
        bodies = simple_beam_bridge() + [box("Post (4-ply UD)", (300, 173.1 + gap, -4),
                                             (310, 233.1, 4))]
        return build_model(bodies, MAT, SP, Overrides.from_dict(ov or {}), seed=None)

    def post(m):
        names = [b.name for b in m.bodies]
        return [p for p in m.patches if "Post" in names[p.body_a] + names[p.body_b]
                and p.kind == "glue"]  # (a part touching nothing is tied on: 'tie')

    def asked(m):
        return [q for q in m.questions if q["id"].startswith("near_miss")]

    assert post(model(0.0)) and not asked(model(0.0))
    for gap in (0.4, -0.5):  # drawn apart, or one inside the other
        m = model(gap)
        assert post(m), gap
        q = asked(m)
        assert len(q) == 1 and q[0]["severity"] == "important" and "Post" in q[0]["id"]
        assert "glued as if they touched" in q[0]["assumed"]
    m = model(2.0)
    assert not post(m) and asked(m) and "not glued" in asked(m)[0]["assumed"]
    m = model(0.4, {"glue": {"Post* | *": 0}})
    assert not asked(m)


def test_large_hand_made_laps_are_weaker_on_average():
    """A Weibull size effect on a glued lap's average strength: the coupon's area holds
    tau_g, a gusset lap 6.6x as large about 0.83 of it at m = 10; a smaller lap is not
    stronger than the coupon, and the law is not carried past glue.size_cap_mm2 (3000:
    pressed crossings of 1600-29,000 mm^2 showed no size effect, Flaig & Blass 2013);
    the simulated coupon still reproduces tau_g."""
    from bridgesim.sim.idealize import lap_size_factor

    assert SP["glue.weibull_m"] == 10.0 and SP["glue.size_cap_mm2"] == 3000.0
    assert lap_size_factor(200.0) == 1.0 and lap_size_factor(50.0) == 1.0
    assert lap_size_factor(1320.0) == pytest.approx(6.6 ** (-1 / 10), rel=1e-9)
    assert lap_size_factor(1320.0, m=6.0) == pytest.approx(6.6 ** (-1 / 6), rel=1e-9)
    assert lap_size_factor(71370.0) == lap_size_factor(3000.0) == pytest.approx(
        15.0 ** (-1 / 10), rel=1e-9)
    assert lap_size_factor(1320.0, m=0.0) == 1.0
    from bridgesim.sim import study

    rep = study.lab()
    lap = next(t for t in rep.tests if "lap" in t["test"])
    assert lap["tau_avg_MPa"] == pytest.approx(MAT.glue.tau_g_MPa.value, rel=0.03)



def test_the_envelope_does_not_reach_across_a_break():
    """The reading's envelope (the raw force around each moment) stops at a break: a
    bounce after a joint let go, higher than any load before it, is not read back as the
    load the bridge carried (a light truss read 4 % high from its bounce)."""
    from bridgesim.sim.crush import reported_load

    v, T = 20.0, 0.02
    t = np.arange(0, 0.3, 1e-4)
    x = v * t
    tb = 0.2
    true = np.where(t < tb, 1000.0 * t, np.where(t < tb + 0.005, 120.0, 150.0))
    raw = true.copy()
    raw[(t > tb + 0.005) & (t < tb + 0.008)] = 260.0  # the bounce, above the peak (200)
    rep = reported_load(t, raw, v, x, breaks=[tb], period_s=T)
    assert rep[t < tb].max() <= raw[t < tb].max() + 1e-9



def test_values_set_in_overrides_stay_placeholders_unless_measured(tmp_path):
    """A value a team types under params (an assumption, like its report's tau_g 2.5)
    stays a placeholder: still listed as assumed and still drawn by montecarlo around
    it; only the keys listed under measured (what calibrate writes) count as measured.
    The crush's stick modulus (11 GPa) replaces the material file's only while that holds
    its shipped value."""
    from bridgesim.sim import study

    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), cad, names=True)
    doc = {"params": {"material.glue.tau_g_MPa": 2.5, "wood.f_c90_MPa": 5.0}}
    _, _, ov, _, mat, sp, _ = study._inputs(cad, None, None, None, None, doc=doc)
    assumed = set(study._assumed(mat, sp))
    assert {"material.glue.tau_g_MPa", "sim.wood.f_c90_MPa"} <= assumed
    drawn = study.placeholders(mat, sp)
    assert drawn["material.glue.tau_g_MPa"] == 2.5 and drawn["wood.f_c90_MPa"] == 5.0
    doc["measured"] = ["material.glue.tau_g_MPa"]
    _, _, ov, _, mat, sp, _ = study._inputs(cad, None, None, None, None, doc=doc)
    assert "material.glue.tau_g_MPa" not in set(study._assumed(mat, sp))
    assert "material.glue.tau_g_MPa" not in study.placeholders(mat, sp)
    assert mat.E_MPa.value == SP["wood.E_placeholder_MPa"]  # the shipped file: 11 GPa
    own = MAT.model_copy(deep=True)
    own.E_MPa.value = 9000.0  # a team's own (assumed) value in its material file
    assert study._crush_material(own, SP).E_MPa.value == 9000.0


def test_stick_sheets_with_unglued_edges_are_softer_and_weaker_in_shear():
    """A 0/90 sheet of sticks glued only face to face carries in-plane shear through the
    glued crossings of its plies, like cross-laminated timber with unglued narrow faces:
    ~0.68 of its shear stiffness for 2 x 10 mm sticks and half its shear strength
    (Bogensperger 2010), or less where the glued crossings fail in torsion first;
    asked as plate_edges; glued edges (glue.plate_edges_glued 1) keep the solid ply's."""
    from bridgesim.sim.idealize import Overrides, build_model, cross_ply_sheet_shear

    assert cross_ply_sheet_shear([0, 90])[1] == 0.5
    assert cross_ply_sheet_shear([0, 0, 0]) == (1.0, 1.0)
    # the glued crossings twist (Boggian et al. 2019): tau_T = 3 tau t* / a, t* = 4 mm
    # for 2 plies of 2 x 10 mm sticks, so the sheet's shear is capped at 0.83 x the
    # crossing's torsional strength (its bond line, 4 MPa, or 2.4 x rolling shear)
    tor = min(SP["glue.interlaminar_MPa"], 2.4 * SP["wood.f_rolling_MPa"])
    gk, sk = cross_ply_sheet_shear([0, 90], 0.2, tor / MAT.f_v_MPa.value)
    assert 0.6 < gk < 0.75
    assert sk * MAT.f_v_MPa.value == pytest.approx(tor * 10 / 12, rel=1e-9)
    assert sk < 0.5  # below half f_v: 3.3 MPa against 4
    # 3 plies: t* = 3 t / 2, so the crossings hold 1.11 x their strength
    assert cross_ply_sheet_shear([0, 90, 0], 0.2, tor / MAT.f_v_MPa.value)[1] == \
        pytest.approx(min(0.5, tor * 2 / (3 * 3 * 0.2) / MAT.f_v_MPa.value), rel=1e-9)
    m0 = build_model(simple_beam_bridge(), MAT, SP, Overrides(), seed=None)
    m1 = build_model(simple_beam_bridge(), MAT, SP.updated({"glue.plate_edges_glued": 1}),
                     Overrides(), seed=None)
    deck = [b.name for b in m0.bodies].index("Deck (2-ply 0-90)")
    q0, q1 = m0.s_Q[m0.s_body == deck, 3], m1.s_Q[m1.s_body == deck, 3]
    assert np.allclose(q0 / q1, gk)
    lim0 = m0.s_lim[m0.s_body == deck, 4]  # the shear strain at failure (within the band)
    assert np.allclose(lim0, sk * MAT.f_v_MPa.value / (gk * MAT.G_MPa.value), rtol=1e-6)
    assert any(q["id"] == "plate_edges" for q in m0.questions)
    assert not any(q["id"] == "plate_edges" for q in m1.questions)


def test_plates_glued_face_to_face_are_laminations():
    """Two plates glued face to face (a deck's plies drawn as separate bodies) are
    glued as the plies inside one laminated body: at glue.interlaminar_MPa, with no size
    effect, and as stiff as a pressed line (glue.lamination_K_s_N_per_mm3); a lap between
    a member and a plate keeps tau_g times the size factor, carried no further than
    glue.size_cap_mm2. inspect reports each joint's capacity at that strength."""
    from bridgesim.sim.idealize import Overrides, build_model, lap_size_factor

    h = 150.0
    bodies = simple_beam_bridge() + [box("Deck Top (2-ply 0-90)", (-550, h + 23.1, -75),
                                         (550, h + 27.2, 75))]
    m = build_model(bodies, MAT, SP, Overrides(), seed=None)
    names = [b.name for b in m.bodies]
    lam = [p for p in m.patches if {names[p.body_a], names[p.body_b]}
           == {"Deck (2-ply 0-90)", "Deck Top (2-ply 0-90)"}]
    laps = [p for p in m.patches if "Stringer" in names[p.body_a] + names[p.body_b]
            and "Deck (2-ply" in names[p.body_a] + names[p.body_b]]
    assert len(lam) == 1 and lam[0].lamination
    assert lam[0].strength_MPa == pytest.approx(SP["glue.interlaminar_MPa"])
    # (since round 25 in series with the two plates' shear through their thickness, the
    # lower one glued behind on its stringers: half its thickness; G of a 0/90 plate
    # sheared along the span, its 0 ply G_LT, its 90 ply the rolling modulus)
    ks = m.c_Ks[np.array(lam[0].springs)]
    G = 1 / (0.5 / MAT.G_MPa.value + 0.5 / SP["wood.G_rolling_MPa"])
    c_wood = (4.1 / 2 + 4.1) / (3 * G)
    assert np.allclose(ks, 1 / (1 / SP["glue.lamination_K_s_N_per_mm3"] + c_wood), rtol=0.01)
    assert laps and not any(p.lamination for p in laps)
    for p in laps:
        assert p.area_mm2 > SP["glue.size_cap_mm2"]
        # a stringer along the span on a 0/90 deck, on its 0 ply or its 90 ply: the glue
        # line (2 MPa) is weaker than the deck's face across its grain (4 MPa)
        base = MAT.glue.tau_g_MPa.value * lap_size_factor(
            p.area_mm2, SP["glue.size_ref_mm2"], SP["glue.weibull_m"],
            SP["glue.size_cap_mm2"])
        assert p.strength_MPa == pytest.approx(base)
        ks_lap = m.c_Ks[np.array(p.springs)]
        assert np.allclose(ks_lap, ks_lap[0]) and 30 < ks_lap[0] < SP["glue.K_s_N_per_mm3"]
    from bridgesim.sim.study import _joint_reports

    rep = {(j.a, j.b): j for j in _joint_reports(m, MAT)}
    j = next(v for k, v in rep.items() if set(k) == {"Deck (2-ply 0-90)",
                                                      "Deck Top (2-ply 0-90)"})
    assert j.lamination and j.shear_capacity_N == pytest.approx(
        j.area_mm2 * SP["glue.interlaminar_MPa"], rel=1e-3)


def test_the_glue_law_bends_before_its_peak():
    """A PVA line bends over before its peak (Serrano 2000: from 60-70 % of it, peaking
    at 0.05-0.07 mm): the trilinear law is straight to glue.prepeak_onset of the
    strength, peaks at glue.peak_slip_ratio x the straight line's slip, and does the same
    work G_c as the bilinear law (onset 1, ratio 1), which it reproduces exactly; the
    kernel and its NumPy mirror agree; the simulated lap's load-slip curve bends before
    its peak, the bilinear lap's barely."""
    from bridgesim.sim import lab
    from bridgesim.sim.kernels import _trilinear_damage
    from bridgesim.sim.solver import trilinear_damage

    a, p = SP["glue.prepeak_onset"], SP["glue.peak_slip_ratio"]
    assert (a, p) == (0.65, 2.0)
    dm0, k, G = 2.2 / 60.0, 60.0, 1.2
    x = np.linspace(0.0, 1.2, 4001)
    for aa, pp in ((a, p), (1.0, 1.0)):
        d = np.array([_trilinear_damage(v, dm0, k, G, aa, pp) for v in x])
        assert np.allclose(d, trilinear_damage(x, np.full_like(x, dm0), k, G, aa, pp))
        T = (1 - d) * k * x
        assert T.max() == pytest.approx(k * dm0, rel=2e-3)
        assert x[T.argmax()] == pytest.approx(pp * dm0, rel=0.01)
        assert np.trapezoid(T, x) == pytest.approx(G, rel=0.01)
        assert d[x <= aa * dm0].max() == 0.0
    dmf = 2 * G / (k * dm0)
    old = np.clip(dmf * (x - dm0) / (np.maximum(x, 1e-12) * (dmf - dm0)), 0, 1)
    assert np.allclose(old, [_trilinear_damage(v, dm0, k, G, 1.0, 1.0) for v in x])

    from bridgesim.sim.study import coupon_K_s

    kw = {"Kn": SP["glue.K_n_N_per_mm3"],
          "Ks": coupon_K_s(SP, 2.0, SP["wood.E_placeholder_MPa"] / 16.0),
          "GIc": SP["glue.G_Ic_N_per_mm"], "GIIc": SP["glue.G_IIc_N_per_mm"],
          "E": SP["wood.E_placeholder_MPa"]}

    def bend(r):
        """Secant to the peak over the 10-40 % secant: 1 for a straight line."""
        i = int(np.argmax(r.force))
        F, u = r.force[: i + 1], r.disp[: i + 1]
        u10, u40 = np.interp([0.1 * F[-1], 0.4 * F[-1]], F, u)
        return (F[-1] / u[-1]) / (0.3 * F[-1] / (u40 - u10))

    tri = lab.lap_shear(2.2, 0.66, **kw, onset=a, peak_slip=p)
    bil = lab.lap_shear(2.2, 0.66, **kw)
    # (with the coupon's own shear stiffness since round 25, ~140 N/mm^3 against 60, the
    # bilinear lap's shear lag bends it too: 0.95 against the trilinear's 0.93)
    assert bend(tri) < min(0.95, bend(bil) - 0.01)


def test_the_glue_is_as_strong_as_a_test_failing_when_the_crush_does():
    """A PVA line's strength depends on its time to failure, tau(t) = tau(t_ref) (t /
    t_ref)^-n (n 0.1: +26 % per decade faster; PVAc +20 % at 16x the rate, Wernersson
    1994; -40 % at a tenth of the rate, Serrano 2000): a lap that fails 10x sooner than
    the coupons tau_g describes
    holds ~10^0.1 = 1.26x, through the compiled kernel and its NumPy mirror alike; at
    the coupons' own time (or before any physical time has passed) tau_g. The crush's
    glue points carry glue.rate_exponent, a face through one piece the wood's (0.02); the
    crush reports the physical time at F_u; calibrate brings a lap pulled apart in 6 s
    to the coupons' 60 s."""
    from bridgesim.sim import lab
    from bridgesim.sim import solver as S_
    from bridgesim.sim.idealize import Overrides, build_model
    from bridgesim.sim.study import at_coupon_time

    n, tref = SP["glue.rate_exponent"], SP["glue.coupon_time_s"]
    assert (n, tref) == (0.1, 60.0)
    from bridgesim.sim.study import coupon_K_s

    kw = {"Kn": SP["glue.K_n_N_per_mm3"],
          "Ks": coupon_K_s(SP, 2.0, SP["wood.E_placeholder_MPa"] / 16.0),
          "GIc": SP["glue.G_Ic_N_per_mm"], "GIIc": SP["glue.G_IIc_N_per_mm"],
          "E": SP["wood.E_placeholder_MPa"], "rate_exponent": n, "coupon_time_s": tref}
    ref = lab.lap_shear(2.2, 0.66, **kw).peak
    assert lab.lap_shear(2.2, 0.66, **kw, test_time_s=tref).peak == pytest.approx(ref,
                                                                                rel=1e-6)
    fast = lab.lap_shear(2.2, 0.66, **kw, test_time_s=tref / 10).peak
    assert fast / ref == pytest.approx(10 ** n, rel=0.03)
    # the NumPy mirror of the kernel takes the same strength
    m = build_model(simple_beam_bridge(), MAT, SP, Overrides(), seed=None)
    st = S_.initial_state(m, 1e12)
    st.t_test = 6.0
    rate, tnow, tr, t0 = S_.glue_rate(m, st)
    assert (tnow, tr) == (6.0, tref) and np.all(t0 == -1.0)
    assert np.allclose(rate[m.c_breakable], n) and np.allclose(rate[~m.c_breakable], 0.0)
    loaded = np.zeros(len(t0))  # loaded from the start: 6 s to failure
    assert np.allclose(S_.glue_rate_factor(rate, tnow, tr, loaded)[m.c_breakable],
                       10 ** n)
    assert np.all(S_.glue_rate_factor(rate, 0.0, tr, loaded) == 1.0)
    # one piece of wood: the wood's own exponent
    bodies = simple_beam_bridge()
    m2 = build_model(bodies, MAT, SP, Overrides.from_dict(
        {"glue": {"Stringer Near* | Deck*": "solid"}}), seed=None)
    solid = [p for p in m2.patches if p.kind == "solid"]
    assert solid and np.allclose(m2.c_rate[np.array(solid[0].springs)],
                                 SP["wood.rate_exponent"])
    # calibrate: a lap pulled apart in 6 s read (6/60)^0.1 low at the coupons' time
    assert at_coupon_time(1000.0, {"time_to_failure_s": "6"}, SP) == pytest.approx(
        1000.0 * 0.1 ** n)
    assert at_coupon_time(1000.0, {}, SP) == 1000.0


def test_of_joints_breaking_together_the_strongest_governs():
    """A diagonal's end-grain foot (a few hundred N of glue) and its two gusset laps
    (thousands) let go in the same instant: the load falls because the laps go, so the
    laps govern; strengthening the foot alone changed F_u by 1 % on Bridge_v10, the laps
    by 9 % (the adversarial review of round 20). A clean break still comes before a
    joint that only cracked."""
    from bridgesim.sim.crush import Event, attribute

    peak = 20000.0
    ev = [Event(0.0700, 2.0, 0.99 * peak, "joint_failure", "Diag R | Stub", capacity_N=550.0),
          Event(0.0701, 2.0, 0.99 * peak, "joint_failure", "Diag R | Gusset Inner",
                capacity_N=6000.0),
          Event(0.0702, 2.0, 0.99 * peak, "joint_failure", "Diag R | Gusset Outer",
                capacity_N=5800.0),
          Event(0.0701, 2.0, 0.99 * peak, "joint_cracking", "Diag R | Big", capacity_N=9e3)]
    gov, competing = attribute(ev, 0.0699, peak, 2.0, 100.0)
    assert gov.what == "Diag R | Gusset Inner"
    assert "joint failure: Diag R | Stub" in competing
    # a plate cracking in the same instant, after the joints let go, does not take
    # over (it has no glue capacity: round-21 adversarial review), and one cracking
    # first keeps its place
    late = ev + [Event(0.0702, 2.0, 0.99 * peak, "plate_crack", "Deck")]
    gov, _ = attribute(late, 0.0699, peak, 2.0, 100.0)
    assert gov.what == "Diag R | Gusset Inner"
    early = [Event(0.06995, 2.0, peak, "plate_crack", "Deck")] + ev
    gov, competing = attribute(early, 0.0699, peak, 2.0, 100.0)
    assert gov.what == "Deck"
    assert "joint failure: Diag R | Gusset Inner" in competing
    # the adversarial review of round 24: on Bridge_v10 the laps let go 2.2-3.3 ms after
    # the foot cracked (standard, seed 0), outside the load cell's window; within half a
    # period of the bridge's ringing (7 ms) they are the same moment, and the laps govern
    r24 = [Event(0.11592, 3.0, 0.99 * peak, "joint_cracking", "Knee-Diagonal R Far | Stub",
                 capacity_N=870.0),
           Event(0.11811, 3.0, 0.98 * peak, "joint_failure",
                 "Knee-Diagonal R Far | plate_M Far Inner", capacity_N=2588.0),
           Event(0.11827, 3.0, 0.97 * peak, "joint_failure",
                 "Knee-Diagonal R Far | plate_M Far Outer", capacity_N=1990.0)]
    gov, _ = attribute(r24, 0.1159, peak, 3.0, 100.0)
    assert gov.what == "Knee-Diagonal R Far | Stub"  # without the period: the foot
    gov, competing = attribute(r24, 0.1159, peak, 3.0, 100.0, period_s=0.007)
    assert gov.what == "Knee-Diagonal R Far | plate_M Far Inner"
    assert "joint failure: Knee-Diagonal R Far | plate_M Far Outer" in competing
    # a plate that had been cracking since before the peak (seed 6 of round 24's Monte
    # Carlo) does not keep its place before joints letting go with it
    seed6 = [Event(0.0937, 2.5, 0.93 * peak, "plate_crack", "plate_M Far Inner"),
             Event(0.11363, 3.1, 0.9 * peak, "plate_crack", "plate_M Far Inner"),
             Event(0.11434, 3.1, 0.9 * peak, "joint_failure", "End Post | gus_A",
                   capacity_N=5000.0)]
    gov, _ = attribute(seed6, 0.1135, peak, 3.1, 100.0, period_s=0.007)
    assert gov.what == "End Post | gus_A"


def test_laps_are_capped_by_what_the_laminate_passes_on_and_short_overlaps_are_laps():
    """A lap glued on a ply face of a laminate is no stronger than the laminate's first
    glue line lets the load into its other plies: n / (n - 1) x its interlaminar strength
    (2 x the rolling shear of a 2-ply 0/90 gusset), however strong the glue; a plate
    overlapping another over a quarter of it (a tongue slid into its groove on build day)
    is a hand-made lap, not a lamination; the model-form error is median-centred (the
    model as likely to read high as low)."""
    from bridgesim.sim.idealize import Overrides, build_model, interlaminar_strength
    from bridgesim.sim.study import _with_error

    strong = MAT.model_copy(deep=True)
    strong.glue.tau_g_MPa.value = 30.0
    m = build_model(simple_beam_bridge(), strong, SP, Overrides(), seed=None)
    names = [b.name for b in m.bodies]
    pair = [(names[p.body_a] + names[p.body_b], p) for p in m.patches]
    # on the deck's ply face (2-ply 0/90): 2 x its interlaminar strength (the stringers
    # are glued on their edge, through no glue line of theirs), or, on its 90 ply, that
    # ply sheared across its grain (rolling shear: the lap's face cap, round 25)
    il = interlaminar_strength([0, 90], MAT.f_v_MPa.value, SP["wood.f_rolling_MPa"],
                               SP["glue.interlaminar_MPa"])
    deck = [p for k, p in pair if "Deck" in k]
    # (a deck's first-named ply is its bottom one: the stringers under it glue on the 0
    # ply of a 0-90 deck, along their grain)
    assert len(deck) == 2 and all(
        p.strength_MPa == pytest.approx(min(2 * il, MAT.f_v_MPa.value)) for p in deck)
    # a pier block on a stringer's edge: no laminate to pass through, tau_g x size, at
    # most the stringer's shear strength along its grain (the force runs along it)
    piers = [p for k, p in pair if "Pier" in k]
    assert piers and all(p.strength_MPa == pytest.approx(MAT.f_v_MPa.value) for p in piers)
    h = 150.0
    bodies = simple_beam_bridge() + [box("Tongue (2-ply 0-90)", (500, h + 23.1, -75),
                                         (700, h + 27.2, 75))]
    m2 = build_model(bodies, MAT, SP, Overrides(), seed=None)
    names2 = [b.name for b in m2.bodies]
    t = [p for p in m2.patches if "Tongue" in names2[p.body_a] + names2[p.body_b]]
    assert len(t) == 1 and not t[0].lamination
    assert np.median(_with_error([1000.0], 0.3)) == pytest.approx(1000.0)


def test_a_block_can_be_given_its_grain_and_butt_joints_cost_stiffness():
    """A post's end bearing on a block bears at the block's crushing strength across the
    grain (no grain known: f_c90), or along it once the block's fibres are given along the
    post (grain: y; a heel seat whose layers run along the post: end-grain bearing, the
    realism review of round 20). A laminated member longer than a stick is softer than
    one without butt joints (shear lag round each joint: ~0.88 for 13 plies of 2 mm)."""
    from bridgesim.sim import solver as S_
    from bridgesim.sim.idealize import Overrides, build_model, butt_stiffness_factor

    post = box("Post (3-ply UD)", (-530, 150, -5), (-520, 262, 5))
    fc = {}
    for g in (None, "y"):
        o = {"bodies": {"Pier L*": {"grain": g}}} if g else {}
        m = build_model(simple_beam_bridge() + [post], MAT, SP, Overrides.from_dict(o),
                        seed=None)
        names = [b.name for b in m.bodies]
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Pier L (block)", "Post (3-ply UD)"})
        fc[g] = float(S_._glue_fc90(m)[np.array(p.springs)].min())
    assert fc[None] == pytest.approx(SP["wood.f_c90_MPa"], rel=0.02)
    assert fc["y"] == pytest.approx(MAT.f_c_MPa.value, rel=0.05)
    # one direction per layer: half along the post, half across it
    o = {"bodies": {"Pier L*": {"grain": [[0, 1, 0], [1, 0, 0]]}}}
    m = build_model(simple_beam_bridge() + [post], MAT, SP, Overrides.from_dict(o), seed=None)
    names = [b.name for b in m.bodies]
    p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
             == {"Pier L (block)", "Post (3-ply UD)"})
    # (its layer across the post: a 10 mm face 20 mm from either end of the 50 mm pier,
    # not borne behind it (0.4 x its 150 mm depth): the fibres beside the face help,
    # f_c90 x sqrt(50 / 10), prEN 1995 since round 25)
    assert float(S_._glue_fc90(m)[np.array(p.springs)].min()) == pytest.approx(
        (MAT.f_c_MPa.value + SP["wood.f_c90_MPa"] * math.sqrt(5.0)) / 2, rel=0.05)
    bad = Overrides.from_dict({"bodies": {"X": {"grain": "w"}}}).problems(["X"])
    assert any("grain" in b for b in bad)
    assert not [b for b in Overrides.from_dict({"bodies": {"X": {"grain": [0.7, 0.7, 0]}}})
                .problems(["X"]) if "grain" in b]
    k13 = butt_stiffness_factor(13, 600.0, 115.0, None, 10500.0, 2.0, 200.0)
    assert 0.85 < k13 < 0.9
    assert butt_stiffness_factor(13, 100.0, 115.0, None, 10500.0, 2.0, 200.0) == 1.0
    assert butt_stiffness_factor(13, 600.0, 115.0, 0, 10500.0, 2.0, 200.0) == 1.0
    chord = box("Chord (13-ply UD)", (-300, 200, -13), (300, 210, 13))
    m = build_model(simple_beam_bridge() + [chord], MAT, SP, Overrides(), seed=None)
    info = {b.name: b for b in m.bodies}["Chord (13-ply UD)"]
    E = MAT.E_MPa.value  # the model's own (this material, not the crush's placeholder)
    assert info.E_eff_MPa == pytest.approx(E * butt_stiffness_factor(
        13, 600.0, MAT.stick.length_mm, None, E, MAT.stick.thickness_mm,
        SP["glue.lamination_K_s_N_per_mm3"]), rel=0.03)



def test_an_end_grain_glue_line_pulled_apart_holds_a_share_of_the_wood_along_its_grain():
    """A post glued by its end grain onto a block and pulled off: the line holds
    glue.end_grain_tension_factor x the wood's tension strength along the grain (the
    fibres run along the pull; end-grain butt joints reach ~25 % of f_t with special
    techniques, USDA Wood Handbook; a hand-made PVA vertical's foot held 2.6 MPa in a
    published truss test), not the side-grain peel strength x the end-grain factor."""
    from bridgesim.sim.idealize import Overrides, build_model

    post = box("Post (3-ply UD)", (-530, 150, -5), (-520, 262, 5))
    sig = {}
    for g in ("y", None):  # the pier's fibres along the post (end grain on end grain), or
        o = {"bodies": {"Pier L*": {"grain": g}}} if g else {}  # no grain (side grain)
        m = build_model(simple_beam_bridge() + [post], MAT, SP, Overrides.from_dict(o),
                        seed=None)
        names = [b.name for b in m.bodies]
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Pier L (block)", "Post (3-ply UD)"})
        sig[g] = m.c_sig[np.array(p.springs)]
    assert np.allclose(sig["y"], SP["glue.end_grain_tension_factor"] * MAT.f_t_MPa.value,
                       rtol=0.05)
    # on a face pulled across its grain, that face lifts first (f_t90, raised for the
    # small volume a 100 mm^2 joint pulls on: t90_size_factor)
    from bridgesim.sim.idealize import t90_size_factor

    assert np.allclose(sig[None], SP["wood.f_t90_MPa"] * t90_size_factor(
        100.0, SP["wood.t90_weibull_m"]), rtol=0.05)


def _two_sticks(cross: bool):
    """Two 10 mm wide sticks glued face to face over 10 x 10 mm, along each other or
    crossing at 90 degrees, beside the simple bridge."""
    a = box("Stick A (1-ply UD)", (600, 0, 100), (720, 2, 110))
    b = (box("Stick B (1-ply UD)", (650, 2, 50), (660, 4, 170)) if cross else
         box("Stick B (1-ply UD)", (710, 2, 100), (830, 4, 110)))
    return [a, b]


def test_small_joints_hold_more_across_the_grain_and_crossed_laps_no_more_than_the_wood():
    """The round-21 literature review: the wood's tension across the grain where a small
    joint pulls on it follows a weakest-link size effect (birch veneer m ~5-11, Kruger et
    al. 2025; Barrett 1974), 1.64 x f_t90 for a 47 mm^2 vertical's foot at m 8 (one held
    2.6 MPa, Moraes et al. 2019), never below f_t90. The round-23 literature review: a lap
    holds the lower of its glue line and the ply it lies on sheared at its angle (birch
    veneer, Wang et al. 2024: 0.69 / 0.62 / 0.66 / 0.72 of f_v at 30 / 45 / 60 / 90, or
    Hankinson's form down to the rolling shear), so a weak glue line holds as much
    crossed as parallel, a strong one less crossed."""
    from bridgesim.sim.idealize import (
        Overrides,
        build_model,
        face_shear_strength,
        t90_size_factor,
    )

    assert t90_size_factor(47.0, 8.0) == pytest.approx(1.635, abs=0.01)
    assert t90_size_factor(2400.0, 8.0) == 1.0 and t90_size_factor(1e5, 8.0) == 1.0
    assert t90_size_factor(0.5, 8.0) == 2.5 and t90_size_factor(47.0, 0.0) == 1.0
    for deg, f in ((0, 8.0), (30, 0.69 * 8), (45, 0.625 * 8), (60, 32 / 7), (90, 4.0)):
        c = math.cos(math.radians(deg))
        assert face_shear_strength(8.0, 4.0, c) == pytest.approx(f, abs=0.03)
    for tau_g, ratio in ((MAT.glue.tau_g_MPa.value, 1.0), (6.0, None)):
        mat = MAT.model_copy(deep=True)
        mat.glue.tau_g_MPa.value = tau_g
        st = {}
        for cross in (False, True):
            m = build_model(simple_beam_bridge() + _two_sticks(cross), mat, SP,
                            Overrides(), seed=None)
            names = [b.name for b in m.bodies]
            p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                     == {"Stick A (1-ply UD)", "Stick B (1-ply UD)"})
            assert p.area_mm2 == pytest.approx(100.0, rel=0.05)
            st[cross] = p.strength_MPa
        if ratio is not None:  # the glue line governs either way
            assert st[True] == pytest.approx(st[False], rel=1e-6)
        else:  # the face across its grain: rolling shear (a 10 mm lap: no panel factor)
            assert st[True] == pytest.approx(SP["wood.f_rolling_MPa"], rel=1e-6)
            assert st[False] > 1.4 * st[True]


def test_a_crossed_lap_takes_the_ply_the_glue_is_on():
    """The realism review of round 22: a stick glued along a 0/90 gusset's 0 ply is a lap
    of parallel sticks, on its 90 ply a crossed one, not the average of the two whatever
    face it is on (with a glue line strong enough for the wood to govern)."""
    from bridgesim.sim.idealize import Overrides, build_model

    mat = MAT.model_copy(deep=True)
    mat.glue.tau_g_MPa.value = 6.0
    g = box("Gusset (2-ply 0-90)", (600, 10, 50), (700, 14, 110))
    top = box("Stick T (1-ply UD)", (610, 14, 75), (690, 16, 85))
    bot = box("Stick B (1-ply UD)", (610, 8, 75), (690, 10, 85))
    m = build_model(simple_beam_bridge() + [g, top, bot], mat, SP, Overrides(), seed=None)
    names = [b.name for b in m.bodies]
    st = sorted(p.strength_MPa for p in m.patches
                if "Gusset (2-ply 0-90)" in (names[p.body_a], names[p.body_b]))
    assert len(st) == 2
    assert st[0] == pytest.approx(SP["wood.f_rolling_MPa"], rel=1e-6)
    assert st[1] > 1.2 * st[0]


def test_a_wide_lap_on_a_thin_gusset_holds_its_panel_factor():
    """The round-23 literature review: a gusset passes less rolling shear under a member
    lapped on it the wider the member is against the gusset's thickness, (8 h_f / b_w)^0.8
    inside the gusset and (4 h_f / b_w)^0.8 at its edge (EN 1995-1-1 eq. 9.14; APA PDS
    Supplement 3 3.6.4: half at the edge stringers); a stick lapped on a stick of its own
    width is sheared evenly (no factor)."""
    from bridgesim.sim.idealize import Overrides, build_model, panel_factor

    assert panel_factor(4.0, 51.5, 2) == pytest.approx(0.68, abs=0.01)
    assert panel_factor(4.0, 51.5, 1) == pytest.approx(0.39, abs=0.01)
    assert panel_factor(4.0, 20.0, 2) == 1.0 and panel_factor(4.0, 51.5, 0) == 1.0
    mat = MAT.model_copy(deep=True)
    mat.glue.tau_g_MPa.value = 6.0
    fr = SP["wood.f_rolling_MPa"]
    # a 1-ply gusset 2 mm thick, its grain along its length z (90 degrees to the members
    # along x) under a member 20 mm wide: inside it (reaching 90 mm past each side) and
    # at its edge
    gus = box("Gusset (1-ply UD)", (600, 10, 0), (760, 12, 200))
    mid = box("Member M (3-ply UD)", (610, 12, 90), (750, 18, 110))
    edge = box("Member E (3-ply UD)", (610, 4, 180), (750, 10, 200))
    m = build_model(simple_beam_bridge() + [gus, mid, edge], mat, SP, Overrides(),
                    seed=None)
    names = [b.name for b in m.bodies]
    st = {names[p.body_a] + names[p.body_b]: p.strength_MPa for p in m.patches
          if "Gusset (1-ply UD)" in (names[p.body_a], names[p.body_b])}
    s_mid = next(v for k, v in st.items() if "Member M" in k)
    s_edge = next(v for k, v in st.items() if "Member E" in k)
    assert s_mid == pytest.approx(fr * panel_factor(2.0, 20.0, 2), rel=0.02)
    assert s_edge == pytest.approx(fr * panel_factor(2.0, 20.0, 1), rel=0.02)
    assert s_edge < s_mid < fr


def test_the_glue_scatter_never_lifts_the_wood_past_its_strength():
    """The joint-to-joint scatter is the glue's (workmanship): a joint drawn 1.5x strong
    has a stronger glue line, but the wood it pulls on across the grain is no stronger
    for it (the adversarial review of round 21: peel drawn at 1.5 x f_t90); a user's
    glue factor above 1 (a what-if's bigger glued area) scales both."""
    from bridgesim.sim.idealize import Overrides, build_model, t90_size_factor

    post = box("Post (3-ply UD)", (-530, 150, -5), (-520, 262, 5))
    cap = SP["wood.f_t90_MPa"] * t90_size_factor(100.0, SP["wood.t90_weibull_m"])
    noisy = SP.updated({"scatter.glue_cov": 0.8})
    tops = []
    for seed in range(1, 9):
        m = build_model(simple_beam_bridge() + [post], MAT, noisy, Overrides(), seed=seed)
        names = [b.name for b in m.bodies]
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Pier L (block)", "Post (3-ply UD)"})
        tops.append(float(m.c_sig[np.array(p.springs)].max()))
    assert max(tops) <= cap * 1.0001 and min(tops) < 0.9 * cap
    m = build_model(simple_beam_bridge() + [post], MAT, SP,
                    Overrides.from_dict({"glue": {"Pier L* | Post*": 2.0}}), seed=None)
    names = [b.name for b in m.bodies]
    p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
             == {"Pier L (block)", "Post (3-ply UD)"})
    assert float(m.c_sig[np.array(p.springs)].max()) == pytest.approx(2 * cap, rel=0.05)


def test_studies_of_a_light_bridge_are_suggested_on_a_fine_enough_mesh():
    """The adversarial review of round 21: the next steps sent a light bridge's diagnose
    to quick and its montecarlo to standard, meshes its own crush flags. The quality
    names the coarsest fidelity fine enough for the members that fail (min_fidelity)."""
    from bridgesim.cad.synthetic import layered_truss
    from bridgesim.sim import study
    from bridgesim.sim.idealize import Overrides, build_model

    m = build_model(layered_truss(), MAT, SP, Overrides(), seed=None, mesh_scale=0.5)
    assert study._quality(_stub_result(m), MAT, SP).min_fidelity == "fine"
    deep = [box("Pier L (block)", (-550, 0, -40), (-500, 150, 40)),
            box("Pier R (block)", (500, 0, -40), (550, 150, 40)),
            box("Beam (13-ply UD)", (-550, 150, -13), (550, 180, 13))]
    m = build_model(deep, MAT, SP, Overrides(), seed=None, mesh_scale=1.6)
    assert study._quality(_stub_result(m), MAT, SP).min_fidelity == "quick"


def test_the_woods_moisture_is_drawn_additively():
    """A placeholder whose value is 0 (the wood's moisture above its coupons') cannot be
    drawn lognormally around it: it is drawn as max(0, sd x z), about half the runs dry
    and the others a few points wetter (members laminated in the days before the test)."""
    from bridgesim.sim.study import draw_placeholders

    v = [draw_placeholders({}, {"wood.moisture_excess_pct": 0.0}, s, 8)["params"][
        "wood.moisture_excess_pct"] for s in range(1, 9)]
    assert min(v) == 0.0 and 3 <= sum(x == 0.0 for x in v) <= 5
    assert 1.0 < max(v) < 6.0


def test_the_reading_window_fits_a_short_rise():
    """The realism review of round 21: a 0.9 mm smoothing window was longer than a
    stiff little truss's whole rise (0.4-1.1 mm) and read 3-7 kgf before the jack had
    touched it. The window is a quarter of the travel to the raw peak there, never less
    than a period of the ringing, and 0.9 mm on a bridge that rises over several mm."""
    from bridgesim.sim.crush import LOAD_CELL_MM, reading_window

    x = np.linspace(0.0, 1.2, 241)
    raw = np.clip(x - 0.4, 0.0, None) * 1000.0
    raw[x > 1.1] = 0.0
    W = reading_window(raw, x, 100.0, period_s=0.0) * 100.0
    assert W == pytest.approx(0.25 * 1.1, rel=0.03)
    assert reading_window(raw, x, 100.0, period_s=0.004) * 100.0 == pytest.approx(0.4)
    xl = np.linspace(0.0, 12.0, 241)
    rl = xl * 100.0
    assert reading_window(rl, xl, 100.0) * 100.0 == pytest.approx(3 * LOAD_CELL_MM)


def test_a_partial_lamination_goes_over_from_a_lap_smoothly():
    """Two plates glued face to face are a lamination when one covers most of the other
    and a hand-made lap when it covers little; in between the joint goes over gradually
    (LAMINATION_RAMP), so a gusset covering 45 % or 55 % of itself does not jump 2.5x
    (the adversarial review of round 21)."""
    from bridgesim.sim.idealize import LAMINATION_RAMP, Overrides, build_model

    h = 150.0
    st = []
    for x1 in (-330, -110, 0, 110, 330, 550):
        bodies = simple_beam_bridge() + [box("Over (2-ply 0-90)", (-550, h + 23.1, -75),
                                             (x1, h + 27.2, 75))]
        m = build_model(bodies, MAT, SP, Overrides(), seed=None)
        names = [b.name for b in m.bodies]
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Deck (2-ply 0-90)", "Over (2-ply 0-90)"})
        st.append(p.strength_MPa)
    assert st[-1] == pytest.approx(SP["glue.interlaminar_MPa"])
    assert all(b >= a - 1e-9 for a, b in zip(st, st[1:], strict=False))
    # neighbouring covers (a tenth to a fifth of the deck apart) never 2.5x apart
    assert max(b / a for a, b in zip(st, st[1:], strict=False)) < 1.75
    assert LAMINATION_RAMP[0] < 0.5 < LAMINATION_RAMP[1]


def test_ab_names_the_bodies_whose_scatter_is_drawn_afresh():
    """Each body's and joint's scatter is seeded by its bodies' names, so a body renamed
    between two versions (SolidWorks renames unnamed bodies after a feature edit) draws
    new luck: ab says which (the adversarial review of round 21); bodies only added
    change nothing of A's draws and are not listed."""
    from bridgesim.sim.study import renamed_bodies

    assert renamed_bodies(["A", "B", "C"], ["A", "B", "C", "D"]) == []
    assert renamed_bodies(["A", "B", "C"], ["A", "B", "Body 7"]) == ["Body 7"]


def test_a_strap_glued_flat_goes_over_to_a_lamination_with_its_width():
    """A 100 mm strap glued flat over its whole face on the deck: one stick wide (10 mm) it
    is a hand-made lap like tau_g's coupon, three sticks wide (an effective width
    2 A / perimeter of 30 mm and more) a lamination, in between partly each, whatever it
    is meshed as (the adversarial review
    of round 22: 2192 N as a frame at 20 mm against 10,500 N as a shell at 30 mm)."""
    from bridgesim.sim.idealize import Overrides, build_model

    h = 150.0 + 23.1
    kinds, st = {}, {}
    for w in (10.0, 20.0, 50.0):
        pad = box("Pad (2-ply UD)", (100, h, -w / 2), (200, h + 4.1, w / 2))
        m = build_model(simple_beam_bridge() + [pad], MAT, SP, Overrides(), seed=None)
        names = [b.name for b in m.bodies]
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Deck (2-ply 0-90)", "Pad (2-ply UD)"})
        kinds[w] = m.bodies[names.index("Pad (2-ply UD)")].kind
        st[w] = (p.lamination, p.strength_MPa)
    from bridgesim.sim.idealize import panel_factor

    assert st[50.0][0] and not st[10.0][0]
    # (50 mm wide on the 4.1 mm deck, inside it: the deck's panel factor, round 25)
    assert st[50.0][1] == pytest.approx(SP["glue.interlaminar_MPa"]
                                        * panel_factor(4.1, 50.0, 2), rel=0.02)
    assert st[10.0][1] < st[20.0][1]
    assert st[50.0][1] / st[20.0][1] < 2.0 and st[20.0][1] / st[10.0][1] < 2.0


def test_each_glue_line_keeps_its_own_clock():
    """The realism review of round 22: a glue line's strength follows its own time under
    load, not the test's elapsed time: a line that picks up load only when a neighbour
    breaks, 9 minutes into a test, is loaded fast (as strong as a lap failing in seconds),
    one loaded since the start is as old as the test. Each line's clock starts when it
    takes RATE_LOAD of its strength (the NumPy mirror of the kernel)."""
    from bridgesim.sim import solver as S_
    from bridgesim.sim.idealize import Overrides, build_model

    rate = np.full(3, 0.1)
    f = S_.glue_rate_factor(rate, 600.0, 60.0, np.array([0.0, 540.0, 599.5]))
    # (a line loaded 0.5 s before is taken no faster than 16x its coupons, the law's data,
    # since round 26: the adversarial review of round 25)
    assert f == pytest.approx([10 ** -0.1, 1.0, 16 ** 0.1], rel=1e-6)
    m = build_model(simple_beam_bridge(), MAT, SP, Overrides(), seed=None)
    st = S_.initial_state(m, 1e12)
    st.t_test = 540.0
    F, Mo = np.zeros((m.n_nodes, 3)), np.zeros((m.n_nodes, 3))
    S_.glue_forces(m, st, F, Mo)
    assert np.all(st.c_t0 == -1.0)  # nothing loaded yet
    deck = [b.name for b in m.bodies].index("Deck (2-ply 0-90)")
    st.x[m.node_body == deck, 1] += 0.05  # the deck lifted off its stringers
    S_.glue_forces(m, st, F, Mo)
    on = st.c_t0 >= 0.0
    assert on.any() and np.all(st.c_t0[on] == 540.0)


def _boxes(specs, d=(0.0, 0.0, 0.0)):
    """Boxes (name, lo, hi), all moved by d (mm)."""
    d = np.asarray(d, float)
    return [box(n, tuple(np.asarray(lo, float) + d), tuple(np.asarray(hi, float) + d))
            for n, lo, hi in specs]


_SIMPLE = [("Pier L (block)", (-550, 0, -75), (-500, 150, 75)),
           ("Pier R (block)", (500, 0, -75), (550, 150, 75)),
           ("Stringer Near (3-ply UD)", (-550, 150, 55), (550, 169, 61)),
           ("Stringer Far (3-ply UD)", (-550, 150, -61), (550, 169, -55)),
           ("Deck (2-ply 0-90)", (-550, 169, -75), (550, 173.1, 75))]


def test_a_face_pressed_across_the_grain_is_softer_than_end_grain():
    """The realism review of round 23: every glue face had one normal stiffness, a side-
    grain face as stiff as end grain. The wood on either side is added in series, its
    modulus at its grain's angle to the face (E_T across, E_L along) over the depth the
    stress reaches (its extent, or half the face's width): a stringer's side-grain face
    on a pier block is far softer than the constant, a post's end grain on a block whose
    fibres run along it nearly as stiff."""
    from bridgesim.sim.idealize import Overrides, bearing_depth, build_model

    post = box("Post (3-ply UD)", (-530, 150, -5), (-520, 262, 5))
    m = build_model(simple_beam_bridge() + [post], MAT, SP, Overrides.from_dict(
        {"bodies": {"Pier L*": {"grain": "y"}}}), seed=None)
    names = [b.name for b in m.bodies]

    def kn(a, b):
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]} == {a, b})
        return float(np.median(m.c_Kn[np.array(p.springs)])), p

    k_end, p_end = kn("Pier L (block)", "Post (3-ply UD)")
    k_side, p_side = kn("Pier R (block)", "Stringer Near (3-ply UD)")
    K = SP["glue.K_n_N_per_mm3"]
    assert k_side < 0.4 * K < k_end <= K
    # round 25 (the literature review of round 24: Blass & Gorlacher's spread): the
    # stringer's side grain, a face 50 mm along its grain spanning its 6 mm width, the
    # stress spreading at 45 degrees along the grain through its 19 mm depth (it is borne
    # on its top face by the deck there); the pier (no grain: across), 150 mm deep and
    # not borne behind (half its depth), spreading both ways at wood.bearing_spread_across
    assert p_side.area_mm2 == pytest.approx(300.0, rel=0.02)
    c = SP["wood.bearing_spread_across"]
    h_s = bearing_depth(50.0, 6.0, 19.0, 1.0, 0.0)
    h_p = bearing_depth(math.sqrt(300.0), math.sqrt(300.0), 75.0, c, c)
    assert h_s == pytest.approx(25.0 * math.log(1 + 38.0 / 50.0), rel=1e-3)
    E_T = SP["wood.E_T_MPa"]
    assert k_side == pytest.approx(1 / (1 / K + h_s / E_T + h_p / E_T), rel=0.05)


def test_a_bar_can_be_told_which_way_it_spans():
    """The adversarial review of round 23: a floor beam wider (200 mm) than it is long
    between its trusses (162 mm) was laid along its outline's length, the wrong way, and no
    override set a bar's axis: `axis` does."""
    from bridgesim.sim.idealize import Overrides, build_model

    beam = box("Floor Beam (block)", (300, 150, -81), (500, 180, 81))
    for ax, want in ((None, 0), ("z", 2)):
        o = {"bodies": {"Floor Beam*": {"kind": "bar", "layup": "1-ply UD",
                                        **({"axis": ax} if ax else {})}}}
        m = build_model(simple_beam_bridge() + [beam], MAT, SP, Overrides.from_dict(o),
                        seed=None)
        info = next(b for b in m.bodies if b.name == "Floor Beam (block)")
        assert info.kind == "bar" and int(np.argmax(np.abs(info.axis))) == want
    assert Overrides.from_dict({"bodies": {"Floor Beam*": {"axis": "q"}}}).problems(
        ["Floor Beam (block)"])


def test_where_the_bridge_sits_in_the_cad_changes_no_joint():
    """The adversarial review of round 23: which ply of a 0/90 gusset a member is glued on
    followed the CAD origin (moving a bridge 300 mm changed 37 joints by up to 16 %), and a
    joint's random draw followed its centre rounded to 10 mm (halves round to even: a 5 mm
    shift redrew 15 joints). A plate's plies are listed from the face toward the bridge's
    middle, and a draw follows the joint's rank between its two bodies."""
    from bridgesim.sim.idealize import Overrides, build_model

    specs = _SIMPLE + [("Gusset (2-ply 0-90)", (600, 10, 50), (700, 14, 110)),
                       ("Stick T (1-ply UD)", (610, 14, 75), (690, 16, 85)),
                       ("Stick B (1-ply UD)", (610, 8, 75), (690, 10, 85)),
                       ("Pad A (1-ply UD)", (-300, 173.1, -20), (-250, 175.1, 20)),
                       ("Pad B (1-ply UD)", (250, 173.1, -20), (300, 175.1, 20))]
    caps = []
    # (the last: a mast far off raises the bridge's mid-height above the deck, which
    # turned the deck's plies over until round 25, the adversarial review of round 24)
    for d, extra in (((0, 0, 0), []), ((0, 0, 300.0), []), ((5.0, 5.0, 5.0), []),
                     ((0, 0, 0), [("Mast (3-ply UD)", (900, 0, -3), (910, 600, 3))])):
        m = build_model(_boxes(specs + extra, d), MAT, SP, Overrides(), seed=1)
        names = [b.name for b in m.bodies]
        caps.append(sorted((names[p.body_a], names[p.body_b], round(p.area_mm2),
                            round(float(m.c_tau[np.array(p.springs)].mean()), 6))
                           for p in m.patches))
    assert caps[0] == caps[1] == caps[2] == caps[3]
    # a sliver of contact gained beside a joint does not redraw it (round 24: a 16 mm^2
    # sliver reseeded the main joint between two bodies)
    from bridgesim.cad.synthetic import upright

    taus = []
    for h2 in (30.0, 29.0):  # the second post touching the chord over 2 x 18 mm, or not
        u = upright("Seat (9-ply UD)", [(0, 0), (60, 0), (60, h2), (58, h2), (58, 10),
                                        (25, 10), (25, 30), (0, 30)], 0, 18)
        chord = box("Chord (9-ply UD)", (-50, 30, 0), (110, 50, 18))
        m = build_model(simple_beam_bridge() + [u, chord], MAT, SP, Overrides(), seed=1)
        names = [b.name for b in m.bodies]
        ps = sorted((p for p in m.patches if {names[p.body_a], names[p.body_b]}
                     == {"Seat (9-ply UD)", "Chord (9-ply UD)"}), key=lambda p: -p.area_mm2)
        assert len(ps) == (2 if h2 == 30.0 else 1)
        taus.append(float(m.c_tau[np.array(ps[0].springs)].mean()))
    assert taus[0] == pytest.approx(taus[1], rel=1e-9)


def test_a_light_truss_is_never_given_more_mass_than_the_limit():
    """The adversarial review of round 23: every run of a light truss was flagged at
    5.0-5.6 % added mass, the last scaling round passing the 5 % limit; a round that would
    pass it is not taken and the time step comes down instead."""
    from bridgesim.cad.synthetic import layered_truss
    from bridgesim.sim import solver as S_
    from bridgesim.sim.crush import stable_dt
    from bridgesim.sim.idealize import Overrides, build_model

    m = build_model(layered_truss(), MAT, SP, Overrides(), seed=None, mesh_scale=0.5)
    s = S_.Settings(dt=1e-6)
    dt, mass, _rot, _log = stable_dt(m, s, dt_target=4e-6, max_added=0.01)
    assert (mass - m.mass).sum() / m.mass.sum() <= 0.01 + 1e-9
    assert dt > 0


def test_the_crusher_pace_is_asked_for(tmp_path):
    """The adversarial review of round 23: the glue's rate law runs on a crusher pace
    nobody has timed, and no question asked for it."""
    from bridgesim.sim import study

    cad = tmp_path / "b.stl"
    write_stl(simple_beam_bridge(), cad, names=True)
    rep = study.inspect(cad)
    q = next(q for q in rep.questions if q.id == "crusher:pace")
    assert "crusher.test_rate_mm_per_min" in q.answer_with["params"]


def test_an_stl_keeps_separate_coplanar_faces_apart(tmp_path):
    """The adversarial review of round 24: STL input put all coplanar triangles of a body
    into one face, whose second region (the top of a U-shaped block's second post) was
    read as a hole and dropped, so a chord resting on both posts was glued to one only.
    Read back from STL, the joints are those of the CAD faces."""
    from bridgesim.cad.synthetic import upright
    from bridgesim.sim.idealize import find_patches

    u = upright("Node Block (9-ply UD)", [(0, 0), (210, 0), (210, 30), (185, 30),
                                          (185, 10), (25, 10), (25, 30), (0, 30)], 0, 18)
    chord = box("Chord (9-ply UD)", (-50, 30, 0), (260, 50, 18))

    def pats(bs):
        return sorted(round(inter.area, 1) for _a, _b, inter, *_ in find_patches(bs))

    write_stl([u, chord], tmp_path / "u.stl", names=True)
    back = load_cad(tmp_path / "u.stl")
    assert pats([u, chord]) == [450.0, 450.0]
    assert pats(back) == pats([u, chord])


def test_the_reading_follows_a_fall_inside_a_hold_extended_by_ringing():
    """The realism and adversarial reviews of round 25: a hold that ended and was
    extended again while the bridge rang was no longer read in time, so a window in
    travel (which stood still) reached into the collapse after it, and the reported load
    rose through a fall while the raw force halved. The held stretch now runs through a
    gap shorter than a period, and every major break cuts the reading."""
    from bridgesim.sim.crush import reported_load

    v, T = 100.0, 1 / 150.0
    t = np.arange(0, 0.08, 3e-5)
    tb = 0.040
    x = v * np.minimum(t, tb) + 0.02 * v * np.maximum(t - tb, 0)  # the jack slows at tb
    hold = (t >= tb) & ~((t > tb + 0.010) & (t < tb + 0.0105))  # a 0.5 ms gap
    tf = tb + 0.020  # the collapse, inside the extended hold
    true = np.where(t < tb, 500.0 * x, 500.0 * v * tb)
    true = np.where(t >= tf, true * np.clip(1 - (t - tf) / 0.002, 0.35, 1), true)
    raw = true + 5.0 * np.sin(2 * np.pi * t / T)
    rep = reported_load(t, raw, v, x, breaks=[tb], hold=hold, period_s=T)
    late = t > tf + 0.002 + T
    assert np.all(np.abs(rep[late] - true[late]) < 0.1 * true[late])
    before = (t > tb + 0.0105) & (t < tf - T)
    assert np.all(np.abs(rep[before] - true[before]) < 0.03 * true[before])


def test_the_strongest_joint_governs_only_at_the_node_where_the_fall_began():
    """The adversarial review of round 25: of the joints letting go within half a period,
    the strongest was named even on the other truss, 1.4 ms after the first: the label
    is now the strongest of those sharing a part with the first joint that let go."""
    from bridgesim.sim.crush import Event, attribute

    peak = 19000.0
    ev = [Event(0.13412, 5.5, 0.99 * peak, "joint_cracking", "Diag R Far | Stub Far",
                capacity_N=673.0),
          Event(0.13434, 5.5, 0.98 * peak, "joint_failure", "Diag R Far | plate_M Far Outer",
                capacity_N=3143.0),
          Event(0.13451, 5.5, 0.97 * peak, "joint_failure", "Diag R Far | plate_M Far Inner",
                capacity_N=2383.0),
          Event(0.13587, 5.5, 0.9 * peak, "joint_failure",
                "Diag L Near | plate_M Near Outer", capacity_N=3410.0)]
    gov, competing = attribute(ev, 0.1340, peak, 5.5, 100.0, period_s=0.007)
    assert gov.what == "Diag R Far | plate_M Far Outer"


def test_a_load_rising_past_f_u_after_the_failure_call_is_flagged():
    """The adversarial review of round 25: a 10 mm truss read 14.7 kgf at its first 30 %
    dip (a deck edge touching a vertical let go) and then carried 39.7, with no warning."""
    from bridgesim.sim import study

    m = build_model_simple()
    res = _stub_result(m)
    res.Fu_kgf, res.reload_kgf = 14.7, 39.7
    q = study._quality(res, MAT, SP)
    assert not q.trustworthy and any("rose again" in r for r in q.reasons)
    res.reload_kgf = 15.0
    assert not any("rose again" in r for r in study._quality(res, MAT, SP).reasons)


def build_model_simple():
    from bridgesim.sim.idealize import Overrides, build_model

    return build_model(simple_beam_bridge(), MAT, SP, Overrides(), seed=None)


def test_a_small_joint_on_a_plate_gets_its_own_grid_and_a_coarse_plate_is_flagged():
    """The literature and adversarial reviews of round 25: a joint on a plate's edge, or
    one narrower than the plate's elements, took the plate's nodes as its glue points (one
    row along the mid-surface: a hinge) or a 2x2 grid, depending on the mesh; a 10 mm truss
    read 14.7 kgf at quick against 48 at standard, unflagged. Such a joint is sampled on
    its own grid, three points a side, and a failing joint narrower than a third of its
    plate's elements is flagged at any member depth."""
    from bridgesim.sim import study
    from bridgesim.sim.idealize import Overrides, build_model

    edge = box("Post (3-ply UD)", (-200, 173.1, -3), (-190, 260, 3))  # on the deck's top
    side = box("Tab (2-ply 0-90)", (-200, 150, 75), (-190, 173.1, 79.1))  # on its edge
    for scale in (1.6, 0.5):
        m = build_model(simple_beam_bridge() + [edge, side], MAT, SP, Overrides(),
                        seed=None, mesh_scale=scale)
        names = [b.name for b in m.bodies]
        p = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Deck (2-ply 0-90)", "Tab (2-ply 0-90)"})
        assert len(p.springs) >= 9  # its own grid, 3 x 3 at least
        q = next(p for p in m.patches if {names[p.body_a], names[p.body_b]}
                 == {"Deck (2-ply 0-90)", "Post (3-ply UD)"})
        assert len(q.springs) >= 9 and q.mesh_ratio > 0
    m = build_model(simple_beam_bridge() + [edge], MAT, SP, Overrides(), seed=None,
                    mesh_scale=1.6)
    res = _stub_result(m, "joint failure: Deck (2-ply 0-90) | Post (3-ply UD)")
    pq = next(p for p in m.patches if "Post" in m.bodies[p.body_a].name
              + m.bodies[p.body_b].name)
    q = study._quality(res, MAT, SP)
    if pq.mesh_ratio > study.JOINT_MESH_RATIO:
        assert any("narrower than the elements" in r for r in q.reasons)
    pq.mesh_ratio = 10.0
    q = study._quality(res, MAT, SP)
    assert any("narrower than the elements" in r for r in q.reasons)
    assert q.min_fidelity == "fine"


# ------------------------------------------------------- compact blocks (round 28)


def _block_cells(m, name):
    """The node ids of a block's cells and its interior planes (patches of kind block)."""
    bi = next(i for i, b in enumerate(m.bodies) if b.name == name)
    return (np.asarray(m.bodies[bi].nodes),
            [k for k, p in enumerate(m.patches) if p.kind == "block" and p.body_a == bi])


def _drive_cell(m, moving, move, dmax, n=1500):
    """Glue-point forces (the NumPy reference law) on the cell ``moving`` while it is
    moved by ``move`` x s (no rotation) and every other cell held: (s, force along
    move)."""
    move = np.asarray(move, float)
    st = S.initial_state(m, 1e12)
    F, Mo = np.zeros((m.n_nodes, 3)), np.zeros((m.n_nodes, 3))
    out = []
    for s_ in np.linspace(0.0, dmax, n):
        st.x = m.X.copy()
        st.x[moving] += s_ * move
        F[:] = 0.0
        Mo[:] = 0.0
        S.glue_forces(m, st, F, Mo, update=True)
        out.append((s_, float(F[moving] @ move)))
    return np.array(out)


def test_block_cell_mm_zero_is_the_rigid_block_of_round_27():
    """block.cell_mm 0 keeps every compact block one rigid body, as until round 27: the
    model is the one kind: rigid gives each block (array for array), each block one node
    at its centroid with its mass and the mean of its principal moments, no plane inside
    it; and a crush of it is the same crush. (Checked bit for bit against round 27's code
    when round 28 was made: the model arrays of the synthetic bridges and a quick crush.)"""
    from bridgesim.sim.crush import CrushOptions, run_crush
    from bridgesim.sim.idealize import Overrides, build_model

    sp0 = SP.updated({"block.cell_mm": 0.0})
    m0 = build_model(simple_beam_bridge(), MAT, sp0, seed=0, mesh_scale=1.6)
    mr = build_model(simple_beam_bridge(), MAT, SP, Overrides.from_dict(
        {"bodies": {"Pier*": {"kind": "rigid"}}}), seed=0, mesh_scale=1.6)
    for k, v in vars(m0).items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(v, getattr(mr, k), err_msg=k)
    assert not any(p.kind == "block" for p in m0.patches)
    rho = MAT.density_kg_m3.value * 1e-12 * (1 + MAT.glue.mass_fraction.value)
    for b, cad in zip(m0.bodies, simple_beam_bridge(), strict=True):
        if b.kind == "block":
            assert len(b.nodes) == 1
            nd = b.nodes[0]
            assert m0.mass[nd] == pytest.approx(rho * cad.mesh.volume, rel=1e-12)
            assert m0.rot_inertia[nd] == pytest.approx(
                rho * float(np.mean(np.linalg.eigvalsh(cad.mesh.inertia))), rel=1e-12)
    opt = CrushOptions(speed_mm_s=300, max_travel_mm=3.0, auto_speed=False,
                       hold_on_failure=False)
    r0 = run_crush(build_model(simple_beam_bridge(), MAT, sp0, seed=0, mesh_scale=1.6), opt)
    rr = run_crush(mr, opt)
    assert r0.Fu_N == rr.Fu_N and r0.steps == rr.steps
    np.testing.assert_array_equal(r0.curve["force"], rr.curve["force"])


def test_a_block_pressed_between_two_faces_has_the_compliance_of_its_depth():
    """A block 25 x 25 mm in section, L = 25, 40 or 60 mm long between two thin rigid pads
    (1, 2 or 3 cells along the load), its glue lines made stiff: the springs in series from
    pad to pad (each face's, each plane's: by symmetry it is pressed evenly) give the
    wood's L / (E A), not twice it (round 27 counted the block's whole depth under each of
    its two faces), with E = E_T across the grain (no grain known) or E_L along it (grain
    y), within 10 % (it is exact for a box); sheared between the pads L / (G A), G the
    block's (both ways in the plane: 1 / G = 0.5 / G_LT + 0.5 / G_rolling with its fibres
    across the load, G_LT with them along it). The pads' own wood (2 mm of end grain,
    E_L; t / 3 G_LT in shear) and the glue lines are taken out by hand."""
    from bridgesim.sim.idealize import Overrides, build_model
    from bridgesim.sim.materials import Ply

    sp = SP.updated({"glue.K_n_N_per_mm3": 1e7, "glue.K_s_N_per_mm3": 1e7})
    ply = Ply.from_material(MAT, sp)
    A = 25.0 * 25.0
    G_across = 1.0 / (0.5 / ply.G_LT + 0.5 / sp["wood.G_rolling_MPa"])
    for grain, E, G in ((None, ply.E_T, G_across), ("y", ply.E_L, ply.G_LT)):
        for L, cells in ((25.0, 1), (40.0, 2), (60.0, 3)):
            bodies = [box("Pad Bottom", (-15, 0, -15), (15, 2, 15)),
                      box("Seat (block)", (-12.5, 2, -12.5), (12.5, 2 + L, 12.5)),
                      box("Pad Top", (-15, 2 + L, -15), (15, 4 + L, 15))]
            ov = {"bodies": {"Pad*": {"kind": "rigid", "grain": "y"}}}
            if grain:
                ov["bodies"]["Seat*"] = {"grain": grain}
            m = build_model(bodies, MAT, sp, ov=Overrides.from_dict(ov), seed=None)
            ids, planes = _block_cells(m, "Seat (block)")
            assert len(ids) == cells and len(planes) == cells - 1
            Cn = Cs = 0.0
            for k, p in enumerate(m.patches):
                sel = m.c_patch == k
                assert abs(p.normal[1]) > 0.99  # the two faces and the planes between
                Cn += 1.0 / float((m.c_Kn[sel] * m.c_area[sel]).sum())
                Cs += 1.0 / float((m.c_Ks[sel] * m.c_area[sel]).sum())
            Cn -= 2 * (1.0 / sp["glue.K_n_N_per_mm3"] + 2.0 / ply.E_L) / A
            Cs -= 2 * (1.0 / sp["glue.K_s_N_per_mm3"] + 2.0 / (3 * ply.G_LT)) / A
            assert 1.0 / Cn == pytest.approx(E * A / L, rel=0.1), (grain, L)
            assert 1.0 / Cs == pytest.approx(G * A / L, rel=0.1), (grain, L)


def test_a_block_in_double_shear_splits_at_its_planes():
    """A block 60 x 25 x 25 mm (three 20 mm cells along x) whose middle cell is pushed
    across while its end cells are held: two planes of A = 625 mm^2 in shear, stiffness
    2 G A / h (h = 20 mm between the cells' centroids) and strength 2 tau A with tau the
    wood's average on the plane (f_v crossing the fibres: grain x; the mean of f_v and the
    rolling shear with the fibres in the plane: no grain known), not f_v / 1.5, which is
    for a notched beam's section. Then in a crush: the block on two rigid piers under its
    end cells, a rigid loader on its middle cell, its strength factor 0.25 (so it fails
    before its faces crush under the loader and the piers, at f_c90): it splits at its two
    planes (block_fracture at x = +-10 mm), which governs, at 2 x 0.25 f_v A within 10 %
    (the piers 5 mm from the planes: the bending there adds a tension of 1.2 tau, against
    f_t along the fibres: ~3 %)."""
    from bridgesim.sim import study
    from bridgesim.sim.crush import KGF, CrushOptions, run_crush
    from bridgesim.sim.idealize import Overrides, build_model
    from bridgesim.sim.materials import Ply

    sp = SP.updated({"wood.rate_exponent": 0.0})
    ply = Ply.from_material(MAT, sp)
    A = 625.0
    roll = min(0.719 * ply.f_v, sp["wood.f_rolling_MPa"])  # 90 degrees to the grain
    G_across = 1.0 / (0.5 / ply.G_LT + 0.5 / sp["wood.G_rolling_MPa"])
    for grain, tau, G in (("x", ply.f_v, ply.G_LT), (None, 0.5 * (ply.f_v + roll),
                                                     G_across)):
        ov = {"bodies": {"B*": {"grain": grain}}} if grain else {}
        m = build_model([box("B (block)", (-30, 0, -12.5), (30, 25, 12.5))], MAT, sp,
                        ov=Overrides.from_dict(ov), seed=None)
        ids, planes = _block_cells(m, "B (block)")
        assert len(ids) == 3 and len(planes) == 2
        mid = ids[int(np.argmin(np.abs(m.X[ids, 0])))]
        r = _drive_cell(m, mid, (0.0, 1.0, 0.0), dmax=3.0 * tau / G * 20.0)
        k0 = abs(r[3, 1]) / r[3, 0]
        assert k0 == pytest.approx(2 * G * A / 20.0, rel=0.1), grain
        assert np.abs(r[:, 1]).max() == pytest.approx(2 * tau * A, rel=0.1), grain
    sf = 0.25
    bodies = [box("Pier L", (-20, 0, -12.5), (-10, 40, 12.5)),
              box("Pier R", (10, 0, -12.5), (20, 40, 12.5)),
              box("Seat (block)", (-30, 40, -12.5), (30, 65, 12.5)),
              box("Loader", (-10, 65, -12.5), (10, 85, 12.5))]
    ov = Overrides.from_dict({"bodies": {
        "Pier*": {"kind": "rigid", "grain": "y", "glue_factor": 5},
        "Loader": {"kind": "rigid", "grain": "y", "glue_factor": 5},
        "Seat*": {"grain": "x", "strength_factor": sf}}})
    m = build_model(bodies, MAT, sp, ov=ov, seed=None)
    res = run_crush(m, CrushOptions(speed_mm_s=50.0, max_travel_mm=8.0,
                                    after_failure_mm=1.0))
    assert res.governing == "block fracture: Seat (block)"
    assert res.Fu_kgf == pytest.approx(2 * sf * ply.f_v * A / KGF, rel=0.1)
    split = [e for e in res.events if e.kind == "block_fracture"]
    assert {round(float(e.where[0])) for e in split} == {-10, 10}
    assert all(abs(e.where[1] - 52.5) < 1e-6 and e.what == "Seat (block)" for e in split)
    assert "across its grain" in split[0].detail
    seq = study._sequence(res.events)
    assert any("block split inside: Seat (block)" in s for s in seq)
    assert res.criteria["joint_or_shear"]["what"] == "Seat (block)"


def test_a_block_pulled_apart_breaks_at_its_tension_strength():
    """A block 40 x 25 x 25 mm (two 20 mm cells along x) pulled apart (one cell moved
    away from the other, no rotation): across its grain (no grain known, its fibres in the
    plane) it breaks at f_t90 x the size factor of the 625 mm^2 plane (t90_size_factor:
    (4800 / 1250)^(1/8) = 1.18) x A; along its grain (grain x) at f_t x A; its normal
    stiffness E / h x A (E_T, E_L; h = 20 mm)."""
    from bridgesim.sim.idealize import Overrides, build_model, t90_size_factor
    from bridgesim.sim.materials import Ply

    sp = SP.updated({"wood.rate_exponent": 0.0})
    ply = Ply.from_material(MAT, sp)
    A = 625.0
    vf = t90_size_factor(A, sp["wood.t90_weibull_m"])
    assert vf == pytest.approx((4800.0 / 1250.0) ** 0.125, rel=1e-9)
    for grain, f, E in ((None, ply.f_t90 * vf, ply.E_T), ("x", ply.f_t, ply.E_L)):
        ov = {"bodies": {"B*": {"grain": grain}}} if grain else {}
        m = build_model([box("B (block)", (-20, 0, -12.5), (20, 25, 12.5))], MAT, sp,
                        ov=Overrides.from_dict(ov), seed=None)
        ids, planes = _block_cells(m, "B (block)")
        assert len(ids) == 2 and len(planes) == 1
        p = m.patches[planes[0]]
        assert p.split == ("across" if grain else "along")
        mover = ids[int(np.argmax(m.X[ids, 0]))]  # the cell at +x, pulled along +x
        r = _drive_cell(m, mover, (1.0, 0.0, 0.0), dmax=4.0 * f / E * 20.0)
        assert abs(r[3, 1]) / r[3, 0] == pytest.approx(E / 20.0 * A, rel=0.02), grain
        assert np.abs(r[:, 1]).max() == pytest.approx(f * A, rel=0.1), grain


def test_a_divided_block_keeps_its_mass_and_inertia():
    """The cells of a block have the solid's mass, centre of gravity and mean principal
    moment of inertia (the one rigid body's): exactly for a box, cell by cell (each a box
    of its own: m (a^2 + b^2 + c^2) / 18 about its centroid); for an L-shaped block, whose
    sampled cells are then scaled to the solid, in total, the lattice cells outside it
    left empty and its cells where the solid is; for a wedge (no lattice cell is whole)
    in total, no cell much lighter than a quarter of a full one."""
    from bridgesim.cad.synthetic import prism
    from bridgesim.sim.idealize import Overrides, build_model

    rho = MAT.density_kg_m3.value * 1e-12 * (1 + MAT.glue.mass_fraction.value)
    shapes = {"Box (block)": box("Box (block)", (0, 0, 0), (60, 40, 25)),
              "L (block)": prism("L (block)", [(0, 0), (60, 0), (60, 20), (20, 20), (20, 60),
                                               (0, 60)], 0.0, 20.0),
              "Wedge (block)": prism("Wedge (block)", [(0, 0), (60, 0), (0, 45)], 0.0, 30.0)}
    for name, cad in shapes.items():
        m = build_model([cad], MAT, SP, Overrides(), seed=None)
        ids = np.asarray(m.bodies[0].nodes)
        M = rho * cad.mesh.volume
        J = rho * float(np.mean(np.linalg.eigvalsh(cad.mesh.inertia)))
        mc = m.mass[ids]
        assert mc.sum() == pytest.approx(M, rel=1e-9), name
        c = (mc[:, None] * m.X[ids]).sum(0) / mc.sum()
        np.testing.assert_allclose(c, cad.mesh.centroid, atol=1e-6)
        r2 = ((m.X[ids] - c) ** 2).sum(1)
        assert m.rot_inertia[ids].sum() + 2.0 / 3.0 * (mc * r2).sum() == pytest.approx(
            J, rel=1e-9), name
        if name == "Box (block)":  # 3 x 2 x 1 cells of 20 x 20 x 25 mm
            assert len(ids) == 6
            assert np.allclose(mc, M / 6)
            assert np.allclose(m.rot_inertia[ids], M / 6 * (20**2 + 20**2 + 25**2) / 18)
        if name == "L (block)":  # 5 of its 3 x 3 lattice cells are wood, 20 mm each
            assert len(ids) == 5
            assert np.allclose(mc, M / 5)
            cells = {(round(x), round(z)) for x, _y, z in m.X[ids]}
            assert cells == {(10, 10), (30, 10), (50, 10), (10, 30), (10, 50)}
        if name == "Wedge (block)":
            assert len(ids) >= 3 and (mc > 0.2 * M / 9).all()


def test_planes_inside_blocks_are_not_joints_and_the_question_says_cells(tmp_path):
    """The planes between a block's cells are no joints of the bridge: inspect lists the
    same joints, glued area and joint count with or without cells (block.cell_mm 0), they
    count in the model's size as block planes, and the rigid_blocks question (same id)
    says the blocks are cells of their wood joined by its stiffness and strength (with
    cells) or rigid (without); kind: rigid answers it."""
    from bridgesim.sim import study

    cad = tmp_path / "blocks.stl"
    write_stl(simple_beam_bridge(), cad, names=True)
    rep = study.inspect(cad)
    rep0 = study.inspect(cad, overrides={"params": {"block.cell_mm": 0}})
    rep_r = study.inspect(cad, overrides={"bodies": {"Pier*": {"kind": "rigid"}}})
    assert [(j.a, j.b, j.area_mm2) for j in rep.joints] == \
        [(j.a, j.b, j.area_mm2) for j in rep0.joints]
    assert all(j.a != j.b for j in rep.joints)
    assert rep.model["glued_patches"] == rep0.model["glued_patches"] == len(rep.joints)
    assert rep.model["block_cells"] == 54 and rep.model["block_planes"] == 108
    assert rep0.model["block_cells"] == 0 and rep0.model["block_planes"] == 0
    q = next(q for q in rep.questions if q.id == "rigid_blocks")
    q0 = next(q for q in rep0.questions if q.id == "rigid_blocks")
    assert "cells" in q.question and "stiffness and strength" in q.question
    assert "rigid blocks" in q0.question and "cells" not in q0.question
    assert not any(q.id == "rigid_blocks" for q in rep_r.questions)
    assert any("27 cell(s)" in b.why for b in rep.bodies if b.kind == "block")


def test_the_synthetic_bridges_stand_on_blocks_of_cells(tmp_path):
    """The synthetic bridges stand on piers and support blocks of cells. Their stiffness
    under the plate (screen) moves little: the piers' own compliance under a stringer
    (each carries P/2; 150 mm of E_T over a 50 x 150 mm section, ~4e-6 mm/N) is ~0.1 %
    of the stringers' bending (~0.09 mm/N); one cell per pier (block.cell_mm 1000: the
    face's depth is half the pier, as round 27 took it under a face not glued behind)
    reads round 27 within 2 %; 20 mm cells read +2.7 % (1.11 -> 1.14 kgf/mm, round 28):
    the stringers' ends sit on cells that spread their load over 50 mm at once and
    restrain their turning more than round 27's 75 mm of wood under each face did; the
    truss on its support blocks within 1 %. The beam bridge settles under its own weight
    on the blocks' cells as on the rigid blocks."""
    from bridgesim.cad.synthetic import layered_truss
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions, run_crush
    from bridgesim.sim.idealize import build_model

    k = {}
    for nm, bodies in (("beam", simple_beam_bridge()), ("truss", layered_truss())):
        cad = tmp_path / f"{nm}.stl"
        write_stl(bodies, cad, names=True)
        for cm in (0.0, 1000.0, 20.0):
            k[nm, cm] = study.screen(cad, overrides={"params": {"block.cell_mm": cm}},
                                     load_kgf=10.0).stiffness_kgf_per_mm
    assert k["beam", 1000.0] == pytest.approx(k["beam", 0.0], rel=0.02)
    assert k["beam", 1000.0] <= k["beam", 20.0] <= 1.05 * k["beam", 0.0]
    assert k["truss", 20.0] == pytest.approx(k["truss", 0.0], rel=0.01)
    m = build_model(simple_beam_bridge(), MAT, SP, seed=0, mesh_scale=1.6)
    res = run_crush(m, CrushOptions(speed_mm_s=300, max_travel_mm=1.0, auto_speed=False))
    assert res.settle_sag_mm < 2.0 and res.criteria["instability"] is None
    assert res.added_mass_frac < 0.05
