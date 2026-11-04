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
    gives the material's tau_g (what the bridge model assumes for every joint)."""
    f = SP["glue.tau_peak_over_avg"]
    r = lab.lap_shear(2.0 * f)
    assert r.peak / 200 == pytest.approx(2.0, rel=0.05)


def _glued_pair():
    cap = {}

    def grab(m, *a, **k):
        cap["m"] = m
        raise StopIteration

    orig = lab.drive
    lab.drive = grab
    try:
        lab.lap_shear(2.0)
    except StopIteration:
        pass
    finally:
        lab.drive = orig
    m = cap["m"]
    m.mass = np.maximum(m.mass, 1e-12)
    m.rot_inertia = np.maximum(m.rot_inertia, 1e-12)
    return m


def test_energy_is_conserved_without_damping():
    from bridgesim.sim.crush import stable_dt

    m = _glued_pair()
    m.c_tau[:] = 1e6
    m.c_sig[:] = 1e6
    s = S.Settings(dt=1e-7, beta_frac=0.0)
    dt, m.mass, m.rot_inertia, _ = stable_dt(m, s, 5e-6, 0.0)
    s.dt = dt
    st = S.initial_state(m, 1e12)
    rng = np.random.default_rng(0)
    st.v = rng.normal(0, 1, st.v.shape)
    F = np.zeros((m.n_nodes, 3))
    Mo = np.zeros_like(F)
    S.internal_and_contact(m, st, s, F, Mo)
    e0 = None
    for k in range(4000):
        S.step(m, st, s, F, Mo, grav_scale=0, alpha=0)
        if k == 10:
            e0 = S.kinetic_energy(m, st)
    assert np.isfinite(st.x).all()
    assert S.kinetic_energy(m, st) < 3 * e0  # bounded (energy exchanges with strain)


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
            "compare_with_test", "lab_tests", "calibrate_material", "get_schema"} <= names


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
    ipk = int(np.argmax(c.load_kgf))
    rows = ["test,bridge_file,load_kgf,midspan_deflection_mm,event,notes"]
    idx = sorted(set(range(0, len(c.load_kgf), max(1, len(c.load_kgf) // 60))) | {ipk})
    for i in idx:  # the whole record, the drop after the peak included
        ev = "failure" if i == ipk else ""
        rows.append(f"T1,b.stl,{c.load_kgf[i]:.3f},{c.deflection_mm[i]:.3f},{ev},")
    (tmp_path / "t.csv").write_text("\n".join(rows))
    gap = study.compare(Path(rep.artifacts["report_json"]), tmp_path / "t.csv")
    k = {m.name: m for m in gap.metrics}["initial stiffness"]
    assert k.simulated > 0 and k.measured > 0
    assert k.measured == pytest.approx(k.simulated, rel=0.25)
    assert gap.score_0_100 > 60
