"""Published craft-stick coupon tests simulated with the shipped material: the gap.

The coupon data are in validation/literature/ (its README says where each number comes
from): ENS Paris-Saclay's 76 stick tension tests, single-stick and laminated 3-point
bending (edu_14235 Table 1), the one lap whose glue line failed (rouge 2019), and
single-stick buckling (bleu 2019 Table 4, rouge 2019). This script measures how far the
model's coupons are from them with the material values as shipped, or with an overrides
file's ``params:`` (``-o``). Nothing is tuned.

(a) study.lab on the ENS coupon folder (stick bending and glue laps, the coupons lab
    simulates) and, beside it, a stick pulled in tension (the bridge's fibre beam between
    rigid grips); then study.calibrate on the folder: what the fitted values would be
    (printed and saved in the summary; nothing is written to materials/);
(b) the 3-point bending series of edu_14235 (1, 2 and 3 glued sticks flat, 2 and 3 on
    edge) crushed through the full pipeline (study.crush, as a bridge is): each coupon a
    CAD model, the beam a body named "Beam (N-ply UD)" resting dry on a rigid fixture of
    two 6 mm posts 100 mm apart, loaded at mid-span through a rigid nose by the press
    plate; each model is checked with study.inspect before it is crushed; seed -1 (the
    perfect coupon) and seeds 0-4; a run flagged only as not quasi-static is crushed again
    at half the jack speed, as montecarlo does;
(c) the single-stick buckling tests with lab.euler_column (elastic, pin-ended, bow L/1000)
    and the same column with the bridge's fibre law (f_t, f_c); a clamped column as a
    pin-ended one of half its free length.

Unglued stacks need no simulation (n sticks carry n times one stick): tabulated.

    uv run python validation/literature_coupons.py --out DIR [-o overrides.yaml]
        [--seeds=-1,0,1,2,3,4] [--fidelity fine] [--jobs 2] [--skip-crush]

writes DIR/summary.json and, per coupon, DIR/<coupon>/coupon.stl, coupon.overrides.yaml,
inspect.json and seed_<s>/crush.json (curve.csv, events.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENS = HERE / "literature" / "ens_paris_saclay"
OTHER = ENS / "other"

# the sticks: the 2018-19 Move Your Bridge mean of 50 sticks (jaune Table 1), the only
# dimensions any of these sources gives (literature/README.md)
STICK_T = 2.02  # mm
STICK_B = 9.6  # mm
STICK_L = 114.0  # mm
SPAN = 100.0  # mm between the supports (edu_14235, stated)
# the bench's supports and nose are steel rollers whose diameter is not stated (about 20
# mm in the photos): taken as flat faces this long along the span (assumed)
CONTACT_MM = 6.0
# a soft coupon at fine fidelity (dt ~0.4 us) needs more steps than a bridge's run limit
# (CrushOptions.max_steps 400 000): the run is allowed to go on, nothing else changes
MAX_STEPS = 4_000_000

# the coupons crushed: (key, sticks, glued in the source table, orientation)
CONFIGS = [("1F", 1, "n/a", "flat"), ("2GF", 2, "yes", "flat"), ("3GF", 3, "yes", "flat"),
           ("2GE", 2, "yes", "edge"), ("3GE", 3, "yes", "edge")]
# unglued stacks against n times one stick
UNGLUED = [("2UF", 2, "flat"), ("3UF", 3, "flat"), ("2UE", 2, "edge"), ("3UE", 3, "edge")]


# ------------------------------------------------------------------------- inputs


def material_and_params(overrides: str | None):
    """The material and simulation values a crush with these overrides would use, built
    as study._inputs builds them: the params: that are not material.* on the simulation
    values (measured when listed under measured:), then material.<key> on the material,
    G following an E given alone (E/G kept). The material is returned before the crush's
    stick modulus is put in (study.lab and study.calibrate do that themselves); the
    third value is the overrides as read."""
    from bridgesim.materials import load_material
    from bridgesim.sim import study
    from bridgesim.sim.materials import SimParams

    ov, raw = study.load_overrides(overrides)
    mat, sp = load_material("popsicle_birch"), SimParams.load()
    meas = set(ov.measured)
    sim_over = {k: v for k, v in ov.params.items() if not k.startswith("material.")}
    for src, part in (("measured", {k: v for k, v in sim_over.items() if k in meas}),
                      ("assumed", {k: v for k, v in sim_over.items() if k not in meas})):
        if part:
            sp = sp.updated(part, source=src)
    mat_over = {k.split(".", 1)[1]: v for k, v in ov.params.items()
                if k.startswith("material.")}
    if mat_over:
        bad = [k for k in mat_over if k not in mat.props()]
        if bad:
            raise ValueError(f"unknown material parameter(s) {bad}")
        g_new = None
        if "E_MPa" in mat_over and "G_MPa" not in mat_over:
            g_new = mat.G_MPa.value * float(mat_over["E_MPa"]) / mat.E_MPa.value
        for src, part in (("measured", {k: v for k, v in mat_over.items()
                                        if f"material.{k}" in meas}),
                          ("assumed", {k: v for k, v in mat_over.items()
                                       if f"material.{k}" not in meas})):
            if part:
                mat = study._material_with(mat, part, source=src)
        if g_new is not None:
            mat = study._material_with(mat, {"G_MPa": g_new}, source="assumed")
    return mat, sp, raw


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def measured_bending() -> dict[str, list[float]]:
    """edu_14235 Table 1 by configuration key (other/laminated_bending.csv, the rows in
    Table 1)."""
    out: dict[str, list[float]] = {}
    for r in read_csv(OTHER / "laminated_bending.csv"):
        if r["in_table_1"] != "yes":
            continue
        key = f"{r['sticks']}{'' if r['glued'] == 'n/a' else 'G' if r['glued'] == 'yes' else 'U'}" \
              f"{'F' if r['orientation'] == 'flat' else 'E'}"
        out.setdefault(key, []).append(float(r["peak_load_N"]))
    return out


def observed_failure(key: str) -> str:
    sticks, rest = key[0], key[1:]
    glued = {"GF": "yes", "GE": "yes", "UF": "no", "UE": "no", "F": "n/a", "E": "n/a"}[rest]
    orient = "flat" if rest.endswith("F") else "edge"
    for r in read_csv(OTHER / "laminated_bending.csv"):
        if (r["sticks"], r["glued"], r["orientation"]) == (sticks, glued, orient):
            return r["failure_observed"]
    return ""


def stats(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1), "min": min(xs),
            "max": max(xs), "values": xs}


# --------------------------------------------------------------- (a) lab, calibrate


def tension_coupon(mat, sp, b: float, t: float, L: float = 60.0, n_el: int = 12):
    """A stick pulled along its length between rigid grips (the far grip moved), made of
    the bridge's fibre beam with the material's f_t, f_c and the wood's fracture energy
    (lab.BeamSpec): its peak is f_t b t, the coupon model having no size effect."""
    from bridgesim.sim import lab as labmod
    from bridgesim.sim import study

    E, ft, fc = mat.E_MPa.value, mat.f_t_MPa.value, mat.f_c_MPa.value
    X = np.c_[np.linspace(0, L, n_el + 1), np.zeros(n_el + 1), np.zeros(n_el + 1)]
    m = labmod.blank_model(X)
    wl = study._wood_law(sp)
    m.params.update(fc_proportional=wl["fc_prop"], crush_softening=wl["crush_softening"],
                    crush_band_mm=wl["crush_band_mm"])
    labmod.add_beam_line(m, list(range(n_el + 1)),
                         labmod.BeamSpec(b=b, d=t, E=E, ft=ft, fc=fc,
                                         Gf=sp["wood.G_f_N_per_mm"]), up=(0, 1, 0))
    return labmod.drive(m, {0: "xyzr"}, n_el, 0, rate=10.0, max_disp=3.0 * ft / E * L,
                        stop_drop=0.3)


def part_a(mat, sp) -> dict:
    from bridgesim.sim import study

    cm = study._crush_material(mat, sp)
    out: dict = {"material_used": {
        "E_MPa": cm.E_MPa.value, "f_t_MPa": cm.f_t_MPa.value, "f_c_MPa": cm.f_c_MPa.value,
        "f_b_MPa": cm.f_b_MPa.value, "glue.tau_g_MPa": cm.glue.tau_g_MPa.value,
        "note": "E is the crush's (wood.E_placeholder_MPa) while the material file holds "
                "its shipped placeholder"}}
    files, warn = study.coupon_files(ENS)
    out["coupon_files"] = {k: [f.name for f in v] for k, v in files.items() if v}
    out["coupon_file_warnings"] = warn
    rep = study.lab(material=mat, sim_params=sp, csv_dir=ENS)
    gaps = []
    for g in rep.gaps:
        sim = [float(m_.simulated) for m_ in g.metrics]
        meas = [float(m_.measured) for m_ in g.metrics]
        gaps.append({"subject": g.subject, "score_0_100": g.score_0_100,
                     "measured_N": stats(meas), "simulated_N": sorted(set(sim)),
                     "ratio_sim_over_mean": round(statistics.fmean(sim)
                                                  / statistics.fmean(meas), 3),
                     "metrics": [m_.model_dump() for m_ in g.metrics]})
    out["lab_gaps"] = gaps
    # tension: lab does not simulate it; the fibre-beam coupon of the same section
    rows = read_csv(ENS / "stick_tension.csv")
    F = [float(r["failure_load_N"]) for r in rows]
    b, t = float(rows[0]["width_mm"]), float(rows[0]["thickness_mm"])
    tc = tension_coupon(cm, sp, b, t)
    cov = round(statistics.stdev(F) / statistics.fmean(F), 3)
    out["tension"] = {"measured_N": stats(F) | {"cov": cov},
                      "measured_strength_MPa": round(statistics.fmean(F) / (b * t), 1),
                      "simulated_N": round(tc.peak, 1),
                      "ratio_sim_over_mean": round(tc.peak / statistics.fmean(F), 3),
                      "section_mm": [b, t]}
    # the lap's two readings (glue_laps.csv notes): 38 mm from the text, 50 mm from the
    # figure; lab simulates the file's (38 mm)
    lap = read_csv(ENS / "glue_laps.csv")[0]
    Fl, w = float(lap["failure_load_N"]), float(lap["width_mm"])
    tau_peak = cm.glue.tau_g_MPa.value * sp.values.get("glue.tau_peak_over_avg", 1.065)
    alt = {}
    for ov in (38.0, 50.0):
        r = study._lap(sp, tau_peak, ov, w, cm.E_MPa.value)
        alt[f"{ov:g}"] = {"measured_tau_MPa": round(Fl / (ov * w), 2),
                          "simulated_N": round(r.peak, 1),
                          "ratio_sim_over_measured": round(r.peak / Fl, 3)}
    out["lap_overlap_readings"] = alt
    # calibrate on the folder: what the fitted material would be (nothing written)
    new, log, params = study.calibrate(ENS, material=mat, sim_params=sp)
    keys = ["E_MPa", "G_MPa", "f_t_MPa", "f_c_MPa", "f_b_MPa"]
    fitted = {k: {"file": getattr(mat, k).value, "calibrated": round(getattr(new, k).value, 3),
                  "changed": getattr(new, k).value != getattr(mat, k).value,
                  "source": getattr(new, k).source} for k in keys}
    fitted["glue.tau_g_MPa"] = {"file": mat.glue.tau_g_MPa.value,
                                "calibrated": round(new.glue.tau_g_MPa.value, 3),
                                "changed": new.glue.tau_g_MPa.value != mat.glue.tau_g_MPa.value,
                                "source": new.glue.tau_g_MPa.source}
    out["calibrate"] = {"fitted": fitted, "params": {k: round(v, 4) for k, v in params.items()},
                        "log": log,
                        "note": "compared with the material file (popsicle_birch.yaml and "
                                "the overrides). No stiffness was measured, so E and G stay "
                                "the file's; calibrate simulates the bending sticks with the "
                                "crush's E (wood.E_placeholder_MPa) to fit f_t and f_c, then "
                                "replaces f_t by the tension tests' mean"}
    return out


# ------------------------------------------------------------------- (b) the crushes


def coupon_bodies(n: int, edge: bool):
    """The beam (n sticks glued face to face, flat or on edge), the fixture (two posts
    CONTACT_MM wide, SPAN apart, on one base: the bench's supports, bolted to its frame,
    so neither can tip or slide apart) and the loading nose (a stem CONTACT_MM wide on
    the beam at mid-span, a cap under the plate)."""
    from bridgesim.cad.synthetic import box, upright

    H, Z = (STICK_B, n * STICK_T) if edge else (n * STICK_T, STICK_B)
    a, s = CONTACT_MM / 2, SPAN / 2
    beam = box(f"Beam ({n}-ply UD)", (-STICK_L / 2, 0.0, -Z / 2), (STICK_L / 2, H, Z / 2))
    zf = max(Z, STICK_B) / 2 + 5.0
    fixture = upright("Fixture", [(-80, -45), (80, -45), (80, -40), (s + a, -40), (s + a, 0),
                                  (s - a, 0), (s - a, -40), (-s + a, -40), (-s + a, 0),
                                  (-s - a, 0), (-s - a, -40), (-80, -40)], -zf, zf)
    zn = Z / 2 + 2.0
    nose = upright("Loading nose", [(-a, H), (a, H), (a, H + 45), (15, H + 45), (15, H + 50),
                                    (-15, H + 50), (-15, H + 45), (-a, H + 45)], -zn, zn)
    return [beam, fixture, nose]


def coupon_overrides(user: dict) -> dict:
    """The coupon's overrides: the fixture and the nose steel (rigid), bearing on the beam
    dry; the press plate light (the bench drives its nose by displacement; a 2 g plate
    keeps a single stick's period under the crush's 10 ms reference); the bench's 0.1 mm/s
    (edu_14235) as the test's pace. Acknowledged as drawn: the orientation check (the
    fixture's base lies on the platform at mid-span, as a bridge lying on its side would;
    the coupon is drawn Y up), a single flat stick as a bar (a beam in 3-point bending), the
    bridge rules, the crusher and the placeholder questions. The user's params: (and
    measured:) on top."""
    ov = {"bodies": {"Fixture": {"kind": "rigid"}, "Loading nose": {"kind": "rigid"},
                     "Beam*": {"kind": "bar"}},
          "glue": {"Fixture | *": 0, "Loading nose | *": 0},
          "params": {"crusher.plate_mass_kg": 0.002, "crusher.test_rate_mm_per_min": 6.0},
          "acknowledge": ["geometry:orientation", "kind:Beam*", "material:measure", "rule:*",
                          "crusher:*"]}
    ov["params"].update(user.get("params") or {})
    if user.get("measured"):
        ov["measured"] = list(user["measured"])
    return ov


def check_model(rep, n: int) -> list[str]:
    """What is not as intended in the inspect report: the beam a bar with its layup read
    from its name, the fixture and the nose rigid and bearing only (no glue), the plate
    on the nose, no open question about the geometry."""
    bad = []
    by = {b.name: b for b in rep.bodies}
    beam = by.get(f"Beam ({n}-ply UD)")
    if beam is None or beam.kind != "bar" or beam.layup != f"{n}-ply UD" \
            or beam.layup_source != "name":
        bad.append(f"beam: {beam.kind if beam else 'missing'}, "
                   f"{beam.layup if beam else ''} ({beam.layup_source if beam else ''})")
    for nm in ("Fixture", "Loading nose"):
        if nm not in by or "rigid" not in (by[nm].why or ""):
            bad.append(f"{nm} is not a rigid body")
    for j in rep.joints:
        if {"Fixture", "Loading nose"} & {j.a, j.b} and j.shear_capacity_N != 0:
            bad.append(f"{j.a} | {j.b} is glued ({j.shear_capacity_N} N)")
    if rep.loading.get("deck_bodies") != ["Loading nose"]:
        bad.append(f"the plate rests on {rep.loading.get('deck_bodies')}")
    for q in rep.questions:
        if q.severity == "important" or q.id.split(":")[0] in (
                "floating", "loose", "near_miss", "overlap", "layup", "orientation",
                "plate_fit", "geometry", "kind"):
            bad.append(f"open question {q.id}: {q.question[:120]}")
    return bad


def _crush_one(task: tuple) -> dict:
    """One coupon crush (run in a worker process)."""
    stl, ovf, seed, fidelity, out_dir, slow, threads = task
    import dataclasses

    import numba

    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    from bridgesim.sim import study
    from bridgesim.sim.crush import FIDELITY, CrushOptions, dt_target_for

    scale, speed = FIDELITY[fidelity]
    # study._screening's options (stop 2 mm after the failure call), more steps allowed
    opt = CrushOptions(speed_mm_s=speed, dt_target=dt_target_for(scale), after_failure_mm=2.0,
                       slow_bridge_speed_mm_s=min(speed, FIDELITY["standard"][1]),
                       max_steps=MAX_STEPS)
    if slow:
        opt = dataclasses.replace(opt, speed_mm_s=opt.speed_mm_s / 2,
                                  slow_bridge_speed_mm_s=opt.slow_bridge_speed_mm_s / 2)
    t0 = time.time()
    rep, _ = study.crush(stl, overrides=str(ovf), seed=None if seed < 0 else seed,
                         fidelity=fidelity, options=opt, out_dir=out_dir, replay=False,
                         stop_at_failure=True)
    return {"seed": seed, "slowed": slow, "wall_s": round(time.time() - t0, 1),
            "Fu_N": rep.Fu_N, "Fu_kgf": rep.Fu_kgf, "failure_mode": rep.failure_mode,
            "governing": rep.governing, "competing": rep.competing,
            "deflection_at_Fu_mm": rep.deflection_at_Fu_mm,
            "failures": [{"kind": f.kind, "what": f.what, "load_N": round(f.load_kgf * 9.80665, 1),
                          "jack_travel_mm": f.jack_travel_mm, "where_mm": f.where_mm}
                         for f in rep.failures[:6]],
            "trustworthy": rep.quality.trustworthy, "reasons": rep.quality.reasons,
            "speed_mm_s": round(rep.quality.speed_mm_s, 1),
            "period_ms": rep.quality.fundamental_period_ms,
            "kinetic_to_work_max": rep.quality.kinetic_to_work_max,
            "steps": rep.quality.steps, "code_sha": rep.code_sha}


def _flag(r: dict) -> str:
    return "" if r["trustworthy"] else " UNTRUSTWORTHY: " + "; ".join(r["reasons"])


def _not_quasi_static_only(r: dict) -> bool:
    return (not r["trustworthy"] and bool(r["reasons"])
            and all(x.startswith("loading not quasi-static") for x in r["reasons"]))


def part_b(out: Path, user: dict, seeds: list[int], fidelity: str, jobs: int,
           threads: int) -> dict:
    import concurrent.futures as cf
    import multiprocessing as mp

    import yaml

    from bridgesim.cad.synthetic import write_stl
    from bridgesim.sim import study

    meas = measured_bending()
    res: dict = {}
    tasks = []
    for key, n, _g, orient in CONFIGS:
        d = out / key
        d.mkdir(parents=True, exist_ok=True)
        stl, ovf = d / "coupon.stl", d / "coupon.overrides.yaml"
        write_stl(coupon_bodies(n, orient == "edge"), stl, names=True)
        ovf.write_text(yaml.safe_dump(coupon_overrides(user), sort_keys=False),
                       encoding="utf-8")
        ins = study.inspect(stl, overrides=str(ovf), seed=None)
        (d / "inspect.json").write_text(ins.model_dump_json(indent=1), encoding="utf-8")
        problems = check_model(ins, n)
        beam = next(b for b in ins.bodies if b.name.startswith("Beam"))
        res[key] = {"sticks": n, "orientation": orient, "measured_N": stats(meas[key]),
                    "observed_failure": observed_failure(key),
                    "model": {"beam": f"{beam.kind}, {beam.layup} (from its {beam.layup_source})",
                              "section_mm": beam.section_mm, "problems": problems,
                              "joints": [f"{j.a} | {j.b}: {j.kind}, {j.shear_capacity_N} N"
                                         for j in ins.joints]},
                    "runs": []}
        print(f"inspect {key}: {res[key]['model']['beam']}, section {beam.section_mm} mm; "
              + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)), flush=True)
        for s in seeds:
            tasks.append((key, (str(stl), str(ovf), s, fidelity, str(d / f"seed_{s}"), False,
                                threads)))
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futs = {ex.submit(_crush_one, t): k for k, t in tasks}
        for fu in cf.as_completed(futs):
            r = fu.result()
            k = futs[fu]
            res[k]["runs"].append(r)
            print(f"  {k} seed {r['seed']}: F_u {r['Fu_N']:.1f} N, {r['governing']}, "
                  f"{r['wall_s']:.0f} s{_flag(r)}", flush=True)
        # a run flagged only as not quasi-static is crushed again at half the jack speed
        again = [(k, (t[0], t[1], t[2], t[3], t[4] + "_slow", True, t[6])) for k, t in tasks
                 if any(r["seed"] == t[2] and _not_quasi_static_only(r) for r in res[k]["runs"])]
        futs = {ex.submit(_crush_one, t): k for k, t in again}
        for fu in cf.as_completed(futs):
            r = fu.result()
            k = futs[fu]
            res[k]["runs"] = [({**r, "slowed_from_N": x["Fu_N"]} if x["seed"] == r["seed"] else x)
                              for x in res[k]["runs"]]
            print(f"  {k} seed {r['seed']} at half speed: F_u {r['Fu_N']:.1f} N, "
                  f"{r['governing']}, {r['wall_s']:.0f} s{_flag(r)}", flush=True)
    for key, *_ in CONFIGS:
        runs = sorted(res[key]["runs"], key=lambda r: r["seed"])
        res[key]["runs"] = runs
        m = res[key]["measured_N"]["mean"]
        perfect = next((r for r in runs if r["seed"] < 0), None)
        sc = [r["Fu_N"] for r in runs if r["seed"] >= 0]
        res[key]["sim_perfect_N"] = perfect["Fu_N"] if perfect else None
        res[key]["ratio_perfect"] = round(perfect["Fu_N"] / m, 3) if perfect else None
        if sc:
            res[key]["sim_seeds_N"] = {"median": round(float(np.median(sc)), 1),
                                       "min": min(sc), "max": max(sc), "n": len(sc)}
            res[key]["ratio_median"] = round(float(np.median(sc)) / m, 3)
        gov: dict[str, int] = {}
        for r in runs:
            gov[r["governing"]] = gov.get(r["governing"], 0) + 1
        res[key]["governing_counts"] = gov
        res[key]["untrustworthy"] = [r["seed"] for r in runs if not r["trustworthy"]]
    return res


# ---------------------------------------------------------------- (c) buckling


def fibre_column(mat, sp, L: float, b: float, t: float, n_el: int = 16):
    """lab.euler_column with the bridge's fibre law: pin-ended, a half-sine bow of L/1000,
    the material's f_t and f_c (and the wood's compression law), shortened at one end."""
    from bridgesim.sim import lab as labmod
    from bridgesim.sim import study

    bow = L / 1000
    s = np.linspace(0, L, n_el + 1)
    X = np.c_[s, bow * np.sin(np.pi * s / L), np.zeros_like(s)]
    m = labmod.blank_model(X)
    wl = study._wood_law(sp)
    m.params.update(fc_proportional=wl["fc_prop"], crush_softening=wl["crush_softening"],
                    crush_band_mm=wl["crush_band_mm"])
    labmod.add_beam_line(m, list(range(n_el + 1)),
                         labmod.BeamSpec(b=b, d=t, E=mat.E_MPa.value, ft=mat.f_t_MPa.value,
                                         fc=mat.f_c_MPa.value, Gf=sp["wood.G_f_N_per_mm"]),
                         up=(0, 1, 0))
    return labmod.drive(m, {0: "xyz", n_el: "yz"}, n_el, 0, 5.0, max_disp=L * 0.02,
                        sign=-1.0, stop_drop=0.0)


def part_c(mat, sp) -> list[dict]:
    from bridgesim.sim import lab as labmod
    from bridgesim.sim import study

    cm = study._crush_material(mat, sp)
    E, b, t = cm.E_MPa.value, STICK_B, STICK_T
    I = b * t**3 / 12
    rows = {r["specimen"]: r for r in read_csv(OTHER / "buckling.csv")}
    # (row, pin-ended length simulated, the reading it stands for)
    cases = [("B-S1", 46.0, "listed length = effective length (as the team's own Euler "
                            "column uses it)"),
             ("B-S1", 23.0, "listed length = free length between the clamps: effective "
                            "length half of it"),
             ("B-S2", 47.0, "listed length = effective length"),
             ("B-S2", 23.5, "listed length = free length between the clamps"),
             ("B-S3", 109.0, "pin-ended over the listed 109 mm"),
             ("R-mean", STICK_L, "pin-ended; length not stated: the whole 114 mm stick "
                                 "(assumed)")]
    out = []
    for sid, L, reading in cases:
        r = rows[sid]
        meas = float(r["peak_load_N"])
        el = labmod.euler_column(E=E, L=L, b=b, t=t)
        fib = fibre_column(cm, sp, L, b, t)
        out.append({"specimen": sid, "team": r["team"],
                    "end_condition": r["end_condition_as_stated"],
                    "listed_length_mm": r["buckling_length_as_stated_mm"] or None,
                    "simulated_pin_ended_length_mm": L, "reading": reading,
                    "measured_N": meas,
                    "euler_N": round(math.pi**2 * E * I / L**2, 1),
                    "euler_column_N": round(el.peak, 1),
                    "fibre_column_N": round(fib.peak, 1),
                    "crushing_limit_N": round(cm.f_c_MPa.value * b * t, 1),
                    "ratio_fibre_over_measured": round(fib.peak / meas, 3)})
    return out


def unglued_table(meas: dict[str, list[float]]) -> list[dict]:
    out = []
    for key, n, orient in UNGLUED:
        one = statistics.fmean(meas["1F" if orient == "flat" else "1E"])
        m = statistics.fmean(meas[key])
        out.append({"coupon": key, "sticks": n, "orientation": orient,
                    "measured_N": stats(meas[key]), "n_times_one_stick_N": round(n * one, 1),
                    "ratio_measured_over_n_times_one": round(m / (n * one), 3)})
    return out


# ------------------------------------------------------------------------- main


def _fmt_range(s: dict) -> str:
    return f"{s['mean']:.0f} [{s['min']:.0f}-{s['max']:.0f}]"


def print_a(pa: dict) -> None:
    mu = pa["material_used"]
    print(f"material: E {mu['E_MPa']:g} MPa, f_t {mu['f_t_MPa']:g}, f_c {mu['f_c_MPa']:g}, "
          f"tau_g {mu['glue.tau_g_MPa']:g} MPa")
    print(f"coupon files read: {pa['coupon_files']}; warnings: {pa['coupon_file_warnings']}")
    print(f"{'coupon':34s} {'n':>3s} {'measured N':>18s} {'simulated N':>12s} {'sim/meas':>8s}")
    tn = pa["tension"]
    print(f"{'stick tension (fibre beam)':34s} {tn['measured_N']['n']:3d} "
          f"{_fmt_range(tn['measured_N']):>18s} {tn['simulated_N']:12.1f} "
          f"{tn['ratio_sim_over_mean']:8.3f}")
    for g in pa["lab_gaps"]:
        print(f"{g['subject'] + ' (lab)':34s} {g['measured_N']['n']:3d} "
              f"{_fmt_range(g['measured_N']):>18s} "
              f"{'/'.join(f'{x:.1f}' for x in g['simulated_N']):>12s} "
              f"{g['ratio_sim_over_mean']:8.3f}   score {g['score_0_100']}")
    for ov, v in pa["lap_overlap_readings"].items():
        print(f"  lap read as {ov} mm: measured {v['measured_tau_MPa']} MPa average, "
              f"simulated {v['simulated_N']} N, sim/meas {v['ratio_sim_over_measured']}")
    print("calibrate on the folder (what it would fit; nothing written):")
    for k, v in pa["calibrate"]["fitted"].items():
        print(f"  {k:16s} file {v['file']:<8g} -> {v['calibrated']:<9g} "
              f"({v['source'] if v['changed'] else 'unchanged: not measured'})")
    for k, v in pa["calibrate"]["params"].items():
        print(f"  {k} = {v}")
    for line in pa["calibrate"]["log"]:
        print(f"  log: {line}")


def print_unglued(rows: list[dict]) -> None:
    for u in rows:
        print(f"  {u['coupon']}: {_fmt_range(u['measured_N'])} N against "
              f"{u['n_times_one_stick_N']} N, ratio {u['ratio_measured_over_n_times_one']}")


def print_b(pb: dict) -> None:
    """F_u against the test, and what failed: the governing label and the first failure
    the model recorded (at or near the peak; for a coupon whose peak is under 50 N the
    crush calls the failure only at a criterion, so its label can name that criterion)."""
    print(f"{'coupon':6s} {'measured N':>16s} {'seed -1':>8s} {'seeds 0-4 N':>18s} "
          f"{'-1/meas':>7s} {'med/meas':>8s}")
    for key, v in pb.items():
        sd = v.get("sim_seeds_N")
        ss = f"{sd['median']:.0f} [{sd['min']:.0f}-{sd['max']:.0f}]" if sd else "-"
        print(f"{key:6s} {_fmt_range(v['measured_N']):>16s} "
              f"{(v['sim_perfect_N'] or float('nan')):8.1f} {ss:>18s} "
              f"{(v['ratio_perfect'] or float('nan')):7.3f} "
              f"{v.get('ratio_median', float('nan')):8.3f}")
        for r in v["runs"]:
            f0 = r["failures"][0] if r["failures"] else None
            first = (f"{f0['kind']} at {f0['load_N']:.0f} N, {f0['jack_travel_mm']:.1f} mm"
                     if f0 else "none")
            print(f"       seed {r['seed']:2d}: {r['Fu_N']:6.1f} N at "
                  f"{r['deflection_at_Fu_mm']:.1f} mm; governing '{r['governing']}'; "
                  f"first failure {first}"
                  + (f" (slowed from {r['slowed_from_N']} N)" if "slowed_from_N" in r else "")
                  + _flag(r))
        print(f"       observed: {v['observed_failure']}")


def print_c(pc: list[dict]) -> None:
    print(f"{'specimen':8s} {'L sim':>6s} {'measured':>8s} {'Euler':>7s} {'euler_col':>9s} "
          f"{'fibre':>7s} {'fibre/meas':>10s}  reading")
    for r in pc:
        print(f"{r['specimen']:8s} {r['simulated_pin_ended_length_mm']:6.1f} "
              f"{r['measured_N']:8.0f} {r['euler_N']:7.1f} {r['euler_column_N']:9.1f} "
              f"{r['fibre_column_N']:7.1f} {r['ratio_fibre_over_measured']:10.3f}  "
              f"{r['reading']}")


def print_summary(summary: dict) -> None:
    print("== (a) coupon files through study.lab"
          + (f", overrides {summary['overrides']}" if summary.get("overrides") else
             ", the shipped values"))
    print_a(summary["a"])
    print("\n== unglued stacks (no simulation): measured against n x one stick")
    print_unglued(summary["unglued"])
    if "b" in summary:
        print(f"\n== (b) 3-point bending crushed through the pipeline "
              f"({summary['fidelity']}; wall time per crush {summary['crush_wall_s']} s)")
        print_b(summary["b"])
    print("\n== (c) single-stick buckling (lab.euler_column; the same column with the "
          "fibre law; a clamped column as a pin-ended one of half its free length)")
    print_c(summary["c"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("-o", "--overrides", default=None,
                    help="an overrides YAML whose params: (and measured:) replace the "
                         "shipped values")
    ap.add_argument("--seeds", default="-1,0,1,2,3,4",
                    help="crush seeds (-1: the perfect coupon); write --seeds=-1,0")
    ap.add_argument("--fidelity", default="fine")
    ap.add_argument("--jobs", type=int, default=2, help="crushes run at once")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per crush")
    ap.add_argument("--skip-crush", action="store_true", help="only (a) and (c)")
    ap.add_argument("--summarize", action="store_true",
                    help="print the tables of DIR/summary.json again, run nothing")
    a = ap.parse_args()
    out = Path(a.out)
    if a.summarize:
        print_summary(json.loads((out / "summary.json").read_text(encoding="utf-8")))
        return
    out.mkdir(parents=True, exist_ok=True)
    mat, sp, raw = material_and_params(a.overrides)
    t0 = time.time()
    summary: dict = {"overrides": raw, "fidelity": a.fidelity}
    summary["a"] = part_a(mat, sp)
    summary["unglued"] = unglued_table(measured_bending())
    if not a.skip_crush:
        seeds = [int(s) for s in a.seeds.split(",")]
        print(f"== (b) crushing ({a.fidelity}, seeds {seeds})", flush=True)
        summary["b"] = part_b(out, raw, seeds, a.fidelity, a.jobs, a.threads)
        walls = [r["wall_s"] for v in summary["b"].values() for r in v["runs"]]
        summary["crush_wall_s"] = {"min": min(walls), "max": max(walls),
                                   "median": float(np.median(walls))}
    summary["c"] = part_c(mat, sp)
    summary["wall_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str),
                                      encoding="utf-8")
    print()
    print_summary(summary)
    print(f"\nwrote {out / 'summary.json'} ({summary['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
