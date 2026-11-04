"""One API for people, scripts and AI agents: inspect a CAD bridge, crush it, compare it
with a real test, run the virtual lab, calibrate, and Monte Carlo.

Each function takes plain inputs (paths, dicts) and returns a typed report from
:mod:`bridgesim.sim.report`, whose ``model_dump()`` is plain JSON. The command line
(``bridgesim inspect|crush|compare|lab|calibrate|montecarlo``), the MCP server
(``bridgesim mcp``) and the web app all call these.

Typical agent loop::

    rep = inspect("Bridge.SLDPRT")            # what bridgesim understood + questions
    # ask the designer the rep.questions; write the answers into overrides.yaml
    res = crush("Bridge.SLDPRT", overrides="overrides.yaml", out_dir="results")
    gap = compare(res, "my_crush_test.csv")   # after the real test: how close?
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bridgesim.cad import CadBody, load_cad, orientation_check
from bridgesim.cad.rules_cad import measure_cad
from bridgesim.materials import load_material
from bridgesim.rules import RuleSet, evaluate_values
from bridgesim.schema import Material
from bridgesim.sim import lab as labmod
from bridgesim.sim.crush import KGF, CrushOptions, CrushResult, run_crush
from bridgesim.sim.idealize import Overrides, build_model
from bridgesim.sim.materials import SimParams
from bridgesim.sim.model import Model
from bridgesim.sim.report import (
    BodyReport,
    CrushReport,
    Curve,
    EventOut,
    GapMetric,
    InspectReport,
    JointReport,
    LabReport,
    MonteCarloReport,
    Quality,
    Question,
    RealityGap,
    RuleLine,
    RulesSummary,
)
from bridgesim.yamlio import load_yaml

MASS_LIMIT_KG = 6.0


# ----------------------------------------------------------------------------- inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_overrides(overrides: str | Path | dict | None) -> tuple[Overrides, dict]:
    if overrides is None:
        return Overrides(), {}
    if isinstance(overrides, dict):
        d = overrides
    else:
        d = load_yaml(Path(overrides).read_text(encoding="utf-8")) or {}
    return Overrides.from_dict(d), d


def _inputs(cad: str | Path, overrides, material, sim_params, ruleset, units=None,
            up: str = "y", span: str = "x"):
    path = Path(cad)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    bodies = load_cad(path, units=units, up=up, span=span)
    ov, ov_raw = load_overrides(overrides)
    mat = material if isinstance(material, Material) else load_material(
        material or "popsicle_birch")
    sp = sim_params if isinstance(sim_params, SimParams) else SimParams.load(sim_params)
    sim_over = {k: v for k, v in ov.params.items() if not k.startswith("material.")}
    if sim_over:
        try:
            sp = sp.updated(sim_over)
        except KeyError as exc:
            raise ValueError(str(exc).strip("'\"")) from None
    mat_over = {k.split(".", 1)[1]: v for k, v in ov.params.items() if k.startswith("material.")}
    if mat_over:
        known = set(mat.props())
        bad = [k for k in mat_over if k not in known]
        if bad:
            raise ValueError(f"unknown material parameter(s) {bad}; known: "
                             f"{sorted('material.' + k for k in known)}")
        mat = _material_with(mat, mat_over)
    rs = ruleset if isinstance(ruleset, RuleSet) else RuleSet.load(ruleset or "troitsky_2027")
    return path, bodies, ov, ov_raw, mat, sp, rs


def _material_with(mat: Material, changes: dict[str, float]) -> Material:
    d = mat.model_dump()
    for k, v in changes.items():
        if k.startswith("glue."):
            d["glue"][k[5:]] = {"value": float(v), "source": "measured"}
        else:
            d[k] = {"value": float(v), "source": "measured"}
    return Material.model_validate(d)


def _rules(bodies: list[CadBody], mat: Material, rs: RuleSet, model: Model | None
           ) -> RulesSummary:
    meas = measure_cad(bodies, mat.density_kg_m3.value, mat.glue.mass_fraction.value,
                       rs.constants,
                       deck_y=model.params.get("deck_y") if model else None,
                       deck_z=model.deck_z if model else None)
    rep = evaluate_values(meas, rs)
    lines = [RuleLine(section=r.section, title=r.title, passed=r.passed,
                      measured=r.measured_text or "", penalty=r.penalty, bans=list(r.bans),
                      cause=r.cause or "", ambiguous=r.ambiguous) for r in rep.results]
    vals = {k: float(v) for k, v in meas.values.items() if isinstance(v, int | float)}
    return RulesSummary(total_penalty=rep.total_penalty, bans=list(rep.bans), lines=lines,
                        measurements=vals)


def _assumed(mat: Material, sp: SimParams) -> list[str]:
    return [f"material.{k}" for k in mat.assumed_keys()] + [f"sim.{k}" for k in
                                                            sp.assumed_keys()]


# ------------------------------------------------------------------------- diagnostics


def _point_in_mesh(P: np.ndarray, V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Ray casting along +X (with a tiny tilt) against a closed mesh."""
    d = np.array([1.0, 1e-4, 2e-4])
    d /= np.linalg.norm(d)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    cnt = np.zeros(len(P), int)
    for i, p in enumerate(P):
        s = p - a
        u = (s * h).sum(1) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = (q * e2).sum(1) * inv
        cnt[i] = int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum())
    return cnt % 2 == 1


def design_questions(bodies: list[CadBody], model: Model, mat: Material, sp: SimParams,
                     rules: RulesSummary) -> list[Question]:
    qs: list[Question] = []
    rng = np.random.default_rng(0)
    # interference: bodies overlapping in the CAD (a modelling error or a joint drawn as
    # an overlap)
    lo = np.array([b.mesh.bbox[0] for b in bodies]) + 0.5
    hi = np.array([b.mesh.bbox[1] for b in bodies]) - 0.5
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if (lo[i] > hi[j]).any() or (lo[j] > hi[i]).any():
                continue
            small, big = (i, j) if bodies[i].mesh.volume < bodies[j].mesh.volume else (j, i)
            # points really inside the smaller body (non-convex bodies are common), then
            # the fraction of them that is also inside the bigger one
            ms = bodies[small].mesh
            P = rng.uniform(*ms.bbox, size=(300, 3))
            P = P[_point_in_mesh(P, ms.vertices, ms.triangles)]
            if len(P) < 20:
                continue
            frac = float(_point_in_mesh(P, bodies[big].mesh.vertices,
                                        bodies[big].mesh.triangles).mean())
            if frac * ms.volume > 50.0 and frac > 0.03:
                qs.append(Question(
                    id=f"overlap:{bodies[small].name}|{bodies[big].name}", topic="geometry",
                    question=f"'{bodies[small].name}' overlaps '{bodies[big].name}' in the "
                             f"CAD (about {frac * ms.volume:.0f} mm^3, {frac:.0%} of it). Two "
                             "solid parts cannot occupy the same space: should one be cut "
                             "into the other (a notch), or is this a modelling error?",
                    assumed="they are treated as separate parts joined only where their "
                            "faces touch",
                    severity="important"))
    # weak joints: a bar whose joints are much weaker in shear than the bar in tension
    tau = mat.glue.tau_g_MPa.value
    by_body: dict[int, list] = {}
    for p in model.patches:
        by_body.setdefault(p.body_a, []).append(p)
        by_body.setdefault(p.body_b, []).append(p)
    for bi, info in enumerate(model.bodies):
        if info.kind != "bar" or info.f_t_eff_MPa <= 0:
            continue
        cap_member = info.f_t_eff_MPa * info.section_mm[0] * info.section_mm[1]
        glue = sum(p.area_mm2 for p in by_body.get(bi, []) if p.kind == "glue") * tau
        if glue and glue < 0.5 * cap_member:
            qs.append(Question(
                id=f"weak_joint:{info.name}", topic="design", body=info.name,
                question=f"The glued area on '{info.name}' carries about {glue / KGF:.0f} kgf "
                         f"in shear, much less than the member itself "
                         f"({cap_member / KGF:.0f} kgf in tension). Is the joint really this "
                         "small, or is a gusset or lap missing from the CAD?",
                assumed="the joint is exactly as drawn",
                severity="check"))
    # rules
    for ln in rules.lines:
        if ln.passed is False:
            qs.append(Question(
                id=f"rule:{ln.section}", topic="rules",
                question=f"Rule §{ln.section} {ln.title} fails ({ln.measured}; penalty "
                         f"{ln.penalty:g}{', bans ' + ', '.join(ln.bans) if ln.bans else ''})"
                         f"{': ' + ln.cause if ln.cause else ''}. Intended?",
                assumed="the design is simulated as drawn", severity="important"))
    mass = rules.measurements.get("mass_kg", 0.0)
    if mass > MASS_LIMIT_KG and mat.density_kg_m3.source == "assumed":
        qs.append(Question(
            id="material:density", topic="material",
            question=f"The bridge comes out at {mass:.2f} kg with an ASSUMED stick density "
                     f"of {mat.density_kg_m3.value:g} kg/m3 and glue at "
                     f"{mat.glue.mass_fraction.value:.0%} of the wood mass. What do your "
                     "sticks weigh (density = mass of 100 sticks / their volume)?",
            assumed=f"{mat.density_kg_m3.value:g} kg/m3",
            answer_with={"params": {"material.density_kg_m3": 600}}, severity="important"))
    assumed = [k for k in ("E_MPa", "f_t_MPa", "f_c_MPa") if k in mat.assumed_keys()]
    if assumed or "glue.tau_g_MPa" in mat.assumed_keys():
        qs.append(Question(
            id="material:measure", topic="material",
            question="Stick stiffness/strength and glue strength are textbook placeholders. "
                     "Have you tested sticks (3-point bending) and glued lap joints? Fill "
                     "validation/physical/*.csv and run `bridgesim calibrate`.",
            assumed="E = 10 GPa, f_t = 40, f_c = 30 MPa, glue 2 MPa (placeholders)",
            severity="important"))
    return qs


# ----------------------------------------------------------------------------- inspect


def _body_reports(model: Model) -> list[BodyReport]:
    out = []
    for b in model.bodies:
        why = b.notes[0].split(": ", 1)[-1] if b.notes else ""
        lo, hi = (b.bbox_mm if b.bbox_mm is not None else (np.zeros(3), np.zeros(3)))
        out.append(BodyReport(
            name=b.name, kind=b.kind, layup=b.layup, layup_source=b.layup_source,
            centroid_mm=[round(float(v), 1) for v in b.centroid_mm],
            bbox_min_mm=[round(float(v), 1) for v in lo],
            bbox_max_mm=[round(float(v), 1) for v in hi],
            axis=[round(float(v), 3) for v in b.axis] if b.axis is not None else None,
            mass_g=round(b.mass_kg * 1000, 2), volume_mm3=round(b.volume_mm3, 1),
            length_mm=round(b.length_mm, 1), section_mm=b.section_mm,
            thickness_mm=round(b.thickness_mm, 2), E_eff_MPa=round(b.E_eff_MPa, 1),
            f_t_eff_MPa=round(b.f_t_eff_MPa, 2), f_c_eff_MPa=round(b.f_c_eff_MPa, 2),
            why=why))
    return out


def _joint_reports(model: Model, mat: Material) -> list[JointReport]:
    tau = mat.glue.tau_g_MPa.value
    return [JointReport(a=model.bodies[p.body_a].name, b=model.bodies[p.body_b].name,
                        kind=p.kind, area_mm2=round(p.area_mm2, 1),
                        shear_capacity_N=round(p.area_mm2 * tau, 1))
            for p in model.patches]


def inspect(cad: str | Path, overrides=None, material=None, sim_params=None,
            ruleset=None, seed: int | None = 0, units=None, up: str = "y",
            span: str = "x", strict: bool = False) -> InspectReport:
    """Read the CAD file and report what bridgesim understood, the rule check and the
    questions to ask the designer. No simulation is run (a few seconds)."""
    path, bodies, ov, ov_raw, mat, sp, rs = _inputs(cad, overrides, material, sim_params,
                                                     ruleset, units, up, span)
    model = _build(bodies, mat, sp, ov, seed, 1.0, strict)
    rules = _rules(bodies, mat, rs, model)
    qs = [Question(**q) for q in model.questions] + design_questions(bodies, model, mat,
                                                                       sp, rules)
    return InspectReport(
        file=path.name, sha256=_sha(path), n_bodies=len(bodies),
        mass_kg=round(rules.measurements.get("mass_kg", model.bridge_mass_kg), 3),
        bodies=_body_reports(model), joints=_joint_reports(model, mat),
        loading={"plate_centre_mm": [round(model.mid_x, 1), round(model.deck_top_y, 1),
                                     round(model.deck_z, 1)],
                 "plate_mm": [200, 90], "deck_top_y_mm": model.params.get("deck_y"),
                 "plate_offset_mm": model.params.get("plate_offset_mm"),
                 "plate_rests_on_y_mm": model.deck_top_y,
                 "deck_bodies": model.params.get("deck_bodies", []),
                 "supports": "both ends rest on the press's flat base platform "
                             "(friction, no anchorage)"},
        rules=rules, questions=qs, assumed_parameters=_assumed(mat, sp),
        model={"nodes": int(model.n_nodes), "beam_elements": int(len(model.b_i)),
               "shell_elements": int(len(model.s_nodes)), "glue_points": int(len(model.c_a)),
               "glued_patches": len(model.patches), "floss_cables": int(len(model.k_a)),
               "platform_contact_points": int(len(model.p_att)),
               "plate_contact_points": int(len(model.q_att))},
        warnings=list(model.notes))


def _build(bodies, mat, sp, ov, seed, mesh_scale, strict):
    """Build the model, with orientation problems explained, and overrides problems
    raised as input errors with ``strict``."""
    problems = orientation_check(bodies)
    try:
        model = build_model(bodies, mat, sp, ov, seed=seed, mesh_scale=mesh_scale)
    except ValueError as exc:
        hint = ("; also: " + "; ".join(problems)) if problems else ""
        raise ValueError(f"{exc}{hint}") from None
    except Exception as exc:  # noqa: BLE001 - a wrongly oriented model breaks anywhere
        if not problems:
            raise
        raise ValueError("the model could not be built, probably because its orientation "
                         "or units are wrong: " + "; ".join(problems)
                         + f" (error: {type(exc).__name__})") from None
    model.notes = problems + model.notes
    for p in problems:
        model.questions.insert(0, {
            "id": "geometry:orientation", "topic": "geometry", "severity": "important",
            "question": f"Orientation or units look wrong: {p}.",
            "assumed": "X along the span, Y up, millimetres",
            "answer_with": {"cli": "--up-axis/--span-axis/--units"}})
    if strict:
        bad = [n for n in model.notes if n.startswith("overrides:")]
        if bad:
            raise ValueError("; ".join(bad))
    return model


# ------------------------------------------------------------------------------- crush


def _sequence(events: list) -> list[str]:
    out = []
    seen = set()
    for e in events:
        if e.kind in ("glue_damage",):
            continue
        key = (e.kind, e.what)
        if key in seen:
            continue
        seen.add(key)
        verb = {"member_fracture": "member broke", "member_shear": "member split in shear",
                "plate_crack": "plate cracked", "joint_failure": "glued joint separated",
                "floss_break": "floss snapped", "criterion": "§12.5 failure criterion",
                "instability": "instability", "numerical": "numerical problem"}.get(
                    e.kind, e.kind)
        out.append(f"{e.force_N / KGF:.1f} kgf (jack {e.travel_mm:.1f} mm): {verb}: {e.what}"
                   + (f" ({e.detail})" if e.detail and e.kind == "criterion" else ""))
        if len(out) >= 40:
            break
    return out


def crush(cad: str | Path, overrides=None, material=None, sim_params=None, ruleset=None,
          seed: int | None = 0, out_dir: str | Path | None = None,
          options: CrushOptions | None = None, progress=None,
          replay: bool = True, fidelity: str = "standard", units=None, up: str = "y",
          span: str = "x", strict: bool = False) -> tuple[CrushReport, CrushResult]:
    """Simulate the §12.5 crushing test of a CAD bridge. Writes ``crush.json``,
    ``curve.csv`` and ``replay.html`` (interactive 3D replay) to ``out_dir`` if given."""
    from bridgesim.sim.crush import FIDELITY

    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {', '.join(FIDELITY)}")
    scale, speed = FIDELITY[fidelity]
    path, bodies, ov, ov_raw, mat, sp, rs = _inputs(cad, overrides, material, sim_params,
                                                     ruleset, units, up, span)
    model = _build(bodies, mat, sp, ov, seed, scale, strict)
    rules = _rules(bodies, mat, rs, model)
    opt = options or CrushOptions(speed_mm_s=speed)
    if progress is not None:
        opt.progress = progress
    res = run_crush(model, opt)
    qs = [Question(**q) for q in model.questions] + design_questions(bodies, model, mat, sp,
                                                                       rules)
    c = res.curve
    step = max(1, len(c["t"]) // 400)
    curve = Curve(jack_travel_mm=[round(v, 3) for v in c["travel"][::step]],
                  load_kgf=[round(v / KGF, 2) for v in c["force"][::step]],
                  deflection_mm=[round(v, 3) for v in c["deflection"][::step]],
                  joints_broken=c["glue_broken"][::step],
                  members_broken=c["members_broken"][::step])
    mass = rules.measurements.get("mass_kg", model.bridge_mass_kg)
    rep = CrushReport(
        file=path.name, sha256=_sha(path), seed=seed, overrides=ov_raw,
        Fu_kgf=round(res.Fu_kgf, 2), Fu_N=round(res.Fu_N, 1), failure_mode=res.failure_mode,
        failure_detail=res.failure_detail, deflection_at_Fu_mm=round(res.deflection_at_Fu_mm, 2),
        mass_kg=round(mass, 3), efficiency_kgf_per_kg=round(res.Fu_kgf / mass, 2),
        criteria=res.criteria,
        events=_compact_events(res.events), n_events_total=len(res.events),
        failure_sequence=_sequence(res.events), curve=curve,
        quality=_quality(res, mat, sp),
        rules=rules, questions=qs, assumed_parameters=_assumed(mat, sp),
        fidelity=fidelity, Fu_first_joint_failure_kgf=_first_joint(res),
        warnings=list(model.notes))
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "curve.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "jack_travel_mm", "load_N", "load_kgf", "deflection_mm",
                        "joints_broken", "members_broken"])
            for i in range(len(c["t"])):
                w.writerow([f"{c['t'][i]:.6f}", f"{c['travel'][i]:.4f}",
                            f"{c['force'][i]:.2f}", f"{c['force'][i] / KGF:.3f}",
                            f"{c['deflection'][i]:.4f}", c["glue_broken"][i],
                            c["members_broken"][i]])
        rep.artifacts["curve_csv"] = str(out / "curve.csv")
        with (out / "events.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "jack_travel_mm", "load_kgf", "kind", "what", "detail",
                        "x_mm", "y_mm", "z_mm"])
            for e in res.events:
                d = e.as_dict()
                p = d.get("where_mm") or ["", "", ""]
                w.writerow([d["t_s"], d["jack_travel_mm"], d["load_kgf"], d["kind"],
                            d["what"], d["detail"], *p])
        rep.artifacts["events_csv"] = str(out / "events.csv")
        if replay:
            from bridgesim.sim.replay import replay_html
            (out / "replay.html").write_text(
                replay_html(res, bodies, title=f"{path.stem}: simulated crush test"),
                encoding="utf-8")
            rep.artifacts["replay_html"] = str(out / "replay.html")
        rep.artifacts["report_json"] = str(out / "crush.json")
        (out / "crush.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    return rep, res


def _compact_events(events) -> list[EventOut]:
    """The report keeps the first event of each kind for each part (the full log is in
    events.csv): 'glue line starts to fail' and repeated 'plate cracked' lines on the same
    part are summarised."""
    seen: dict[tuple[str, str], int] = {}
    out: list[EventOut] = []
    for e in events:
        key = (e.kind, e.what)
        if key in seen:
            seen[key] += 1
            continue
        seen[key] = 1
        out.append(EventOut(**e.as_dict()))
    for o in out:
        n = seen[(o.kind, o.what)]
        if n > 1:
            o.detail = (o.detail + f" (+{n - 1} more)").strip()
    return out


def _first_joint(res: CrushResult) -> float | None:
    hit = [e for e in res.events if e.kind in ("joint_failure", "member_shear")]
    return round(hit[0].force_N / KGF, 2) if hit else None


def _quality(res: CrushResult, mat: Material, sp: SimParams) -> Quality:
    reasons = []
    if res.kinetic_ratio_max > 0.1:
        reasons.append(f"loading not quasi-static (peak kinetic/work {res.kinetic_ratio_max:.2f}"
                       " > 0.1): run slower")
    if res.added_mass_frac > 0.05:
        reasons.append(f"{res.added_mass_frac:.1%} mass added for stability (> 5 %)")
    if res.failure_mode == "none":
        reasons.append("the bridge did not fail within the jack travel")
    if res.notes:
        reasons.append("the run was restarted with a smaller time step")
    return Quality(dt_s=res.dt, steps=res.steps, wall_s=round(res.wall_s, 1),
                   added_mass_fraction=round(res.added_mass_frac, 4),
                   kinetic_to_work_max=round(res.kinetic_ratio_max, 4),
                   settle_sag_mm=round(res.settle_sag_mm, 3), notes=res.notes,
                   trustworthy=not reasons, reasons=reasons,
                   speed_mm_s=getattr(res, "speed_mm_s", 0.0))


# ----------------------------------------------------------------------------- compare


def _read_test_csv(path: str | Path) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    load = [float(r["load_kgf"]) for r in rows if r.get("load_kgf") not in (None, "")]
    defl = [float(r["midspan_deflection_mm"]) for r in rows
            if r.get("midspan_deflection_mm") not in (None, "")]
    n = min(len(load), len(defl))
    ev = [(r.get("event") or "", r.get("notes") or "") for r in rows]
    fail_parts = [n_.strip() for e, n_ in ev if e.lower().startswith("fail") and n_]
    return {"load": np.array(load[:n]), "defl": np.array(defl[:n]), "fail_parts": fail_parts}


def compare(sim: CrushReport | str | Path, test_csv: str | Path) -> RealityGap:
    """Reality gap between a simulated crush (report or crush.json) and a physical crush
    test recorded in ``validation/physical/crush_test_template.csv`` format (rows of
    load_kgf and midspan_deflection_mm; a row with event 'failure' and the broken part in
    notes)."""
    rep = sim if isinstance(sim, CrushReport) else CrushReport.model_validate_json(
        Path(sim).read_text(encoding="utf-8"))
    t = _read_test_csv(test_csv)
    if len(t["load"]) < 2:
        raise ValueError(f"{test_csv}: need at least two rows with load_kgf and "
                         "midspan_deflection_mm")
    sl = np.array(rep.curve.load_kgf)
    sd = np.array(rep.curve.deflection_mm)
    tl, td = t["load"], t["defl"]
    peak_s, peak_t = float(rep.Fu_kgf), float(tl.max())
    i_t = int(np.argmax(tl))
    d_s, d_t = float(rep.deflection_at_Fu_mm), float(td[i_t])

    def slope(load, defl, pk):
        """Secant stiffness on the rising branch only (before the peak), between 10 % and
        50 % of the peak, so neither the seating toe nor the post-peak drop enters."""
        ipk = int(np.argmax(load))
        load, defl = load[: ipk + 1], defl[: ipk + 1]
        sel = (load > 0.1 * pk) & (load < 0.5 * pk) & (defl > 0)
        if sel.sum() < 2:
            sel = (load < 0.6 * pk) & (defl > 0)
        if sel.sum() < 2:
            return math.nan
        return float(np.polyfit(defl[sel], load[sel], 1)[0])

    k_s, k_t = slope(sl, sd, peak_s), slope(tl, td, peak_t)
    # curve: load at the test's deflections up to the test's peak
    upto = td <= d_t
    sim_at = np.interp(td[upto], sd[: int(np.argmax(sl)) + 1] if np.argmax(sl) > 0 else sd,
                       sl[: int(np.argmax(sl)) + 1] if np.argmax(sl) > 0 else sl)
    curve_err = float(np.sqrt(np.mean((sim_at - tl[upto]) ** 2)) / max(peak_t, 1e-9))

    def rel(a, b):
        return None if not (math.isfinite(a) and math.isfinite(b) and b) else (a - b) / b

    metrics = [
        GapMetric(name="ultimate load", simulated=round(peak_s, 2), measured=round(peak_t, 2),
                  error=rel(peak_s, peak_t), unit="kgf", meaning="§12.5 F_u"),
        GapMetric(name="initial stiffness", simulated=round(k_s, 2) if k_s == k_s else "n/a",
                  measured=round(k_t, 2) if k_t == k_t else "n/a", error=rel(k_s, k_t),
                  unit="kgf/mm", meaning="slope of load vs mid-span deflection"),
        GapMetric(name="deflection at ultimate load", simulated=round(d_s, 2),
                  measured=round(d_t, 2), error=rel(d_s, d_t), unit="mm"),
        GapMetric(name="curve shape", simulated="", measured="", error=curve_err,
                  meaning="RMS load difference along the test's curve / test peak"),
    ]
    if t["fail_parts"]:
        # what the simulation broke within 15 % of the peak load, compared by keywords
        near = [e for e in rep.events
                if e.kind in ("joint_failure", "member_fracture", "member_shear",
                              "plate_crack", "floss_break")
                and e.load_kgf >= 0.85 * rep.Fu_kgf][:12]
        hit = any(_same_part(p, e.what) for p in t["fail_parts"] for e in near)
        metrics.append(GapMetric(
            name="failure location", simulated="; ".join(e.what for e in near[:3]),
            measured="; ".join(t["fail_parts"]), error=0.0 if hit else 1.0,
            meaning="did the simulation break the same part near the peak"))
    w = {"ultimate load": 0.45, "initial stiffness": 0.2, "deflection at ultimate load": 0.1,
         "curve shape": 0.1, "failure location": 0.15}
    num = den = 0.0
    for mtr in metrics:
        if mtr.error is not None and math.isfinite(mtr.error):
            num += w[mtr.name] * min(1.0, abs(mtr.error))
            den += w[mtr.name]
    score = max(0.0, 100 * (1 - num / den)) if den else 0.0
    advice = []
    e_peak = metrics[0].error or 0.0
    e_k = metrics[1].error
    if e_k is not None and abs(e_k) > 0.2:
        advice.append("Stiffness is off by more than 20 %: measure E (stick bending) and the "
                      "glue line stiffness (sim.glue.K_n/K_s), and check the support "
                      "contact (supports sliding?).")
    if abs(e_peak) > 0.15:
        advice.append("Ultimate load is off by more than 15 %: compare the first thing that "
                      "broke. If it was a joint, calibrate tau_g with lap tests of that joint; "
                      "if a member, calibrate f_t/f_c with stick tests.")
    return RealityGap(subject=f"{rep.file} vs {Path(test_csv).name}", metrics=metrics,
                      score_0_100=round(score, 1),
                      competition_prediction_error=round(abs(peak_t - peak_s) / peak_t, 4),
                      advice=advice)


_STOP = {"the", "a", "of", "and", "at", "side", "ply", "plate", "near", "far", "left",
         "right", "broke", "failed", "joint", "glue"}


def _words(s: str) -> set[str]:
    import re

    w = set(re.findall(r"[a-z]+", s.lower()))
    side = {"l": "left", "r": "right"}
    w |= {side[x] for x in re.findall(r"\b([lr])\b", s.lower())}
    return w


def _same_part(observed: str, simulated: str) -> bool:
    """Keyword match: 'top chord far side' vs 'Top Chord (10-ply UD) - Far'."""
    o = _words(observed) - _STOP
    s_ = _words(simulated)
    if not o:
        return False
    return len(o & s_) / len(o) >= 0.6


# --------------------------------------------------------------------------- lab tests


def lab(material=None, sim_params=None, csv_dir: str | Path | None = None) -> LabReport:
    """Simulate the standard coupon tests with the current material (and, where the CSVs
    in ``csv_dir`` hold data, the team's own specimens) and report the reality gap."""
    mat = material if isinstance(material, Material) else load_material(
        material or "popsicle_birch")
    sp = sim_params if isinstance(sim_params, SimParams) else SimParams.load(sim_params)
    E, ft, fc = mat.E_MPa.value, mat.f_t_MPa.value, mat.f_c_MPa.value
    st = mat.stick
    tests: list[dict] = []
    r = labmod.three_point_bending(E, ft, fc, span=100.0, b=st.width_mm, t=st.thickness_mm,
                                   Gf=sp["wood.G_f_N_per_mm"])
    I = st.width_mm * st.thickness_mm**3 / 12
    mor = r.peak * 100 / 4 / (st.width_mm * st.thickness_mm**2 / 6)
    tests.append({"test": "stick 3-point bending, span 100 mm", "peak_N": round(r.peak, 2),
                  "stiffness_N_per_mm": round(r.stiffness, 3),
                  "theory_stiffness_N_per_mm": round(48 * E * I / 100**3, 3),
                  "modulus_of_rupture_MPa": round(mor, 1),
                  "material_f_b_MPa": mat.f_b_MPa.value})
    ov, wd = mat.glue.overlap_mm.value, st.width_mm
    tau_peak = mat.glue.tau_g_MPa.value * sp.values.get("glue.tau_peak_over_avg", 1.17)
    g = labmod.lap_shear(tau_peak, sp["glue.peel_to_shear"], sp["glue.K_n_N_per_mm3"],
                         sp["glue.K_s_N_per_mm3"], sp["glue.G_Ic_N_per_mm"],
                         sp["glue.G_IIc_N_per_mm"], overlap=ov, width=wd,
                         t=st.thickness_mm, E=E)
    tests.append({"test": f"single-lap glue joint {ov:g} x {wd:g} mm", "peak_N": round(g.peak, 1),
                  "tau_avg_MPa": round(g.peak / (ov * wd), 3),
                  "material_tau_g_MPa": mat.glue.tau_g_MPa.value})
    col = labmod.euler_column(E=E, L=st.length_mm, b=st.width_mm, t=st.thickness_mm)
    pcr = math.pi**2 * E * I / st.length_mm**2
    tests.append({"test": "one stick as a pin-ended column (L/1000 bow)",
                  "peak_N": round(col.peak, 2), "euler_N": round(pcr, 2)})
    gaps: list[RealityGap] = []
    if csv_dir:
        gaps = _lab_gaps(Path(csv_dir), mat, sp)
    return LabReport(tests=tests, gaps=gaps)


def _float(r: dict, k: str) -> float | None:
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def _lab_gaps(d: Path, mat: Material, sp: SimParams) -> list[RealityGap]:
    gaps = []
    f = d / "stick_bending_template.csv"
    if f.exists():
        rows = [r for r in csv.DictReader(f.read_text(encoding="utf-8").splitlines())]
        mets = []
        for r in rows:
            L, b, t = _float(r, "span_L_mm"), _float(r, "width_b_mm"), _float(r,
                                                                             "thickness_t_mm")
            F, dl, Ff = _float(r, "load_F_N"), _float(r, "deflection_mm"), _float(
                r, "failure_load_N")
            if not (L and b and t) or not ((F and dl) or Ff):
                continue
            sim = labmod.three_point_bending(mat.E_MPa.value, mat.f_t_MPa.value,
                                             mat.f_c_MPa.value, span=L, b=b, t=t,
                                             Gf=sp["wood.G_f_N_per_mm"])
            if F and dl:
                mets.append(GapMetric(name=f"{r['specimen']} stiffness",
                                      simulated=round(sim.stiffness, 3),
                                      measured=round(F / dl, 3),
                                      error=(sim.stiffness - F / dl) / (F / dl),
                                      unit="N/mm"))
            if Ff:
                mets.append(GapMetric(name=f"{r['specimen']} failure load",
                                      simulated=round(sim.peak, 2), measured=Ff,
                                      error=(sim.peak - Ff) / Ff, unit="N"))
        if mets:
            err = np.mean([min(1, abs(m_.error)) for m_ in mets if m_.error is not None])
            gaps.append(RealityGap(subject="stick bending", metrics=mets,
                                   score_0_100=round(100 * (1 - err), 1)))
    f = d / "glue_joint_template.csv"
    if f.exists():
        rows = [r for r in csv.DictReader(f.read_text(encoding="utf-8").splitlines())]
        mets = []
        for r in rows:
            ov, w, Ff = _float(r, "overlap_mm"), _float(r, "width_mm"), _float(
                r, "failure_load_N")
            faces = _float(r, "faces") or 1
            if not (ov and w and Ff):
                continue
            tau_peak = mat.glue.tau_g_MPa.value * sp.values.get("glue.tau_peak_over_avg",
                                                                 1.17)
            sim = labmod.lap_shear(tau_peak, sp["glue.peel_to_shear"],
                                   sp["glue.K_n_N_per_mm3"], sp["glue.K_s_N_per_mm3"],
                                   sp["glue.G_Ic_N_per_mm"], sp["glue.G_IIc_N_per_mm"],
                                   overlap=ov, width=w, E=mat.E_MPa.value)
            mets.append(GapMetric(name=f"{r['specimen']} joint strength",
                                  simulated=round(sim.peak * faces, 1), measured=Ff,
                                  error=(sim.peak * faces - Ff) / Ff, unit="N"))
        if mets:
            err = np.mean([min(1, abs(m_.error)) for m_ in mets if m_.error is not None])
            gaps.append(RealityGap(subject="glued lap joints", metrics=mets,
                                   score_0_100=round(100 * (1 - err), 1)))
    return gaps


# --------------------------------------------------------------------------- calibrate


def calibrate(csv_dir: str | Path, material=None) -> tuple[Material, list[str]]:
    """Fit E, f_b (and f_t, f_c in proportion) and tau_g from the team's test CSVs.
    Returns the new material (values marked ``measured``) and what was changed."""
    mat = material if isinstance(material, Material) else load_material(
        material or "popsicle_birch")
    d = Path(csv_dir)
    log = []
    changes: dict[str, float] = {}
    f = d / "stick_bending_template.csv"
    if f.exists():
        Es, fbs = [], []
        for r in csv.DictReader(f.read_text(encoding="utf-8").splitlines()):
            L, b, t = _float(r, "span_L_mm"), _float(r, "width_b_mm"), _float(r,
                                                                             "thickness_t_mm")
            F, dl, Ff = _float(r, "load_F_N"), _float(r, "deflection_mm"), _float(
                r, "failure_load_N")
            if L and b and t and F and dl:
                Es.append(F * L**3 / (48 * dl * b * t**3 / 12))
            if L and b and t and Ff:
                fbs.append(Ff * L / 4 / (b * t * t / 6))
        if Es:
            changes["E_MPa"] = float(np.mean(Es))
            log.append(f"E = {changes['E_MPa']:.0f} MPa from {len(Es)} bending tests")
        if fbs:
            fb = float(np.mean(fbs))
            ratio = fb / mat.f_b_MPa.value
            changes["f_b_MPa"] = fb
            changes["f_t_MPa"] = mat.f_t_MPa.value * ratio
            changes["f_c_MPa"] = mat.f_c_MPa.value * ratio
            log.append(f"f_b = {fb:.1f} MPa from {len(fbs)} tests; f_t and f_c scaled by "
                       f"{ratio:.2f} (their ratio to f_b kept)")
    f = d / "glue_joint_template.csv"
    if f.exists():
        taus = []
        for r in csv.DictReader(f.read_text(encoding="utf-8").splitlines()):
            ov, w, Ff = _float(r, "overlap_mm"), _float(r, "width_mm"), _float(
                r, "failure_load_N")
            faces = _float(r, "faces") or 1
            if ov and w and Ff:
                taus.append(Ff / (ov * w * faces))
        if taus:
            changes["glue.tau_g_MPa"] = float(np.mean(taus))
            log.append(f"tau_g = {changes['glue.tau_g_MPa']:.2f} MPa from {len(taus)} joints")
    if not changes:
        return mat, ["no test data found in " + str(d)]
    return _material_with(mat, changes), log


# ------------------------------------------------------------------------- monte carlo


def montecarlo(cad: str | Path, n: int = 8, overrides=None, material=None,
               sim_params=None, ruleset=None, options: CrushOptions | None = None,
               processes: int | None = None, fidelity: str = "standard",
               out_dir: str | Path | None = None, progress=None, units=None,
               up: str = "y", span: str = "x") -> MonteCarloReport:
    """Crush the bridge ``n`` times with different random imperfections, material scatter
    and plate placement (seeds 1..n) and report the spread of the ultimate load. Runs
    ``processes`` crushes at a time (default: one per two CPU cores), splitting the
    cores between them."""
    import concurrent.futures as cf
    import os

    cpus = os.cpu_count() or 1
    procs = max(1, min(n, processes or max(1, cpus // 2)))
    threads = max(1, cpus // procs)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    args = [(str(cad), overrides, material, sim_params, ruleset, s, options, fidelity,
             str(Path(out_dir) / f"seed_{s}") if out_dir else None, units, up, span)
            for s in range(1, n + 1)]
    res = []
    import multiprocessing as mp

    # spawn, not fork: numba's thread pool must not be forked
    with cf.ProcessPoolExecutor(max_workers=procs, initializer=_mc_init,
                                initargs=(threads,), mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_mc_one, a) for a in args]
        for k, fu in enumerate(cf.as_completed(futs)):
            res.append(fu.result())
            if progress:
                progress((k + 1) / n, f"run {k + 1}/{n}: F_u {res[-1][0]:.1f} kgf")
    F = sorted(r[0] for r in res)
    modes: dict[str, int] = {}
    for r in res:
        modes[r[1]] = modes.get(r[1], 0) + 1
    p = np.percentile(F, [5, 50, 95])
    rep = MonteCarloReport(file=Path(cad).name, n=n, Fu_kgf=[round(x, 2) for x in F],
                           p05_kgf=round(float(p[0]), 2), p50_kgf=round(float(p[1]), 2),
                           p95_kgf=round(float(p[2]), 2), failure_modes=modes,
                           recommended_prediction_kgf=round(float(p[1]), 1),
                           fidelity=fidelity,
                           untrustworthy_runs=sum(1 for r in res if not r[2]))
    if out_dir:
        (Path(out_dir) / "montecarlo.json").write_text(rep.model_dump_json(indent=2),
                                                      encoding="utf-8")
    return rep


def _mc_init(threads: int) -> None:
    try:
        import numba

        numba.set_num_threads(threads)
    except Exception:  # noqa: BLE001 - numba missing or fewer cores
        pass


def _mc_one(a):
    cad, overrides, material, sim_params, ruleset, seed, options, fidelity, out, units, \
        up, span = a
    rep, _ = crush(cad, overrides, material, sim_params, ruleset, seed=seed,
                   options=options, replay=False, fidelity=fidelity, out_dir=out,
                   units=units, up=up, span=span)
    first = rep.failure_sequence[0] if rep.failure_sequence else rep.failure_mode
    where = first.split(": ", 2)[-1] if ": " in first else first
    return rep.Fu_kgf, where, rep.quality.trustworthy


def dump(model_obj) -> str:
    return json.dumps(model_obj.model_dump(mode="json"), indent=2)
