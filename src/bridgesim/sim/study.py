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
from bridgesim.sim.idealize import Overrides, _group, build_model
from bridgesim.sim.materials import SimParams
from bridgesim.sim.model import Model
from bridgesim.sim.report import (
    BodyReport,
    Criteria,
    CriterionHit,
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
    weak_seen: set[str] = set()
    for bi, info in enumerate(model.bodies):
        if info.kind != "bar" or info.f_t_eff_MPa <= 0:
            continue
        cap_member = info.f_t_eff_MPa * info.section_mm[0] * info.section_mm[1]
        glue = sum(p.area_mm2 for p in by_body.get(bi, []) if p.kind == "glue") * tau
        if glue and glue < 0.5 * cap_member and _group(info.name) not in weak_seen:
            weak_seen.add(_group(info.name))
            qs.append(Question(
                id=f"weak_joint:{info.name}", topic="design", body=info.name,
                question=f"The glued area on '{info.name}' (and its copies) carries about "
                         f"{glue / KGF:.0f} kgf "
                         f"in shear, much less than the member itself "
                         f"({cap_member / KGF:.0f} kgf in tension). Is the joint really this "
                         "small, or is a gusset or lap missing from the CAD?",
                assumed="the joint is exactly as drawn",
                answer_with={"cad": f"add the missing gusset or lap on '{info.name}' and "
                                    "crush again; if the joint is as drawn, nothing to do",
                             "bodies": {info.name: {
                                 "glue_factor": "<only if its glue is weaker or stronger "
                                                "than a full glue line, e.g. 0.7>"}}},
                severity="check"))
    # rules
    for ln in rules.lines:
        if ln.passed is False:
            qs.append(Question(
                id=f"rule:{ln.section}", topic="rules",
                question=f"Rule §{ln.section} {ln.title} fails ({ln.measured}; penalty "
                         f"{ln.penalty:g}{', bans ' + ', '.join(ln.bans) if ln.bans else ''})"
                         f"{': ' + ln.cause if ln.cause else ''}. Intended?",
                assumed="the design is simulated as drawn",
                answer_with={"cad": f"change the design so §{ln.section} passes "
                                    f"({ln.cause or ln.measured}), or accept the penalty"},
                severity="important"))
    mass = rules.measurements.get("mass_kg", 0.0)
    if mass > MASS_LIMIT_KG and mat.density_kg_m3.source == "assumed":
        qs.append(Question(
            id="material:density", topic="material",
            question=f"The bridge comes out at {mass:.2f} kg with an ASSUMED stick density "
                     f"of {mat.density_kg_m3.value:g} kg/m3 and glue at "
                     f"{mat.glue.mass_fraction.value:.0%} of the wood mass. What do your "
                     "sticks weigh (density = mass of 100 sticks / their volume)?",
            assumed=f"{mat.density_kg_m3.value:g} kg/m3",
            answer_with={"params": {"material.density_kg_m3": "<measured density, kg/m3>"}},
            severity="important"))
    assumed = [k for k in ("E_MPa", "f_t_MPa", "f_c_MPa") if k in mat.assumed_keys()]
    if assumed or "glue.tau_g_MPa" in mat.assumed_keys():
        qs.append(Question(
            id="material:measure", topic="material",
            question="Stick stiffness/strength and glue strength are textbook placeholders. "
                     "Have you tested sticks (3-point bending) and glued lap joints? Fill "
                     "validation/physical/*.csv and run `bridgesim calibrate`.",
            assumed="E = 10 GPa, f_t = 40, f_c = 30 MPa, glue 2 MPa (placeholders)",
            answer_with={"cli": "bridgesim calibrate validation/physical --out "
                                "my_material.yaml, then crush with --material "
                                "my_material.yaml"},
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
                "joint_cracking": "glued joint cracking",
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
          span: str = "x", strict: bool = False, rate_check: bool = False
          ) -> tuple[CrushReport, CrushResult]:
    """Simulate the §12.5 crushing test of a CAD bridge. Writes ``crush.json``,
    ``curve.csv``, ``events.csv`` and ``replay.html`` (interactive 3D replay) to
    ``out_dir`` if given. ``rate_check`` crushes the bridge a second time at half the
    jack speed and reports how much F_u changes (``quality.rate_check``)."""
    from bridgesim.sim.crush import FIDELITY

    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {', '.join(FIDELITY)}")
    scale, speed = FIDELITY[fidelity]
    path, bodies, ov, ov_raw, mat, sp, rs = _inputs(cad, overrides, material, sim_params,
                                                     ruleset, units, up, span)
    if progress is not None:
        progress(0.0, f"read {len(bodies)} bodies; building the structural model")
    model = _build(bodies, mat, sp, ov, seed, scale, strict)
    rules = _rules(bodies, mat, rs, model)
    opt = options or CrushOptions(speed_mm_s=speed)
    if progress is not None:
        opt.progress = progress
    res = run_crush(model, opt)
    rc = None
    if rate_check:
        import dataclasses

        slow = dataclasses.replace(opt, speed_mm_s=opt.speed_mm_s / 2)
        if progress is not None:
            slow.progress = lambda f, msg: progress(f, "rate check (half speed): " + msg)
        res2 = run_crush(_build(bodies, mat, sp, ov, seed, scale, False), slow)
        rc = {"speed_mm_s": slow.speed_mm_s, "Fu_kgf": round(res2.Fu_kgf, 2),
              "change": round((res.Fu_kgf - res2.Fu_kgf) / max(res2.Fu_kgf, 1e-9), 4),
              "failure_mode": res2.failure_mode}
    qs = [Question(**q) for q in model.questions] + design_questions(bodies, model, mat, sp,
                                                                       rules)
    c = res.curve
    step = max(1, len(c["t"]) // 400)
    keep = set(range(0, len(c["t"]), step))
    if c["force"] and res.Fu_N > 0:  # always keep the point where F_u was reached
        keep.add(next((i for i, f in enumerate(c["force"]) if f >= 0.999999 * res.Fu_N),
                      len(c["force"]) - 1))
    keep = sorted(keep)
    curve = Curve(jack_travel_mm=[round(c["travel"][i], 3) for i in keep],
                  load_kgf=[round(c["force"][i] / KGF, 2) for i in keep],
                  deflection_mm=[round(c["deflection"][i], 3) for i in keep],
                  joints_broken=[c["glue_broken"][i] for i in keep],
                  members_broken=[c["members_broken"][i] for i in keep])
    mass = rules.measurements.get("mass_kg", model.bridge_mass_kg)
    rep = CrushReport(
        file=path.name, sha256=_sha(path), seed=seed, overrides=ov_raw,
        Fu_kgf=round(res.Fu_kgf, 2), Fu_N=round(res.Fu_N, 1), failure_mode=res.failure_mode,
        failure_detail=res.failure_detail, deflection_at_Fu_mm=round(res.deflection_at_Fu_mm, 2),
        mass_kg=round(mass, 3), efficiency_kgf_per_kg=round(res.Fu_kgf / mass, 2),
        criteria=_criteria(res.criteria),
        failures=_failures(res.events, _fail_time(res)),
        events=_compact_events(res.events), n_events_total=len(res.events),
        failure_sequence=_sequence(res.events), curve=curve,
        quality=_quality(res, mat, sp, rc),
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
                        "x_mm", "y_mm", "z_mm", "fraction_separated"])
            for e in res.events:
                d = e.as_dict()
                p = d.get("where_mm") or ["", "", ""]
                fr = d.get("fraction_separated")
                w.writerow([d["t_s"], d["jack_travel_mm"], d["load_kgf"], d["kind"],
                            d["what"], d["detail"], *p, "" if fr is None else fr])
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


MAJOR = ("joint_cracking", "joint_failure", "member_fracture", "member_shear",
         "plate_crack", "floss_break")


def _failures(events, fail_t: float | None, limit: int = 60) -> list[EventOut]:
    """Major events in order, first per (kind, part), up to the collapse plus a little."""
    out, seen = [], set()
    for e in events:
        if e.kind not in MAJOR or (e.kind, e.what) in seen:
            continue
        if fail_t is not None and e.t > fail_t + 0.002:
            break
        seen.add((e.kind, e.what))
        out.append(EventOut(**e.as_dict()))
        if len(out) >= limit:
            break
    return out


def _criteria(c: dict) -> Criteria:
    def hit(d):
        if d is None:
            return None
        ld = d.get("load_N")
        return CriterionHit(t_s=round(float(d.get("t_s", 0.0)), 5),
                            jack_travel_mm=None if d.get("travel_mm") is None
                            else round(float(d["travel_mm"]), 2),
                            load_kgf=None if ld is None else round(float(ld) / KGF, 2),
                            what=str(d.get("what") or d.get("part") or d.get("detail") or ""))

    return Criteria(**{k: hit(c.get(k)) for k in ("deflection", "contact", "joint_or_shear",
                                                  "instability")})


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


def brief(rep: CrushReport, n_sequence: int = 12) -> dict:
    """The crush report without the curve and the event log (they are in the files of
    ``artifacts``): what an agent needs to decide, in a few kB."""
    d = rep.model_dump(mode="json", exclude={"curve", "events"})
    d["kind"] = "crush_brief"
    seq = d["failure_sequence"]
    d["failure_sequence"] = seq[:n_sequence] + (
        [f"... {len(seq) - n_sequence} more (events.csv)"] if len(seq) > n_sequence else [])
    r = d["rules"]
    r["lines"] = [ln for ln in r.get("lines", []) if not ln.get("passed", True)
                  or ln.get("penalty") or ln.get("bans")]
    order = {"important": 0, "check": 1, "info": 2}
    d["questions"] = sorted(
        ({"id": q["id"], "severity": q["severity"], "question": q["question"],
          "answer_with": q.get("answer_with")} for q in d["questions"]),
        key=lambda q: order.get(q["severity"], 3))
    c = rep.curve
    if c.load_kgf and rep.Fu_kgf > 0:  # where F_u was reached (not a later spike)
        k = next((i for i, v in enumerate(c.load_kgf) if v >= 0.999 * rep.Fu_kgf),
                 len(c.load_kgf) - 1)
        d["peak"] = {"load_kgf": c.load_kgf[k], "jack_travel_mm": c.jack_travel_mm[k],
                     "deflection_mm": c.deflection_mm[k]}
    return d


def _fail_time(res: CrushResult) -> float | None:
    """Simulated time at which the bridge was judged failed (from its jack travel)."""
    tr = res.curve.get("travel") or []
    if not tr or res.failure_mode in ("none", "instability"):
        return None
    i = next((k for k, v in enumerate(tr) if v >= res.t_failure), len(tr) - 1)
    return float(res.curve["t"][i])


def _first_joint(res: CrushResult) -> float | None:
    hit = [e for e in res.events if e.kind in ("joint_failure", "member_shear")]
    return round(hit[0].force_N / KGF, 2) if hit else None


def _quality(res: CrushResult, mat: Material, sp: SimParams, rc: dict | None = None
             ) -> Quality:
    reasons = []
    if res.kinetic_ratio_max > 0.1 and not (rc and abs(rc["change"]) < 0.05):
        reasons.append(f"loading not quasi-static (peak kinetic/work {res.kinetic_ratio_max:.2f}"
                       " > 0.1): run slower or use --rate-check")
    if rc and abs(rc["change"]) >= 0.05:
        reasons.append(f"F_u changes by {rc['change']:+.0%} at half the jack speed: either "
                       "the simulated test is too fast, or the failure path is sensitive "
                       "to small differences (two failure modes compete); run montecarlo "
                       "to see the spread and give F_u as a range")
    if res.added_mass_frac > 0.0505:  # the solver caps it at 5 %
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
                   speed_mm_s=getattr(res, "speed_mm_s", 0.0), rate_check=rc,
                   holds=getattr(res, "holds", 0),
                   hold_time_s=round(getattr(res, "hold_time_s", 0.0), 4))


# ----------------------------------------------------------------------------- compare


def _read_test_csv(path: str | Path) -> dict[str, Any]:
    """Rows of a crush test record. The curve may be load vs mid-span deflection
    (``midspan_deflection_mm``) or vs jack stroke (``jack_travel_mm``); a record of the
    peak alone (one row, or only a ``failure`` row with its load) is accepted too."""
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    if not rows or "load_kgf" not in rows[0]:
        raise ValueError(f"{path}: needs a load_kgf column (see "
                         "validation/physical/crush_test_template.csv)")
    xcol = "midspan_deflection_mm" if any(r.get("midspan_deflection_mm") not in (None, "")
                                          for r in rows) else "jack_travel_mm"
    load, x = [], []
    for r in rows:
        lv, xv = _float(r, "load_kgf"), _float(r, xcol)
        if lv is not None and xv is not None:
            load.append(lv)
            x.append(xv)
    peaks = [_float(r, "load_kgf") for r in rows if _float(r, "load_kgf") is not None]
    ev = [(r.get("event") or "", r.get("notes") or "") for r in rows]
    fail_parts = [n_.strip() for e, n_ in ev if e.lower().startswith("fail") and n_]
    return {"load": np.array(load), "x": np.array(x), "x_kind": xcol,
            "peak": max(peaks) if peaks else math.nan, "fail_parts": fail_parts}


def compare(sim: CrushReport | str | Path, test_csv: str | Path) -> RealityGap:
    """Reality gap between a simulated crush (report or crush.json) and a physical crush
    test (``validation/physical/crush_test_template.csv``): load_kgf against
    midspan_deflection_mm or jack_travel_mm, a row with event 'failure' and the broken part
    in notes. With the peak load alone (and the broken part), the ultimate load, the §12.7
    prediction error and the failure location are still compared."""
    rep = sim if isinstance(sim, CrushReport) else CrushReport.model_validate_json(
        Path(sim).read_text(encoding="utf-8"))
    t = _read_test_csv(test_csv)
    peak_t = float(t["peak"])
    if not math.isfinite(peak_t) or peak_t <= 0:
        raise ValueError(f"{test_csv}: no load_kgf value")
    peak_s = float(rep.Fu_kgf)
    has_curve = len(t["load"]) >= 3
    jack = t["x_kind"] == "jack_travel_mm"
    sl = np.array(rep.curve.load_kgf)
    sx = np.array(rep.curve.jack_travel_mm if jack else rep.curve.deflection_mm)
    # the simulated record up to F_u: after the bridge is judged failed the jack goes on
    # (for the replay) and may press on the wreckage, which is not part of the test
    if len(sl) and peak_s > 0:
        i_fu = int(np.argmax(sl >= 0.999 * peak_s))
        sl, sx = sl[: i_fu + 1], sx[: i_fu + 1]

    def slope(load, x, pk):
        """Secant stiffness on the rising branch between the first points where the load
        reaches 10 % and 50 % of the peak (interpolated: the sampling does not matter)."""
        ipk = int(np.argmax(load))
        load, x = load[: ipk + 1], x[: ipk + 1]

        def cross(level):
            i = int(np.argmax(load >= level))
            if load[i] < level:
                return math.nan
            if i == 0:
                return float(x[0])
            f = (level - load[i - 1]) / max(load[i] - load[i - 1], 1e-12)
            return float(x[i - 1] + f * (x[i] - x[i - 1]))

        d10, d50 = cross(0.1 * pk), cross(0.5 * pk)
        if not (math.isfinite(d10) and math.isfinite(d50)) or d50 <= d10:
            return math.nan
        return 0.4 * pk / (d50 - d10)

    def rel(a, b):
        return None if not (math.isfinite(a) and math.isfinite(b) and b) else (a - b) / b

    what = "jack stroke" if jack else "mid-span deflection"
    metrics = [GapMetric(name="ultimate load", simulated=round(peak_s, 2),
                         measured=round(peak_t, 2), error=rel(peak_s, peak_t), unit="kgf",
                         meaning="§12.5 F_u")]
    if has_curve:
        tl, tx = t["load"], t["x"]
        i_t = int(np.argmax(tl))
        k_s, k_t = slope(sl, sx, peak_s), slope(tl, tx, float(tl.max()))
        x_s = float(sx[-1]) if len(sx) else math.nan
        x_t = float(tx[i_t])
        upto = tx <= x_t
        sim_at = np.interp(tx[upto], sx, sl) if len(sx) > 1 else np.zeros(int(upto.sum()))
        curve_err = float(np.sqrt(np.mean((sim_at - tl[upto]) ** 2)) / peak_t)
        metrics += [
            GapMetric(name="initial stiffness",
                      simulated=round(k_s, 2) if math.isfinite(k_s) else "n/a",
                      measured=round(k_t, 2) if math.isfinite(k_t) else "n/a",
                      error=rel(k_s, k_t), unit="kgf/mm",
                      meaning=f"secant of load vs {what} between 10 % and 50 % of the peak"),
            GapMetric(name=f"{what} at ultimate load", simulated=round(x_s, 2),
                      measured=round(x_t, 2), error=rel(x_s, x_t), unit="mm"),
            GapMetric(name="curve shape", simulated="", measured="", error=curve_err,
                      meaning="RMS load difference along the test's curve / test peak"),
        ]
    if t["fail_parts"]:
        # the test's first failure against the simulation's governing failures: the
        # first ones within 10 % of F_u, before the collapse
        gov = [f for f in rep.failures if f.kind != "joint_cracking"
               and f.load_kgf >= 0.9 * peak_s][:3] or [f for f in rep.failures
                                                       if f.kind != "joint_cracking"][:3]
        obs = t["fail_parts"][0]
        score = max((_part_match(obs, f.what) for f in gov), default=0.0)
        metrics.append(GapMetric(
            name="failure location", simulated="; ".join(f.what for f in gov),
            measured=obs, error=1.0 - score,
            meaning="did the simulation break the same part, on the same side, first "
                    "(0: yes; 0.5: same part, other side; 1: no)"))
    # score: each metric 0 (as far off as tol, or worse) to 1 (exact); ratios are
    # compared on a log scale, so 2x too stiff and 2x too soft count the same
    tol = {"ultimate load": math.log(1.5), "initial stiffness": math.log(2.0),
           f"{what} at ultimate load": math.log(2.0), "curve shape": 0.4,
           "failure location": 1.0}
    w = {"ultimate load": 0.4, "initial stiffness": 0.2, f"{what} at ultimate load": 0.1,
         "curve shape": 0.15, "failure location": 0.15}
    num = den = 0.0
    for mtr in metrics:
        e = mtr.error
        if e is None or not math.isfinite(e):
            continue
        if mtr.name in ("curve shape", "failure location"):
            dev = abs(e)
        else:
            dev = abs(math.log(max(1 + e, 1e-6)))
        num += w[mtr.name] * min(1.0, dev / tol[mtr.name])
        den += w[mtr.name]
    score = max(0.0, 100 * (1 - num / den)) if den else 0.0
    advice = []
    e_peak = metrics[0].error or 0.0
    e_k = next((m_.error for m_ in metrics if m_.name == "initial stiffness"), None)
    if not has_curve:
        advice.append("Only the peak load was compared: log load against mid-span "
                      "deflection (or jack stroke) during the test to compare stiffness "
                      "and the shape of the curve.")
    if e_k is not None and abs(e_k) > 0.2:
        advice.append("Stiffness is off by more than 20 %: measure E (stick bending) and the "
                      "glue line stiffness (glue.K_n/K_s), and check the support contact "
                      "(supports sliding?)" + (". The simulated jack stroke includes a "
                      "20 kN/mm machine stiffness (crusher)." if jack else "."))
    if abs(e_peak) > 0.15:
        advice.append("Ultimate load is off by more than 15 %: compare the first thing that "
                      "broke. If it was a joint, calibrate tau_g with lap tests of that joint; "
                      "if a member, calibrate f_t/f_c with stick tests "
                      "(bridgesim sensitivity says which values matter most).")
    loc = next((m_ for m_ in metrics if m_.name == "failure location"), None)
    if loc is not None and loc.error and loc.error > 0:
        advice.append("The test broke elsewhere: look at the simulated failure sequence for "
                      "that part; its glue or strength may be overestimated (glue_factor, "
                      "strength_factor in the overrides).")
    return RealityGap(subject=f"{rep.file} vs {Path(test_csv).name}", metrics=metrics,
                      score_0_100=round(score, 1),
                      competition_prediction_error=round(abs(peak_t - peak_s) / peak_t, 4),
                      advice=advice)


_STOP = {"the", "a", "of", "and", "at", "side", "ply", "plate", "broke", "failed", "joint",
         "glue", "ud", "mirror", "copy", "x", "member"}
_SIDES = {"near", "far", "left", "right"}


def _words(s: str) -> set[str]:
    import re

    low = s.lower()
    w = set(re.findall(r"[a-z]+", low))
    side = {"l": "left", "r": "right"}
    w |= {side[x] for x in re.findall(r"\b([lr])\b", low)}
    return w


def _part_match(observed: str, simulated: str) -> float:
    """1 if the observed part ('top chord, far side, left') is the simulated one ('Top
    Chord (10-ply UD) - L Far'), 0.5 if it is the same part on another side, else 0."""
    o, s_ = _words(observed) - _STOP, _words(simulated) - _STOP
    o_sides, s_sides = o & _SIDES, s_ & _SIDES
    o_part = o - _SIDES
    if not o_part or len(o_part & s_) / len(o_part) < 0.6:
        return 0.0
    return 1.0 if o_sides <= s_sides else 0.5


def _same_part(observed: str, simulated: str) -> bool:
    return _part_match(observed, simulated) == 1.0


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
    memo: dict = {}

    def bending(L, b, t):  # specimens of the same size share one simulation
        key = (round(L, 3), round(b, 3), round(t, 3))
        if key not in memo:
            memo[key] = labmod.three_point_bending(mat.E_MPa.value, mat.f_t_MPa.value,
                                                   mat.f_c_MPa.value, span=L, b=b, t=t,
                                                   Gf=sp["wood.G_f_N_per_mm"])
        return memo[key]

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
            sim = bending(L, b, t)
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


def calibrate(csv_dir: str | Path, material=None, sim_params=None
              ) -> tuple[Material, list[str]]:
    """Fit the material to the team's coupon tests so that the *simulated* coupons
    (``bridgesim lab``) reproduce them: E from the bending stiffness (the simulated
    coupon matches 48 E I / L^3), the strengths f_t and f_c (scaled together, keeping
    their ratio) so the simulated sticks break at the measured loads, and tau_g so the
    simulated lap joints of the measured overlaps break at the measured loads. Returns the
    new material (values marked ``measured``) and what was changed."""
    mat = material if isinstance(material, Material) else load_material(
        material or "popsicle_birch")
    sp = sim_params if isinstance(sim_params, SimParams) else SimParams.load(sim_params)
    d = Path(csv_dir)
    log = []
    changes: dict[str, float] = {}
    f = d / "stick_bending_template.csv"
    if f.exists():
        Es, fails = [], []
        for r in csv.DictReader(f.read_text(encoding="utf-8").splitlines()):
            L, b, t = _float(r, "span_L_mm"), _float(r, "width_b_mm"), _float(r,
                                                                             "thickness_t_mm")
            F, dl, Ff = _float(r, "load_F_N"), _float(r, "deflection_mm"), _float(
                r, "failure_load_N")
            if L and b and t and F and dl:
                Es.append(F * L**3 / (48 * dl * b * t**3 / 12))
            if L and b and t and Ff:
                fails.append((L, b, t, Ff))
        E = float(np.mean(Es)) if Es else mat.E_MPa.value
        if Es:
            changes["E_MPa"] = E
            log.append(f"E = {E:.0f} MPa from {len(Es)} bending tests")
        if fails:
            fb = float(np.mean([Ff * L / 4 / (b * t * t / 6) for L, b, t, Ff in fails]))
            changes["f_b_MPa"] = fb
            # inverse fit: scale f_t and f_c until the simulated sticks break at the
            # measured loads (bending strength of wood comes from both)
            L, b, t = (float(np.mean([x[i] for x in fails])) for i in range(3))
            target = float(np.mean([x[3] for x in fails]))
            k = 1.0
            for _ in range(3):
                sim = labmod.three_point_bending(E, mat.f_t_MPa.value * k,
                                                 mat.f_c_MPa.value * k, span=L, b=b, t=t,
                                                 Gf=sp["wood.G_f_N_per_mm"])
                k *= target / max(sim.peak, 1e-9)
                if abs(sim.peak / target - 1) < 0.005:
                    break
            changes["f_t_MPa"] = mat.f_t_MPa.value * k
            changes["f_c_MPa"] = mat.f_c_MPa.value * k
            log.append(f"f_b = {fb:.1f} MPa from {len(fails)} tests; f_t and f_c scaled by "
                       f"{k:.3f} so the simulated sticks break at the measured "
                       f"{target:.1f} N")
    f = d / "glue_joint_template.csv"
    if f.exists():
        joints = []
        for r in csv.DictReader(f.read_text(encoding="utf-8").splitlines()):
            ov, w, Ff = _float(r, "overlap_mm"), _float(r, "width_mm"), _float(
                r, "failure_load_N")
            faces = _float(r, "faces") or 1
            if ov and w and Ff:
                joints.append((ov, w, Ff / faces))
        if joints:
            tau = float(np.mean([F_ / (ov * w) for ov, w, F_ in joints]))
            ratio = sp.values.get("glue.tau_peak_over_avg", 1.17)
            E = changes.get("E_MPa", mat.E_MPa.value)
            # inverse fit through the simulated lap joint of each measured geometry
            for _ in range(3):
                errs = []
                for ov, w, F_ in joints:
                    sim = labmod.lap_shear(tau * ratio, sp["glue.peel_to_shear"],
                                           sp["glue.K_n_N_per_mm3"], sp["glue.K_s_N_per_mm3"],
                                           sp["glue.G_Ic_N_per_mm"],
                                           sp["glue.G_IIc_N_per_mm"], overlap=ov, width=w,
                                           E=E)
                    errs.append(F_ / max(sim.peak, 1e-9))
                k = float(np.mean(errs))
                tau *= k
                if abs(k - 1) < 0.005:
                    break
            changes["glue.tau_g_MPa"] = tau
            log.append(f"tau_g = {tau:.2f} MPa from {len(joints)} joints (so the simulated "
                       "lap joints of the measured overlaps break at the measured loads)")
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
                progress((k + 1) / n, f"run {k + 1}/{n}: F_u {res[-1]['Fu_kgf']:.1f} kgf")
    F = sorted(r["Fu_kgf"] for r in res)
    modes: dict[str, int] = {}
    for r in res:
        modes[r["first_failure"]] = modes.get(r["first_failure"], 0) + 1
    p = np.percentile(F, [5, 50, 95])
    rep = MonteCarloReport(file=Path(cad).name, n=n, Fu_kgf=[round(x, 2) for x in F],
                           p05_kgf=round(float(p[0]), 2), p50_kgf=round(float(p[1]), 2),
                           p95_kgf=round(float(p[2]), 2), failure_modes=modes,
                           recommended_prediction_kgf=round(best_prediction(F), 1),
                           fidelity=fidelity,
                           untrustworthy_runs=sum(1 for r in res if not r["trustworthy"]))
    _, _, _, _, mat_, sp_, _ = _inputs(cad, overrides, material, sim_params, ruleset, units,
                                       up, span)
    rep.assumed_parameters = _assumed(mat_, sp_)
    rep.note = ("The spread covers member bow, stick-to-stick and joint-to-joint scatter "
                "and plate placement. It does not cover the material values themselves: "
                + (f"{len(rep.assumed_parameters)} are still placeholders, which can move "
                   "F_u by tens of % (bridgesim sensitivity shows which); measure them "
                   "before trusting this range." if rep.assumed_parameters else
                   "they are all measured."))
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
    return {"seed": seed, "Fu_kgf": rep.Fu_kgf, "first_failure": where,
            "trustworthy": rep.quality.trustworthy, "mass_kg": rep.mass_kg,
            "efficiency_kgf_per_kg": rep.efficiency_kgf_per_kg}


def best_prediction(F) -> float:
    """Prediction minimising the expected §12.7 error E = |F - p| / F over the runs F:
    the median of F weighted by 1/F (lower than the plain median)."""
    F = np.sort(np.asarray(F, float))
    if not len(F):
        return 0.0
    w = 1.0 / np.maximum(F, 1e-9)
    c = np.cumsum(w) / w.sum()
    return float(F[int(np.searchsorted(c, 0.5))])


def _pool_runs(args, n_runs, processes, progress, label):
    import concurrent.futures as cf
    import multiprocessing as mp
    import os

    cpus = os.cpu_count() or 1
    procs = max(1, min(n_runs, processes or max(1, cpus // 2)))
    threads = max(1, cpus // procs)
    out = []
    with cf.ProcessPoolExecutor(max_workers=procs, initializer=_mc_init,
                                initargs=(threads,), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_mc_one, a): tag for tag, a in args}
        for k, fu in enumerate(cf.as_completed(futs)):
            r = fu.result()
            r["tag"] = futs[fu]
            out.append(r)
            if progress:
                progress((k + 1) / n_runs, f"{label} {k + 1}/{n_runs}: {r['tag']} seed "
                                           f"{r['seed']} F_u {r['Fu_kgf']:.1f} kgf")
    return out


def summarize_pairs(ra: list[dict], rb: list[dict], name_a: str, name_b: str,
                    fidelity: str):
    """Paired statistics of crush runs of two designs with the same seeds (dicts with
    seed, Fu_kgf, mass_kg, efficiency_kgf_per_kg, first_failure, trustworthy)."""
    from scipy import stats

    from bridgesim.sim.report import DesignComparison, DesignRun

    ra = sorted(ra, key=lambda r: r["seed"])
    rb = sorted(rb, key=lambda r: r["seed"])
    if [r["seed"] for r in ra] != [r["seed"] for r in rb] or not ra:
        raise ValueError("the two designs must have runs with the same seeds")
    n = len(ra)

    def change(key):
        d = np.array([(b[key] - a[key]) / max(a[key], 1e-9) for a, b in zip(ra, rb,
                                                                           strict=True)])
        mean = float(d.mean())
        if len(d) < 2:
            return mean, (math.nan, math.nan), d
        half = float(stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
        return mean, (round(mean - half, 4), round(mean + half, 4)), d

    dF, ciF, d = change("Fu_kgf")
    dE, ciE, _ = change("efficiency_kgf_per_kg")
    sig = bool(ciF[0] > 0 or ciF[1] < 0) if not math.isnan(ciF[0]) else False
    wins = int((d > 0).sum())

    def gov(rs):
        cnt: dict[str, int] = {}
        for r in rs:
            cnt[r["first_failure"]] = cnt.get(r["first_failure"], 0) + 1
        return max(cnt, key=cnt.get) if cnt else ""

    seeds = f"B stronger with {wins} of {n} seeds"
    if math.isnan(ciF[0]):
        verdict = (f"B differs from A by {dF:+.1%} in F_u with one seed: run n >= 3 to know "
                   "whether that is more than the scatter")
    elif sig:
        verdict = (f"B is {'stronger' if dF > 0 else 'weaker'} than A: F_u {dF:+.1%} (90 % "
                   f"interval {ciF[0]:+.1%} to {ciF[1]:+.1%}; {seeds}); efficiency "
                   f"{dE:+.1%}")
    else:
        verdict = (f"F_u changes by {dF:+.1%} ({seeds}), but the 90 % interval "
                   f"{ciF[0]:+.1%} to {ciF[1]:+.1%} includes 0: not proven with {n} seeds; "
                   "run more seeds if the difference matters")
    runs = [[DesignRun(seed=r["seed"], Fu_kgf=r["Fu_kgf"], mass_kg=r["mass_kg"],
                       efficiency_kgf_per_kg=r["efficiency_kgf_per_kg"],
                       first_failure=r["first_failure"], trustworthy=r["trustworthy"])
             for r in rs] for rs in (ra, rb)]
    return DesignComparison(
        a=name_a, b=name_b, n=n, fidelity=fidelity, runs_a=runs[0], runs_b=runs[1],
        Fu_mean_a_kgf=round(float(np.mean([r["Fu_kgf"] for r in ra])), 2),
        Fu_mean_b_kgf=round(float(np.mean([r["Fu_kgf"] for r in rb])), 2),
        Fu_change=round(dF, 4), Fu_change_ci90=ciF, efficiency_change=round(dE, 4),
        efficiency_change_ci90=ciE, significant=sig, verdict=verdict,
        governing_a=gov(ra), governing_b=gov(rb),
        untrustworthy_runs=sum(1 for r in ra + rb if not r["trustworthy"]))


SENSITIVITY = {  # key: (meaning, how to measure)
    "material.E_MPa": ("stiffness of a stick along the grain",
                       "3-point bending of single sticks (stick_bending_template.csv)"),
    "material.f_t_MPa": ("tension strength of a stick",
                         "pull single sticks to failure, or bend them (MOR)"),
    "material.f_c_MPa": ("crushing strength of a stick along the grain",
                         "short glued stack in compression"),
    "material.f_v_MPa": ("shear strength of a stick", "short-span bending of a stick stack"),
    "material.glue.tau_g_MPa": ("glue joint strength",
                                "single-lap joints pulled apart (glue_joint_template.csv)"),
    "glue.G_IIc_N_per_mm": ("toughness of a glue line in shear",
                            "the area under the force-slip curve of a lap test"),
    "wood.f_c90_MPa": ("crushing strength across the grain", "press a block of sticks flat"),
    "wood.G_f_N_per_mm": ("fracture energy of a stick in tension",
                          "area under the load-extension curve of a tension test"),
}


def sensitivity(cad: str | Path, overrides=None, material=None, sim_params=None,
                ruleset=None, params: list[str] | None = None, rel: float = 0.2,
                seed: int | None = 0, fidelity: str = "quick", processes: int | None = None,
                progress=None, units=None, up: str = "y", span: str = "x"):
    """Crush the bridge with each material value lowered and raised by ``rel`` (same
    seed) and rank the values by how much they move F_u: the measurements that would
    narrow the prediction most."""
    from bridgesim.sim.report import SensitivityItem, SensitivityReport

    if not 0 < rel < 0.9:
        raise ValueError("rel must be between 0 and 0.9")
    keys = list(params or SENSITIVITY)
    _, _, _, ov_raw, mat, sp, _ = _inputs(cad, overrides, material, sim_params, ruleset,
                                          units, up, span)
    props = mat.props()
    base_vals, sources = {}, {}
    for k in keys:
        if k.startswith("material."):
            kk = k.split(".", 1)[1]
            if kk not in props:
                raise ValueError(f"unknown parameter {k!r}")
            base_vals[k], sources[k] = float(props[kk].value), props[kk].source
        else:
            if k not in sp.values:
                raise ValueError(f"unknown parameter {k!r}")
            base_vals[k], sources[k] = float(sp[k]), sp.sources.get(k, "assumed")

    def with_param(k, v):
        d = json.loads(json.dumps(ov_raw or {}))
        d.setdefault("params", {})
        d["params"] = dict(d["params"] or {})
        if k is not None:
            d["params"][k] = v
        return d

    args = [("base", (str(cad), with_param(None, 0), material, sim_params, ruleset, seed,
                      None, fidelity, None, units, up, span))]
    for k in keys:
        for tag, f in (("low", 1 - rel), ("high", 1 + rel)):
            args.append((f"{k}|{tag}", (str(cad), with_param(k, base_vals[k] * f), material,
                                        sim_params, ruleset, seed, None, fidelity, None,
                                        units, up, span)))
    res = {r["tag"]: r for r in _pool_runs(args, len(args), processes, progress, "crush")}
    F0 = res["base"]["Fu_kgf"]
    items = []
    for k in keys:
        lo, hi = res[f"{k}|low"]["Fu_kgf"], res[f"{k}|high"]["Fu_kgf"]
        meaning, how = SENSITIVITY.get(k, (k, "see materials/simulation.yaml"))
        items.append(SensitivityItem(
            param=k, meaning=meaning, how_to_measure=how, source=sources[k],
            base=base_vals[k], Fu_low_kgf=round(lo, 2), Fu_high_kgf=round(hi, 2),
            elasticity=round((hi - lo) / (2 * rel * max(F0, 1e-9)), 3)))
    items.sort(key=lambda it: -abs(it.elasticity))
    first = [f"{it.meaning} ({it.param}, elasticity {it.elasticity:+.2f}): {it.how_to_measure}"
             for it in items if it.source != "measured" and abs(it.elasticity) >= 0.1][:3]
    return SensitivityReport(
        file=Path(cad).name, fidelity=fidelity, seed=seed, rel=rel,
        Fu_base_kgf=round(F0, 2), items=items, measure_first=first,
        note="One seed: a value that changes which part fails first can move F_u by more "
             "than its own effect; repeat with another --seed if a ranking matters.")


def compare_designs(cad_a: str | Path, cad_b: str | Path, n: int = 3, overrides_a=None,
                    overrides_b=None, material=None, sim_params=None, ruleset=None,
                    fidelity: str = "quick", processes: int | None = None,
                    out_dir: str | Path | None = None, progress=None, units=None,
                    up: str = "y", span: str = "x"):
    """Crush design A and design B with the same seeds 1..n and report whether B is
    really stronger (paired differences with a 90 % confidence interval), with the mass,
    the §12.6 efficiency and what fails first in each."""
    if n < 1:
        raise ValueError("n must be at least 1")
    for c in (cad_a, cad_b):
        if not Path(c).exists():
            raise FileNotFoundError(f"no such file: {c}")
    # check both inputs before spending minutes on them
    _inputs(cad_a, overrides_a, material, sim_params, ruleset, units, up, span)
    _inputs(cad_b, overrides_b, material, sim_params, ruleset, units, up, span)
    args = []
    for tag, cad, ov in (("A", cad_a, overrides_a), ("B", cad_b, overrides_b)):
        for sd in range(1, n + 1):
            args.append((tag, (str(cad), ov, material, sim_params, ruleset, sd, None,
                               fidelity, str(Path(out_dir) / f"{tag}_seed_{sd}")
                               if out_dir else None, units, up, span)))
    res = _pool_runs(args, 2 * n, processes, progress, "crush")
    rep = summarize_pairs([r for r in res if r["tag"] == "A"],
                          [r for r in res if r["tag"] == "B"],
                          Path(cad_a).name + (" (overrides)" if overrides_a else ""),
                          Path(cad_b).name + (" (overrides)" if overrides_b else ""),
                          fidelity)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "comparison.json").write_text(rep.model_dump_json(indent=2),
                                                      encoding="utf-8")
    return rep


def dump(model_obj) -> str:
    return json.dumps(model_obj.model_dump(mode="json"), indent=2)
