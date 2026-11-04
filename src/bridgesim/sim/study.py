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
import fnmatch
import hashlib
import json
import math
import os
import shlex
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
    NextStep,
    Quality,
    Question,
    RealityGap,
    RuleLine,
    RulesSummary,
    ScreenLimit,
    ScreenReport,
)
from bridgesim.yamlio import load_yaml

MASS_LIMIT_KG = 6.0


# ----------------------------------------------------------------------------- inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_overrides(overrides: str | Path | dict | None) -> tuple[Overrides, dict]:
    """Overrides from a dict (already parsed: MCP, the app) or from a YAML file the user
    names (the CLI's -o). Only a .yaml/.yml file is read, and its content must be a
    mapping; it is never echoed back in an error."""
    if overrides is None:
        return Overrides(), {}
    if isinstance(overrides, dict):
        d = overrides
    elif isinstance(overrides, (str, Path)):
        path = Path(overrides)
        if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
            raise ValueError(f"overrides file {path.name!r}: give an existing .yaml or .yml "
                             "file")
        d = load_yaml(path.read_text(encoding="utf-8")) or {}
        if not isinstance(d, dict):
            raise ValueError(f"overrides file {path.name!r} must hold a YAML mapping "
                             "(bodies:, glue:, params:, acknowledge:)")
    else:
        raise ValueError("overrides must be a YAML mapping (bodies:, glue:, params:, "
                         f"acknowledge:), not a {type(overrides).__name__}")
    return Overrides.from_dict(d), d


def _from_mapping(doc) -> tuple[Overrides, dict]:
    """Overrides already parsed (typed in the app, sent through MCP): a mapping, never a
    file name (this path never opens a file)."""
    if not isinstance(doc, dict):
        raise ValueError("overrides must be a YAML mapping (bodies:, glue:, params:, "
                         f"acknowledge:), not a {type(doc).__name__}")
    return Overrides.from_dict(doc), doc


def _inputs(cad: str | Path, overrides, material, sim_params, ruleset, units=None,
            up: str = "y", span: str = "x", doc: dict | None = None):
    path = Path(cad)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    bodies = load_cad(path, units=units, up=up, span=span)
    ov, ov_raw = _from_mapping(doc) if doc is not None else load_overrides(overrides)
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


def _material_with(mat: Material, changes: dict[str, float],
                   notes: dict[str, str] | None = None, source: str = "measured") -> Material:
    d = mat.model_dump()
    notes = notes or {}
    for k, v in changes.items():
        entry = {"value": float(v), "source": source, "note": notes.get(k, "")}
        if k.startswith("glue."):
            d["glue"][k[5:]] = entry
        else:
            d[k] = entry
    return Material.model_validate(d)


def _rules(bodies: list[CadBody], mat: Material, rs: RuleSet, model: Model | None
           ) -> RulesSummary:
    meas = measure_cad(bodies, mat.density_kg_m3.value, mat.glue.mass_fraction.value,
                       rs.constants,
                       deck_y=model.params.get("deck_y") if model else None,
                       deck_z=model.deck_z if model else None)
    dm = float(model.params.get("mass_change_kg", 0.0)) if model else 0.0
    if dm and "mass_kg" in meas.values:  # a section_scale what-if adds or removes wood
        meas.values["mass_kg"] = float(meas.values["mass_kg"]) + dm
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
        # the load goes in at one end and out at the other: the weaker end governs
        ends = [0.0, 0.0]
        ax = np.asarray(info.axis if info.axis is not None else (1.0, 0.0, 0.0), float)
        for p in by_body.get(bi, []):
            if p.kind == "glue":
                s_ = float((np.asarray(p.centroid) - np.asarray(info.centroid_mm)) @ ax)
                ends[int(s_ >= 0)] += p.area_mm2
        glue = min(e for e in ends if e > 0) * tau if any(ends) else 0.0
        if glue and glue < 0.5 * cap_member and _group(info.name) not in weak_seen:
            weak_seen.add(_group(info.name))
            qs.append(Question(
                id=f"weak_joint:{info.name}", topic="design", body=info.name,
                question=f"The glued area at the weaker end of '{info.name}' (and its "
                         f"copies) carries about {glue / KGF:.0f} kgf "
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
            assumed=", ".join(f"{k} = {mat.props()[k].value:g}" for k in
                              ("E_MPa", "f_t_MPa", "f_c_MPa", "glue.tau_g_MPa")
                              if k in mat.assumed_keys()) + " (placeholders)",
            answer_with={"cli": "bridgesim calibrate validation/physical --out "
                                "my_material.yaml, then crush with --material "
                                "my_material.yaml"},
            severity="important"))
    return qs


# ----------------------------------------------------------------------------- inspect


def _open_questions(qs: list[Question], ov: Overrides, model: Model
                    ) -> tuple[list[Question], list[str]]:
    """Questions still open, and the ids the overrides acknowledged (``acknowledge:``).
    Acknowledge patterns that match no question become warnings."""
    open_, acked = [], []
    for q in qs:
        (acked if ov.acknowledged(q.id) else open_).append(q)
    ids = [q.id for q in qs]
    for a in ov.acknowledge:
        if not any(i == a or fnmatch.fnmatchcase(i, a) for i in ids):
            model.notes.append(f"overrides: acknowledge {a!r} matches no question (ids: "
                               f"{', '.join(sorted(ids)[:8])}{' ...' if len(ids) > 8 else ''})")
    return open_, [q.id for q in acked]


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
            span: str = "x", strict: bool = False,
            overrides_data: dict | None = None) -> InspectReport:
    """Read the CAD file and report what bridgesim understood, the rule check and the
    questions to ask the designer. No simulation is run (a few seconds). ``overrides``
    is a mapping or the user's YAML file; ``overrides_data`` a mapping typed by someone
    else (the app, an MCP client), which is never taken as a file name."""
    path, bodies, ov, ov_raw, mat, sp, rs = _inputs(cad, overrides, material, sim_params,
                                                     ruleset, units, up, span,
                                                     doc=overrides_data)
    model = _build(bodies, mat, sp, ov, seed, 1.0, strict)
    rules = _rules(bodies, mat, rs, model)
    qs, acked = _open_questions([Question(**q) for q in model.questions]
                                + design_questions(bodies, model, mat, sp, rules), ov, model)
    rep = InspectReport(
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
        rules=rules, questions=qs, acknowledged_questions=acked,
        assumed_parameters=_assumed(mat, sp),
        model={"nodes": int(model.n_nodes), "beam_elements": int(len(model.b_i)),
               "shell_elements": int(len(model.s_nodes)), "glue_points": int(len(model.c_a)),
               "glued_patches": len(model.patches), "floss_cables": int(len(model.k_a)),
               "platform_contact_points": int(len(model.p_att)),
               "plate_contact_points": int(len(model.q_att))},
        warnings=list(model.notes))
    rep.next_steps = _next_inspect(rep, path, _ov_ref(overrides, ov_raw))
    return rep


def screen(cad: str | Path, overrides=None, material=None, sim_params=None, ruleset=None,
           load_kgf: float = 100.0, units=None, up: str = "y", span: str = "x",
           strict: bool = False, overrides_data: dict | None = None,
           n_limits: int = 25) -> ScreenReport:
    """Linear pre-screen (seconds): the perfect bridge's elastic response to ``load_kgf``
    on the plate, and the load at which each joint, member and plate would reach its
    limit if it stayed linear (:mod:`bridgesim.sim.screen`). Ranks the weak spots and
    compares design variants (the same overrides as crush, what-ifs included); F_u
    needs crush."""
    import time

    from bridgesim.sim import screen as scr

    t0 = time.time()
    path, bodies, ov, ov_raw, mat, sp, rs = _inputs(cad, overrides, material, sim_params,
                                                     ruleset, units, up, span,
                                                     doc=overrides_data)
    model = _build(bodies, mat, sp, ov, None, 1.0, strict)
    rules = _rules(bodies, mat, rs, model)
    F = load_kgf * KGF
    sol = scr.solve(model, F)
    lims = scr.utilisations(model, sol, F,
                            peak_over_avg=sp.values.get("glue.tau_peak_over_avg", 1.155))
    mass = rules.measurements.get("mass_kg", model.bridge_mass_kg)
    first = lims[0] if lims else None
    rep = ScreenReport(
        file=path.name, sha256=_sha(path), overrides=ov_raw, load_kgf=load_kgf,
        stiffness_kgf_per_mm=round(load_kgf / max(sol["plate_dy"], 1e-12), 2),
        first_limit_kgf=round(first.load_kgf, 1) if first else 0.0,
        first_limit=f"{first.mode}: {first.part}" if first else "none",
        limits=[ScreenLimit(part=lm.part, mode=lm.mode,
                            utilisation=round(lm.utilisation, 4),
                            load_kgf=round(lm.load_kgf, 1), where_mm=list(lm.where_mm))
                for lm in lims[:n_limits]],
        n_limits=len(lims), mass_kg=round(mass, 3),
        wall_s=round(time.time() - t0, 1),
        note="Linear and elastic (contacts compression only): no damage, no load "
             "redistribution, no imperfections, small displacements; the perfect bridge. "
             "first_limit_kgf is where damage starts, often far below F_u: a joint that "
             "starts to give way early can be harmless when the load finds another path, "
             "which only crush follows. Use the screen for the bridge's stiffness and its "
             "most loaded joints and members, and to see in seconds how a what-if moves "
             "them; judge F_u and design changes with crush and ab.",
        warnings=list(model.notes))
    f, ov_arg = shlex.quote(str(path.resolve())), _ov_arg(_ov_ref(overrides, ov_raw))
    weak = list(dict.fromkeys(lm.part for lm in lims[:5]))
    rep.next_steps = [
        NextStep(action="crush", why="the screen ranks the weak spots; the crush follows "
                                     "the bridge to its collapse and gives F_u",
                 parts=weak, cli=f"bridgesim crush {f}{ov_arg} --fidelity quick "
                                 "--stop-at-failure --out results --json --brief"),
        NextStep(action="strengthen", parts=weak[:3],
                 why="the most loaded parts, where damage starts: whether strengthening "
                     "them raises F_u depends on what governs the collapse; crush shows "
                     "it, then test a what-if (overrides: glue_factor, section_scale) "
                     "with ab")]
    return rep


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
          span: str = "x", strict: bool = False, rate_check: bool = False,
          mesh_scale: float | None = None,
          overrides_data: dict | None = None,
          stop_at_failure: bool = False) -> tuple[CrushReport, CrushResult]:
    """Simulate the §12.5 crushing test of a CAD bridge. Writes ``crush.json``,
    ``curve.csv``, ``events.csv`` and ``replay.html`` (interactive 3D replay) to
    ``out_dir`` if given. ``rate_check`` crushes the bridge a second time at half the
    jack speed and reports how much F_u changes (``quality.rate_check``). ``mesh_scale``
    (element size relative to standard) overrides the fidelity's mesh, for convergence
    studies. ``stop_at_failure``: stop 2 mm of jack travel after the bridge is judged
    failed (F_u and the failures are known; about a third faster; the replay ends
    there) instead of following the collapse for 15 mm."""
    from bridgesim.sim.crush import FIDELITY

    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {', '.join(FIDELITY)}")
    scale, speed = FIDELITY[fidelity]
    if mesh_scale is not None:
        if not 0.2 <= mesh_scale <= 4:
            raise ValueError("mesh_scale must be between 0.2 and 4")
        scale = float(mesh_scale)
    path, bodies, ov, ov_raw, mat, sp, rs = _inputs(cad, overrides, material, sim_params,
                                                     ruleset, units, up, span,
                                                     doc=overrides_data)
    if progress is not None:
        progress(0.0, f"read {len(bodies)} bodies; building the structural model")
    model = _build(bodies, mat, sp, ov, seed, scale, strict)
    if scale < 0.75:
        model.notes.append(f"mesh_scale {scale:g} is finer than the fine fidelity (0.75): "
                           "plate elements may be thinner than their plates, and a crack "
                           "starts earlier at a stress concentration the finer the mesh "
                           "(local damage; ASSUMPTIONS-SIM.md)")
    rules = _rules(bodies, mat, rs, model)
    # the fine mesh needs a smaller time step, or too much mass scaling would be added
    opt = options or CrushOptions(speed_mm_s=speed,
                                  dt_target=2.2e-6 if fidelity == "fine" else 3e-6)
    if stop_at_failure:
        import dataclasses

        opt = dataclasses.replace(opt, after_failure_mm=min(opt.after_failure_mm, 2.0))
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
    qs, acked = _open_questions([Question(**q) for q in model.questions]
                                + design_questions(bodies, model, mat, sp, rules), ov, model)
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
        rules=rules, questions=qs, acknowledged_questions=acked,
        assumed_parameters=_assumed(mat, sp),
        fidelity=fidelity, Fu_first_joint_failure_kgf=_first_joint(res),
        warnings=list(model.notes))
    rep.next_steps = _next_crush(rep, path, _ov_ref(overrides, ov_raw),
                                 out_dir)
    if out_dir:
        out = Path(out_dir).resolve()
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
        wi = next((s.overrides for s in rep.next_steps
                   if s.action == "strengthen" and s.overrides), None)
        if wi:  # the what-if of next_steps, ready for ab --overrides-b
            import yaml

            (out / "whatif.yaml").write_text(
                "# your overrides plus the what-if of next_steps (strengthen)\n"
                + yaml.safe_dump(merge_whatif(ov_raw, wi), sort_keys=False,
                                 allow_unicode=True), encoding="utf-8")
            rep.artifacts["whatif_yaml"] = str(out / "whatif.yaml")
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


def _ask_steps(qs: list[Question]) -> list[NextStep]:
    out = []
    for sev, what in (("important", "the answers change the model or the result"),
                      ("check", "worth confirming with the designer")):
        sel = [q for q in qs if q.severity == sev and not q.id.startswith("rule:")
               and q.id != "material:measure"]
        if sel:
            out.append(NextStep(
                action="ask_designer", ids=[q.id for q in sel],
                why=f"{len(sel)} {sev} question(s) open ({what}): answer them in the "
                    "overrides (answer_with), or acknowledge the ones that are as drawn"))
    return out


def _ov_ref(overrides, ov_raw: dict):
    """What next_steps say about the overrides: the file the CLI user named, True when
    they came inline (typed in the app or sent over MCP: never their content, which is
    not a file name), None without overrides."""
    if not ov_raw:
        return None
    return overrides if isinstance(overrides, (str, Path)) and str(overrides) else True


def _ov_arg(overrides) -> str:
    """The overrides as a command-line option: the file the caller gave, a placeholder
    when they came as a dict, nothing without overrides."""
    if isinstance(overrides, (str, Path)) and str(overrides):
        return " -o " + shlex.quote(os.path.abspath(overrides))
    return " -o overrides.yaml" if overrides else ""


def merge_whatif(user: dict | None, whatif: dict) -> dict:
    """The user's overrides with a what-if on top: what-if bodies and joints are added,
    and a part or joint the user already scaled is scaled by both (factors multiply)."""
    import copy

    out = copy.deepcopy(user or {})
    for k, v in (whatif.get("bodies") or {}).items():
        cur = dict((out.setdefault("bodies", {}) or {}).get(k) or {})
        for f, x in v.items():
            cur[f] = round(float(cur.get(f, 1.0)) * float(x), 6) if f in (
                "section_scale", "glue_factor", "strength_factor") else x
        out["bodies"][k] = cur
    for k, x in (whatif.get("glue") or {}).items():
        g = out.setdefault("glue", {}) or {}
        g[k] = round(float(g.get(k, 1.0)) * float(x), 6)
        out["glue"] = g
    return out


def _rule_step(rules: RulesSummary, qs: list[Question],
               acked: list[str] = ()) -> list[NextStep]:
    bad = [ln for ln in rules.lines if (ln.passed is False or ln.bans)
           and f"rule:{ln.section}" not in acked]
    if not bad:
        return []
    def say(ln: RuleLine) -> str:
        out = f"§{ln.section} {ln.title} fails: {ln.measured}"
        if ln.bans:
            out += f" (bans: {', '.join(ln.bans)})"
        elif ln.penalty:
            out += f" (penalty {ln.penalty:g})"
        return out + (f"; {ln.cause}" if ln.cause else "")

    return [NextStep(action="fix_cad", ids=[q.id for q in qs if q.id.startswith("rule:")],
                     why="; ".join(say(ln) for ln in bad))]


def _measure_step(assumed: list[str], f: str, ov: str) -> list[NextStep]:
    if not assumed:
        return []
    return [NextStep(action="measure",
                     why=f"{len(assumed)} material values are placeholders, so the absolute "
                         "F_u is uncertain (design comparisons much less so): sensitivity "
                         "ranks what to measure first; then coupon tests, lab and "
                         "calibrate", cli=f"bridgesim sensitivity {f}{ov} --json")]


def _next_inspect(rep: InspectReport, path: Path, overrides) -> list[NextStep]:
    f, ov = shlex.quote(str(path.resolve())), _ov_arg(overrides)
    steps = _rule_step(rep.rules, rep.questions, rep.acknowledged_questions) \
        + _ask_steps(rep.questions)
    steps.append(NextStep(action="screen", why="in seconds, linear: the bridge's stiffness "
                          "and its most loaded joints and members (where damage starts)",
                          cli=f"bridgesim screen {f}{ov} --json"))
    steps.append(NextStep(action="crush", why="simulate the §12.5 crushing test (standard "
                          "fidelity, ~5 min; --fidelity quick ~2 min to screen variants)",
                          cli=f"bridgesim crush {f}{ov} --out results --json --brief"))
    return steps + _measure_step(rep.assumed_parameters, f, ov)


def _copies(name: str) -> str:
    """A glob matching a part and its copies on the other sides ('Top Chord - Far' ->
    'Top Chord*'): a design change is made on every copy, not on the one this seed broke."""
    import glob as _glob

    g = _group(name)
    # only names with a side or copy suffix have copies ("Body 4" must not match "Body 41")
    return _glob.escape(name) if g == name else _glob.escape(g) + "*"


def _whatif(gov: list[EventOut]) -> dict:
    """Overrides that strengthen the parts that broke at the peak, on every copy of them:
    a joint's glue (a bigger glued area or a gusset), a member's section or a plate's
    thickness."""
    bodies: dict[str, dict] = {}
    glue: dict[str, float] = {}
    for e in gov:
        if e.kind in ("joint_failure", "joint_cracking") and len(e.parts) == 2:
            glue[f"{_copies(e.parts[0])} | {_copies(e.parts[1])}"] = 1.5
        elif e.kind in ("member_shear", "member_fracture", "plate_crack"):
            for p in e.parts or [e.what]:
                bodies[_copies(p)] = {"section_scale": 1.25}
    return {k: v for k, v in (("bodies", bodies), ("glue", glue)) if v}


def _next_crush(rep: CrushReport, path: Path, overrides, out_dir) -> list[NextStep]:
    f, ov = shlex.quote(str(path.resolve())), _ov_arg(overrides)
    out = shlex.quote(str(Path(out_dir).resolve())) if out_dir else "results"
    # the what-if (with the user's overrides) is written there by crush(out_dir=...)
    wy = (shlex.quote(str(Path(out_dir).resolve() / "whatif.yaml")) if out_dir
          else "whatif.yaml")
    steps: list[NextStep] = []
    q = rep.quality
    if not q.trustworthy:
        steps.append(NextStep(
            action="rerun", why="not trustworthy: " + "; ".join(q.reasons),
            cli=f"bridgesim crush {f}{ov} --fidelity {rep.fidelity} --rate-check "
                f"--out {out}_check --json --brief"))
    broke = [e for e in rep.failures if e.kind != "joint_cracking"]
    gov = [e for e in broke if e.load_kgf >= 0.9 * rep.Fu_kgf][:3] or broke[:3]
    if rep.failure_mode == "instability":
        steps.append(NextStep(action="strengthen", why="the bridge collapses under its own "
                              "weight: " + rep.failure_detail,
                              parts=sorted({p for e in broke[:3] for p in e.parts})))
    elif not gov and rep.failure_mode == "collapse" and "glue of" in rep.failure_detail:
        at = {}  # each joint whose glue was giving way at the peak, and where it is
        for e in rep.events:
            if e.kind == "glue_damage" and e.load_kgf >= 0.9 * rep.Fu_kgf:
                at.setdefault(e.what, e.where_mm)
        soft = list(at)[:3]
        wi = {"glue": {f"{_copies(a)} | {_copies(b)}": 1.5
                       for a, b in (w.split(" | ", 1) for w in soft if " | " in w)}}
        ova = ov.replace(" -o ", " --overrides-a ")
        steps.append(NextStep(
            action="strengthen", parts=sorted({p for w in soft for p in w.split(" | ")}),
            where_mm=[at[w] for w in soft if at[w]],
            why=rep.failure_detail + ". Those joints gave way without a clean break: give "
                "them more glued area or a gusset; try it first as the what-if in "
                "`overrides` (with your own overrides: whatif.yaml in the output folder) "
                "with ab on the same seeds",
            overrides=wi if wi["glue"] else {},
            cli=(f"bridgesim ab {f} {f}{ova} --overrides-b {wy} -n 3 --json"
                 if wi["glue"] else f"bridgesim ab {f} <changed design> -n 3 --json")))
    elif not gov and rep.failure_mode in ("collapse", "deflection", "contact"):
        what = {"collapse": rep.failure_detail + ". Brace the members in compression and "
                            "against sway (the replay shows which one buckled)",
                "deflection": f"it deflected more than 50 mm (§12.5) at {rep.Fu_kgf:.0f} kgf "
                              "without breaking: make it stiffer (a deeper truss, stiffer "
                              "chords)",
                "contact": "it touched the platform (§12.5) before anything broke: "
                           + rep.failure_detail}[rep.failure_mode]
        steps.append(NextStep(action="strengthen", why=what + "; then check the change "
                              "against this design with ab (same seeds)",
                              cli=f"bridgesim ab {f} <changed design> -n 3 --json"))
    elif gov:
        parts: list[str] = []
        for e in gov:
            parts += [p for p in (e.parts or [e.what]) if p not in parts]
        wi = _whatif(gov)
        ova = ov.replace(" -o ", " --overrides-a ")
        steps.append(NextStep(
            action="strengthen", parts=parts,
            where_mm=[e.where_mm for e in gov if e.where_mm],
            why="; ".join(f"{e.what}: {e.kind.replace('_', ' ')} at {e.load_kgf:.0f} kgf "
                          f"({e.load_kgf / max(rep.Fu_kgf, 1e-9):.0%} of F_u)" for e in gov)
            + ". The bridge failed there: strengthen it (glued area, a gusset, a deeper "
              "member). Try it first as a what-if (`overrides`: 1.5x the glue of a joint "
              "that separated, 1.25x the section of a part that broke, on every copy of "
              "it; with your own overrides: whatif.yaml in the output folder) with ab on "
              "the same seeds; if it helps, make the change in the CAD and compare the two "
              "files",
            overrides=wi,
            cli=(f"bridgesim ab {f} {f}{ova} --overrides-b {wy} -n 3 --json" if wi
                 else f"bridgesim ab {f} <changed design> -n 3 --json")))
        close = {e.what for e in gov if e.load_kgf <= 1.03 * gov[0].load_kgf}
        if rep.fidelity == "quick":
            steps.append(NextStep(
                action="confirm", why="quick fidelity is for ranking designs: its F_u is "
                "~3.5 % high and the part that breaks first can differ at standard",
                cli=f"bridgesim crush {f}{ov} --fidelity standard --out {out}_standard "
                    "--json --brief"))
        elif rep.fidelity == "standard" and len(close) > 1:
            steps.append(NextStep(
                action="confirm", parts=sorted(close),
                why="several parts fail within 3 % of each other: which goes first can "
                    "change with the mesh; check with the fine mesh before redesigning",
                cli=f"bridgesim crush {f}{ov} --fidelity fine --out {out}_fine "
                    "--json --brief"))
    steps += _rule_step(rep.rules, rep.questions, rep.acknowledged_questions) \
        + _ask_steps(rep.questions)
    steps += _measure_step(rep.assumed_parameters, f, ov)
    if rep.Fu_kgf > 0:
        steps.append(NextStep(
            action="predict", why="one seed is one possible build: the spread over "
                                  "imperfections and scatter gives recommended_prediction_kgf "
                                  "for the Load Prediction Form",
            cli=f"bridgesim montecarlo {f}{ov} -n 8 --json"))
    steps.append(NextStep(
        action="compare", why="after a real crush test (a prototype or the competition): "
                              "the reality gap and what to measure next",
        cli=f"bridgesim compare {out}/crush.json my_test.csv --json"))
    return steps


def select(rep: InspectReport, body: str | list[str] | None = None,
           joint: str | list[str] | None = None) -> InspectReport:
    """The inspect report with only the bodies whose name matches ``body`` and the joints
    one of whose parts (or whose 'A | B' name) matches ``joint`` (glob patterns, case
    ignored), and only the questions about them: to look up a few parts of a big bridge."""
    def pats(p):
        return [x.lower() for x in ([p] if isinstance(p, str) else list(p or []))]

    bp, jp = pats(body), pats(joint)

    def hit(name: str, ps: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(name.lower(), p) for p in ps)

    upd: dict[str, Any] = {}
    names: set[str] = set()
    if bp:
        upd["bodies"] = [b for b in rep.bodies if hit(b.name, bp)]
        names |= {b.name for b in upd["bodies"]}
    else:
        upd["bodies"] = [] if jp else rep.bodies
    if jp:
        upd["joints"] = [j for j in rep.joints if hit(f"{j.a} | {j.b}", jp)
                         or hit(j.a, jp) or hit(j.b, jp)]
        names |= {p for j in upd["joints"] for p in (j.a, j.b)}
    else:
        upd["joints"] = [] if bp else rep.joints
    if bp or jp:
        upd["questions"] = [q for q in rep.questions
                            if any(n in q.id or n in q.question for n in names)]
    return rep.model_copy(update=upd)


def inspect_brief(rep: InspectReport) -> dict:
    """The inspect report without the per-body and per-joint lists: counts of each kind,
    the questions, failed rules and warnings (a few kB). Bodies and joints: full report."""
    d = rep.model_dump(mode="json", exclude={"bodies", "joints"})
    d["kind"] = "inspect_brief"
    kinds: dict[str, int] = {}
    for b in rep.bodies:
        kinds[b.kind] = kinds.get(b.kind, 0) + 1
    d["bodies_by_kind"] = kinds
    d["n_joints"] = len(rep.joints)
    r = d.get("rules") or {}
    if "lines" in r:
        r["lines"] = [ln for ln in r["lines"] if ln.get("passed") is False
                      or ln.get("penalty") or ln.get("bans")]
    order = {"important": 0, "check": 1, "info": 2}
    d["questions"] = sorted(d["questions"], key=lambda q: order.get(q["severity"], 3))
    return d


def brief(rep: CrushReport, n_failures: int = 10) -> dict:
    """The crush report cut down to what an agent needs to decide its next step, in about
    10 kB: no curve and event log (curve.csv, events.csv), the first ``n_failures``
    failures (without their time, newtons and parts list), the important questions in full and the
    others counted (their ids are in the ask_designer step; inspect has their text and
    answer_with), the next steps without empty fields. crush.json has everything."""
    d = rep.model_dump(mode="json", exclude={"curve", "events", "failure_sequence",
                                             "assumed_parameters"})
    d["kind"] = "crush_brief"
    d["n_failures_total"] = len(d["failures"])
    # without time, newtons and parts (`what` names them: 'A | B' for a joint)
    d["failures"] = [{k: v for k, v in f.items() if k not in ("t_s", "load_N", "parts")}
                     for f in d["failures"][:n_failures]]
    d["n_assumed_parameters"] = len(rep.assumed_parameters)
    r = d["rules"]
    r["lines"] = [ln for ln in r.get("lines", []) if ln.get("passed") is False
                  or ln.get("penalty") or ln.get("bans")]
    d["n_questions"] = {sev: sum(q["severity"] == sev for q in d["questions"])
                        for sev in ("important", "check", "info")}
    d["questions"] = [{"id": q["id"], "severity": q["severity"], "question": q["question"],
                       "assumed": q.get("assumed"), "answer_with": q.get("answer_with")}
                      for q in d["questions"] if q["severity"] == "important"]
    d["next_steps"] = [{k: v for k, v in st.items() if v not in (None, [], {}, "")}
                       for st in d["next_steps"]]
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
                   speed_mm_s=getattr(res, "speed_mm_s", 0.0), rate_check=rc,
                   holds=getattr(res, "holds", 0),
                   hold_time_s=round(getattr(res, "hold_time_s", 0.0), 4))


# ----------------------------------------------------------------------------- compare


def _secant(load, x, pk) -> float:
    """Secant stiffness on the rising branch between the first points where the load
    reaches 10 % and 50 % of the peak (interpolated: the sampling does not matter)."""
    load, x = np.asarray(load, float), np.asarray(x, float)
    if not len(load):
        return math.nan
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


def _compare_range(mc: MonteCarloReport, test_csv: str | Path) -> RealityGap:
    """A physical test against the spread of a montecarlo run: is the measured F_u
    inside the predicted range, where, and how good the Load Prediction Form value was."""
    t = _read_test_csv(test_csv)
    peak_t = float(t["peak"])
    if not math.isfinite(peak_t) or peak_t <= 0:
        raise ValueError(f"{test_csv}: no load_kgf value")
    F = np.sort(np.asarray(mc.Fu_kgf, float))
    if not len(F):
        raise ValueError("the montecarlo report has no runs")
    rank = float((F < peak_t).mean())
    inside = bool(F[0] <= peak_t <= F[-1])
    err = (mc.p50_kgf - peak_t) / peak_t
    pred = abs(peak_t - mc.recommended_prediction_kgf) / peak_t
    metrics = [
        GapMetric(name="ultimate load vs the predicted range",
                  simulated=f"{F[0]:.0f}-{F[-1]:.0f} (median {mc.p50_kgf:.0f})",
                  measured=round(peak_t, 2), error=round(err, 4), unit="kgf",
                  meaning=f"median of {len(F)} runs vs the test; the test is "
                          f"{'inside' if inside else 'OUTSIDE'} the range of the runs, "
                          f"above {rank:.0%} of them"),
        GapMetric(name="§12.7 prediction", simulated=mc.recommended_prediction_kgf,
                  measured=round(peak_t, 2), error=round(pred, 4), unit="kgf",
                  meaning="recommended_prediction_kgf against the test (E of §12.7)")]
    score = 100 * max(0.0, 1 - abs(math.log(max(mc.p50_kgf, 1e-9) / peak_t)) / math.log(1.5))
    if inside:
        advice = [f"The test falls inside the predicted range ({rank:.0%} of the runs are "
                  "below it): the model is consistent with this test. Narrow the range by "
                  "measuring what `sensitivity` ranks first."]
    else:
        side = "too strong" if F[0] > peak_t else "too weak"
        advice = [f"The test is outside the range of {len(F)} runs: the model is {side} for "
                  "this bridge, more than the scatter explains. Compare its failure with "
                  "the runs' failure modes, then measure the material values that govern "
                  "it (coupon tests) or check the joints' real glued area (glue_factor)."]
        if mc.assumed_parameters and not mc.sampled_placeholders:
            advice.append("This range leaves out the placeholder material values: "
                          "montecarlo --sample-placeholders gives the range that includes "
                          "them.")
    return RealityGap(subject=f"{mc.file}: montecarlo ({len(F)} runs, {mc.fidelity}) vs "
                              f"{Path(test_csv).name}",
                      metrics=metrics, score_0_100=round(score, 1),
                      competition_prediction_error=round(pred, 4), advice=advice)


def compare(sim: CrushReport | MonteCarloReport | str | Path,
            test_csv: str | Path) -> RealityGap:
    """Reality gap between a simulated crush (report or crush.json) and a physical crush
    test (``validation/physical/crush_test_template.csv``): load_kgf against
    midspan_deflection_mm or jack_travel_mm, a row with event 'failure' and the broken part
    in notes. With the peak load alone (and the broken part), the ultimate load, the §12.7
    prediction error and the failure location are still compared. Given a montecarlo
    report (montecarlo.json) instead, the test is placed in the predicted range."""
    if isinstance(sim, MonteCarloReport):
        return _compare_range(sim, test_csv)
    if not isinstance(sim, CrushReport):
        raw = json.loads(Path(sim).read_text(encoding="utf-8"))
        if raw.get("kind") == "montecarlo":
            return _compare_range(MonteCarloReport.model_validate(raw), test_csv)
        sim = CrushReport.model_validate(raw)
    rep = sim
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

    slope = _secant

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
        # the simulation's first governing failure counts fully; a later one near the
        # peak (a close competitor that this seed broke second) half
        score = max((_part_match(obs, f.what) * (1.0 if i == 0 else 0.5)
                     for i, f in enumerate(gov)), default=0.0)
        metrics.append(GapMetric(
            name="failure location", simulated="; ".join(f.what for f in gov),
            measured=obs, error=1.0 - score,
            meaning="did the simulation break the same part, on the same side, first "
                    "(0: yes; 0.5: same part but on the other side, or the part the "
                    "simulation broke second near the peak; 1: no)"))
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
    tau_peak = mat.glue.tau_g_MPa.value * sp.values.get("glue.tau_peak_over_avg", 1.155)
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
                                                                 1.155)
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
              ) -> tuple[Material, list[str], dict[str, float]]:
    """Fit the material to the team's coupon tests so that the *simulated* coupons
    (``bridgesim lab``) reproduce them: E from the bending stiffness (the simulated
    coupon matches 48 E I / L^3), the strengths f_t and f_c (scaled together, keeping
    their ratio) so the simulated sticks break at the measured loads, and tau_g so the
    simulated lap joints of the measured overlaps break at the measured loads. With three
    or more specimens of a kind, their scatter gives the coefficients of variation of
    ``materials/simulation.yaml`` (scatter.*). Returns the new material, what was changed,
    and those simulation parameters (for an overrides file's ``params:``)."""
    mat = material if isinstance(material, Material) else load_material(
        material or "popsicle_birch")
    sp = sim_params if isinstance(sim_params, SimParams) else SimParams.load(sim_params)
    d = Path(csv_dir)
    log = []
    changes: dict[str, float] = {}
    assumed_changes: dict[str, float] = {}
    notes: dict[str, str] = {}
    params: dict[str, float] = {}
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
            notes["E_MPa"] = f"from {len(Es)} 3-point bending tests (calibrate)"
            log.append(f"E = {E:.0f} MPa from {len(Es)} bending tests")
            # G is not measured by a bending test: keep its ratio to E (still assumed)
            g_new = mat.G_MPa.value * E / mat.E_MPa.value
            assumed_changes["G_MPa"] = g_new
            notes["G_MPa"] = (f"assumed: scaled with the measured E (E/G = "
                              f"{mat.E_MPa.value / mat.G_MPa.value:.0f} kept)")
            if len(Es) >= 3:
                params["scatter.E_cov"] = float(np.std(Es, ddof=1) / np.mean(Es))
        if len(fails) >= 3:
            fl = [x[3] * x[0] / (x[1] * x[2] ** 2) for x in fails]  # ~ bending strength
            params["scatter.strength_cov"] = float(np.std(fl, ddof=1) / np.mean(fl))
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
            notes["f_b_MPa"] = f"from {len(fails)} bending failures (calibrate)"
            for kk in ("f_t_MPa", "f_c_MPa"):
                notes[kk] = ("fitted so the simulated sticks break at the measured bending "
                             "loads; the f_t / f_c ratio is still the placeholder's")
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
            ratio = sp.values.get("glue.tau_peak_over_avg", 1.155)
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
            notes["glue.tau_g_MPa"] = (f"fitted to {len(joints)} lap joints through the "
                                       "simulated lap test (calibrate)")
            log.append(f"tau_g = {tau:.2f} MPa from {len(joints)} joints (so the simulated "
                       "lap joints of the measured overlaps break at the measured loads)")
            if len(joints) >= 3:
                tj = [F_ / (ov * w) for ov, w, F_ in joints]
                params["scatter.glue_cov"] = float(np.std(tj, ddof=1) / np.mean(tj))
    _direct_coupons(d, mat, sp, changes, notes, params, log)
    if not changes and not params:
        if not d.is_dir():
            raise FileNotFoundError(f"no such folder: {d}")
        raise ValueError(
            f"no coupon test results in {d.name}: fill one or more of the templates of "
            "validation/physical/ (stick_bending, glue_joint, stick_tension, "
            "stick_compression, crush_across, end_grain, lap_slip: *_template.csv, a row "
            "per specimen) and run calibrate on that folder")
    for k, v in params.items():
        if k.startswith("scatter."):
            log.append(f"{k} = {v:.3f} from the scatter of the specimens")
    new = _material_with(mat, changes, notes)
    if assumed_changes:
        new = _material_with(new, assumed_changes, notes, source="assumed")
    new = new.model_copy(update={
        "name": (mat.name.replace(" (placeholder)", "") + " (calibrated)")[:200],
        "description": f"Calibrated by bridgesim from the coupon tests in {d.name}: "
                       + "; ".join(log)})
    return new, log, params


def _rows(f: Path) -> list[dict]:
    return list(csv.DictReader(f.read_text(encoding="utf-8").splitlines())) \
        if f.exists() else []


def _direct_coupons(d: Path, mat: Material, sp: SimParams, changes: dict, notes: dict,
                    params: dict, log: list) -> None:
    """Strength and glue values measured directly (no inverse fit): stick tension and
    short-stack compression (f_t, f_c, replacing the ratio-scaled pair from bending), a
    stick block pressed flat (f_c90), sticks glued end to end and pulled (the end-grain
    factor) and the load-slip record of a lap joint (glue shear stiffness and
    toughness)."""
    vals = [_float(r, "failure_load_N") / (_float(r, "width_mm") * _float(r, "thickness_mm"))
            for r in _rows(d / "stick_tension_template.csv")
            if _float(r, "failure_load_N") and _float(r, "width_mm") and _float(r, "thickness_mm")]
    if vals:
        changes["f_t_MPa"] = float(np.mean(vals))
        notes["f_t_MPa"] = f"measured: {len(vals)} stick tension tests (calibrate)"
        log.append(f"f_t = {changes['f_t_MPa']:.1f} MPa from {len(vals)} tension tests")
    comp = []
    for r in _rows(d / "stick_compression_template.csv"):
        F, w, t, L = (_float(r, k) for k in ("failure_load_N", "width_mm", "thickness_mm",
                                            "length_mm"))
        if F and w and t:
            if L and L > 5 * min(w, t):
                log.append(f"compression specimen {r.get('specimen', '?')} skipped: longer "
                           "than 5 x its smallest side, it buckles instead of crushing")
                continue
            comp.append(F / (w * t))
    if comp:
        changes["f_c_MPa"] = float(np.mean(comp))
        notes["f_c_MPa"] = f"measured: {len(comp)} short-stack compression tests (calibrate)"
        log.append(f"f_c = {changes['f_c_MPa']:.1f} MPa from {len(comp)} compression tests")
    c90 = [_float(r, "plateau_load_N") / (_float(r, "contact_length_mm") * _float(r,
                                                                               "width_mm"))
           for r in _rows(d / "crush_across_template.csv")
           if _float(r, "plateau_load_N") and _float(r, "contact_length_mm")
           and _float(r, "width_mm")]
    if c90:
        params["wood.f_c90_MPa"] = float(np.mean(c90))
        log.append(f"wood.f_c90_MPa = {params['wood.f_c90_MPa']:.2f} from {len(c90)} blocks "
                   "pressed flat")
    ends = [_float(r, "failure_load_N") / (_float(r, "width_mm") * _float(r, "thickness_mm"))
            for r in _rows(d / "end_grain_template.csv")
            if _float(r, "failure_load_N") and _float(r, "width_mm") and _float(r, "thickness_mm")]
    if ends:
        tau = changes.get("glue.tau_g_MPa", mat.glue.tau_g_MPa.value)
        fac = float(np.mean(ends)) / (sp["glue.peel_to_shear"] * tau)
        params["glue.end_grain_factor"] = float(np.clip(fac, 0.05, 1.5))
        log.append(f"glue.end_grain_factor = {params['glue.end_grain_factor']:.2f} from "
                   f"{len(ends)} end-to-end joints (their strength over the peel strength "
                   "of a face joint)")
    ks, gs = [], []
    by_spec: dict[str, list] = {}
    for r in _rows(d / "lap_slip_template.csv"):
        if _float(r, "load_N") is not None and _float(r, "slip_mm") is not None:
            by_spec.setdefault(r.get("specimen", ""), []).append(r)
    for rows in by_spec.values():
        A = _float(rows[0], "overlap_mm") * _float(rows[0], "width_mm") \
            if _float(rows[0], "overlap_mm") and _float(rows[0], "width_mm") else None
        F = np.array([_float(r, "load_N") for r in rows], float)
        x = np.array([_float(r, "slip_mm") for r in rows], float)
        if A is None or len(F) < 3 or F.max() <= 0:
            continue
        k_sec = _secant(F, x, float(F.max()))
        if math.isfinite(k_sec):
            ks.append(k_sec / A)
        gs.append(float(np.trapezoid(F, x)) / A)
    if ks:
        params["glue.K_s_N_per_mm3"] = float(np.mean(ks))
        log.append(f"glue.K_s_N_per_mm3 = {params['glue.K_s_N_per_mm3']:.1f} from {len(ks)} "
                   "lap load-slip records (secant 10-50 % of the peak / glued area; it "
                   "includes the sticks' own stretching over the overlap)")
    if gs:
        params["glue.G_IIc_N_per_mm"] = float(np.mean(gs))
        log.append(f"glue.G_IIc_N_per_mm = {params['glue.G_IIc_N_per_mm']:.2f} from "
                   f"{len(gs)} lap load-slip records (work to break / glued area)")


# ------------------------------------------------------------------------- monte carlo


def montecarlo(cad: str | Path, n: int = 8, overrides=None, material=None,
               sim_params=None, ruleset=None, options: CrushOptions | None = None,
               processes: int | None = None, fidelity: str = "standard",
               out_dir: str | Path | None = None, progress=None, units=None,
               up: str = "y", span: str = "x",
               sample_placeholders: bool = False) -> MonteCarloReport:
    """Crush the bridge ``n`` times with different random imperfections, material scatter
    and plate placement (seeds 1..n) and report the spread of the ultimate load. Runs
    ``processes`` crushes at a time (default: one per two CPU cores), splitting the
    cores between them. ``sample_placeholders``: each run also draws the material values
    that are still placeholders (those of ``sensitivity``) around their placeholder
    (lognormal, ``PLACEHOLDER_COV``), so the spread includes what they leave unknown."""
    import concurrent.futures as cf
    import os

    if n < 1:
        raise ValueError("n must be at least 1 (number of crushes)")
    cpus = os.cpu_count() or 1
    procs = max(1, min(n, processes or max(1, cpus // 2)))
    threads = max(1, cpus // procs)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    _, _, _, ov_raw, mat_, sp_, _ = _inputs(cad, overrides, material, sim_params, ruleset,
                                            units, up, span)
    assumed = _assumed(mat_, sp_)
    drawn = placeholders(mat_, sp_) if sample_placeholders else {}
    args = [(str(cad), draw_placeholders(ov_raw, drawn, s) if drawn else overrides,
             material, sim_params, ruleset, s, options, fidelity,
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
    Fs = sorted(r["Fu_strict_kgf"] for r in res)
    rep.Fu_strict_kgf = [round(x, 2) for x in Fs]
    rep.p50_strict_kgf = round(float(np.percentile(Fs, 50)), 2)
    rep.recommended_prediction_strict_kgf = round(best_prediction(Fs), 1)
    rep.min_kgf, rep.max_kgf = round(F[0], 2), round(F[-1], 2)
    rep.assumed_parameters = assumed
    rep.sampled_placeholders = sorted(drawn)
    small = (f" With {n} runs the 5 % and 95 % points are interpolated between the "
             "extreme runs: read min_kgf and max_kgf instead, and use n >= 20 for real "
             "percentiles." if n < 20 else "")
    if drawn:
        rep.note = ("The spread covers member bow, stick-to-stick and joint-to-joint "
                    "scatter, plate placement and the placeholder material values "
                    f"({', '.join(sorted(drawn))}), each drawn per run around its "
                    f"placeholder (lognormal, {PLACEHOLDER_COV:.0%}): an honest range "
                    "until they are measured." + small)
    else:
        rep.note = ("The spread covers member bow, stick-to-stick and joint-to-joint "
                    "scatter and plate placement. It does not cover the material values "
                    "themselves: "
                    + (f"{len(assumed)} are still placeholders, which can move F_u by tens "
                       "of % (bridgesim sensitivity shows which); measure them, or add "
                       "--sample-placeholders to include them in the range."
                       if assumed else "they are all measured.") + small)
    if out_dir:
        (Path(out_dir) / "montecarlo.json").write_text(rep.model_dump_json(indent=2),
                                                      encoding="utf-8")
    return rep


def placeholders(mat: Material, sp: SimParams) -> dict[str, float]:
    """The values of ``sensitivity``'s list that are still placeholders, as override keys
    with their placeholder value."""
    assumed = set(_assumed(mat, sp))
    out = {}
    for k in SENSITIVITY:
        if k.startswith("material.") and k in assumed:
            out[k] = mat.props()[k[len("material."):]].value
        elif not k.startswith("material.") and f"sim.{k}" in assumed:
            out[k] = sp[k]
    return out


def draw_placeholders(overrides: dict | None, drawn: dict[str, float], seed: int) -> dict:
    """The overrides of one montecarlo run with each placeholder value drawn around its
    placeholder (lognormal, median the placeholder, sigma ``PLACEHOLDER_COV``), the same
    for the same seed; the user's own overrides are kept."""
    rng = np.random.default_rng([seed, 104729])
    doc = json.loads(json.dumps(overrides or {}))
    prm = dict(doc.get("params") or {})
    for k, v in drawn.items():
        x = v * float(np.exp(rng.normal(0.0, PLACEHOLDER_COV)))
        if k == "glue.end_grain_factor":
            x = min(x, 1.0)  # end grain never glues better than side grain
        prm[k] = round(x, 6)
    doc["params"] = prm
    return doc


def _mc_init(threads: int) -> None:
    try:
        import numba

        numba.set_num_threads(threads)
    except Exception:  # noqa: BLE001 - numba missing or fewer cores
        pass


def _screening(fidelity: str) -> CrushOptions:
    """Options of a crush run only for its numbers (ab, montecarlo, sensitivity): stop
    2 mm after the bridge is judged failed instead of following the collapse (F_u and
    the failures are known by then; about half the time of a full run)."""
    from bridgesim.sim.crush import FIDELITY

    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {', '.join(FIDELITY)}")
    return CrushOptions(speed_mm_s=FIDELITY[fidelity][1],
                        dt_target=2.2e-6 if fidelity == "fine" else 3e-6,
                        after_failure_mm=2.0)


def _mc_one(a):
    cad, overrides, material, sim_params, ruleset, seed, options, fidelity, out, units, \
        up, span = a
    rep, _ = crush(cad, overrides, material, sim_params, ruleset, seed=seed,
                   options=options or _screening(fidelity), replay=False,
                   fidelity=fidelity, out_dir=out, units=units, up=up, span=span)
    major = next((e for e in rep.failures if e.kind != "joint_cracking"), None)
    where = f"{e_kind(major)}: {major.what}" if major else rep.failure_mode
    strict = rep.Fu_first_joint_failure_kgf
    return {"seed": seed, "Fu_kgf": rep.Fu_kgf, "first_failure": where,
            "Fu_strict_kgf": min(strict, rep.Fu_kgf) if strict else rep.Fu_kgf,
            "trustworthy": rep.quality.trustworthy, "mass_kg": rep.mass_kg,
            "efficiency_kgf_per_kg": rep.efficiency_kgf_per_kg}


def e_kind(e: EventOut) -> str:
    return e.kind.replace("_", " ")


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

    def gov(rs):  # most frequent first failure; ties: the one seed 1 shows
        cnt: dict[str, int] = {}
        for r in rs:
            cnt[r["first_failure"]] = cnt.get(r["first_failure"], 0) + 1
        if not cnt:
            return ""
        top = max(cnt.values())
        return next(r["first_failure"] for r in rs if cnt[r["first_failure"]] == top)

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


#: spread of a placeholder material value drawn by montecarlo --sample-placeholders
#: (lognormal sigma): textbook values for birch sticks and PVA glue differ by about this
PLACEHOLDER_COV = 0.2

SENSITIVITY = {  # key: (meaning, how to measure)
    "material.E_MPa": ("stiffness of a stick along the grain",
                       "3-point bending of single sticks (stick_bending_template.csv)"),
    "material.f_t_MPa": ("tension strength of a stick",
                         "pull single sticks to failure (stick_tension_template.csv)"),
    "material.f_c_MPa": ("crushing strength of a stick along the grain",
                         "short glued stack in compression (stick_compression_template.csv)"),
    "material.f_v_MPa": ("shear strength of a stick", "short-span bending of a stick stack"),
    "material.glue.tau_g_MPa": ("glue joint strength",
                                "single-lap joints pulled apart (glue_joint_template.csv)"),
    "glue.G_IIc_N_per_mm": ("toughness of a glue line in shear",
                            "the load-slip record of a lap test (lap_slip_template.csv)"),
    "wood.f_c90_MPa": ("crushing strength across the grain",
                       "press a block of sticks flat (crush_across_template.csv)"),
    "wood.G_f_N_per_mm": ("fracture energy of a stick in tension",
                          "area under the load-extension curve of a tension test"),
    "glue.end_grain_factor": ("glue strength on end grain / on side grain",
                              "glue two sticks end to end and pull them apart "
                              "(end_grain_template.csv)"),
}
# other values worth a --param (meaning, how to measure); any key of
# materials/simulation.yaml or material.<key> works
SENSITIVITY_MORE = {
    "glue.peel_to_shear": ("peel strength of a glue line / its shear strength",
                           "T-peel test of two glued sticks"),
    "glue.K_n_N_per_mm3": ("stiffness of a glue line", "the slope of a lap test's curve"),
    "glue.K_s_N_per_mm3": ("shear stiffness of a glue line",
                           "the load-slip record of a lap test (lap_slip_template.csv)"),
    "glue.G_Ic_N_per_mm": ("toughness of a glue line in peel", "T-peel test"),
    "wood.f_t90_MPa": ("tension strength across the grain", "pull two sticks glued edge to edge"),
    "wood.densification_strain": ("how far crushed wood compacts",
                                  "press a stick stack flat and record the curve"),
    "contact.seating_mm": ("how far the bearing faces bed in",
                           "the soft start of your crush-test curve"),
    "scatter.support_level_mm": ("how far the support corners are out of level",
                                 "the gap under the corner that rocks on a flat table"),
    "floss.strand_break_N": ("breaking load of one floss strand",
                             "hang weights on a strand"),
}


def sensitivity(cad: str | Path, overrides=None, material=None, sim_params=None,
                ruleset=None, params: list[str] | None = None, rel: float = 0.2,
                seed: int | None = 0, fidelity: str = "quick", processes: int | None = None,
                progress=None, units=None, up: str = "y", span: str = "x",
                seeds: int = 1):
    """Crush the bridge with each material value lowered and raised by ``rel`` (same
    seeds) and rank the values by how much they move F_u: the measurements that would
    narrow the prediction most. ``seeds`` > 1 averages the effect over seeds seed,
    seed + 1, ... (paired), so one seed's choice of failure does not decide the
    ranking."""
    from bridgesim.sim.report import SensitivityItem, SensitivityReport

    if not 0 < rel < 0.9:
        raise ValueError("rel must be between 0 and 0.9")
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    keys = list(dict.fromkeys(params or SENSITIVITY))
    seed_list = [None] if seed is None else [seed + k for k in range(seeds)]
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

    args = []
    for sd in seed_list:
        args.append((f"base|{sd}", (str(cad), with_param(None, 0), material, sim_params,
                                    ruleset, sd, None, fidelity, None, units, up, span)))
        for k in keys:
            for tag, f in (("low", 1 - rel), ("high", 1 + rel)):
                args.append((f"{k}|{tag}|{sd}", (str(cad), with_param(k, base_vals[k] * f),
                                                 material, sim_params, ruleset, sd, None,
                                                 fidelity, None, units, up, span)))
    res = {r["tag"]: r for r in _pool_runs(args, len(args), processes, progress, "crush")}

    def mean_F(tag):
        return float(np.mean([res[f"{tag}|{sd}"]["Fu_kgf"] for sd in seed_list]))

    F0 = mean_F("base")
    items = []
    catalog = {**SENSITIVITY, **SENSITIVITY_MORE}
    for k in keys:
        lo, hi = mean_F(f"{k}|low"), mean_F(f"{k}|high")
        meaning, how = catalog.get(k, (k, "see materials/simulation.yaml"))
        items.append(SensitivityItem(
            param=k, meaning=meaning, how_to_measure=how, source=sources[k],
            base=base_vals[k], Fu_low_kgf=round(lo, 2), Fu_high_kgf=round(hi, 2),
            elasticity=round((hi - lo) / (2 * rel * max(F0, 1e-9)), 3)))
    items.sort(key=lambda it: -abs(it.elasticity))
    first = [f"{it.meaning} ({it.param}, elasticity {it.elasticity:+.2f}): {it.how_to_measure}"
             for it in items if it.source != "measured" and abs(it.elasticity) >= 0.1][:3]
    return SensitivityReport(
        file=Path(cad).name, fidelity=fidelity, seed=seed, seeds=len(seed_list), rel=rel,
        Fu_base_kgf=round(F0, 2), items=items, measure_first=first,
        note=(f"{len(seed_list)} seed(s) (F_u averaged over them), {fidelity} fidelity"
              + (" (coarse mesh: the part that fails first can differ from the standard "
                 "mesh)" if fidelity == "quick" else "")
              + ". A value that changes which part fails first can move F_u by more than "
                "its own effect; use --seeds 3 or --fidelity standard if a ranking "
                f"matters. Values tested: {', '.join(keys)}; any other with --param "
                f"(e.g. {', '.join(list(SENSITIVITY_MORE)[:4])})."))


STIFFNESS_KEYS = ("material.E_MPa", "material.G_MPa", "wood.E_T_MPa",
                  "glue.K_n_N_per_mm3", "glue.K_s_N_per_mm3")
STRENGTH_KEYS = ("material.f_t_MPa", "material.f_c_MPa", "material.f_v_MPa",
                 "material.f_b_MPa", "material.glue.tau_g_MPa", "wood.f_c90_MPa",
                 "wood.f_t90_MPa")


def calibrate_bridge(cad: str | Path, test_csv: str | Path, overrides=None, material=None,
                     sim_params=None, ruleset=None, fidelity: str = "standard",
                     seed: int | None = 0, iterations: int = 4, tol: float = 0.03,
                     progress=None, units=None, up: str = "y", span: str = "x"):
    """Fit a global stiffness factor (E, G, E_T, glue stiffness) and a global strength
    factor (every strength) so the simulated crush of this bridge reproduces a physical
    crush test of it: the secant stiffness (10-50 % of the peak, if the record has a
    curve) and the ultimate load. A fixed-point iteration, one crush per step. The fit
    is model updating: a test used here cannot validate the model any more."""
    from bridgesim.sim.report import BridgeCalibration, CalibrationStep

    t = _read_test_csv(test_csv)
    peak_t = float(t["peak"])
    if not math.isfinite(peak_t) or peak_t <= 0:
        raise ValueError(f"{test_csv}: no load_kgf value")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    jack = t["x_kind"] == "jack_travel_mm"
    k_t = _secant(t["load"], t["x"], float(np.max(t["load"]))) if len(t["load"]) >= 3 \
        else math.nan
    _, _, _, ov_raw, mat, sp, _ = _inputs(cad, overrides, material, sim_params, ruleset,
                                          units, up, span)
    props = mat.props()

    def base(k):
        if k.startswith("material."):
            kk = k.split(".", 1)[1]
            return float(props[kk].value) if kk in props else None
        return float(sp[k]) if k in sp.values else None

    stiff = {k: base(k) for k in STIFFNESS_KEYS if base(k) is not None}
    strong = {k: base(k) for k in STRENGTH_KEYS if base(k) is not None}

    def with_factors(fE, fS):
        d = json.loads(json.dumps(ov_raw or {}))
        d["params"] = dict(d.get("params") or {})
        d["params"].update({k: round(v * fE, 6) for k, v in stiff.items()})
        d["params"].update({k: round(v * fS, 6) for k, v in strong.items()})
        return d

    fE = fS = 1.0
    steps: list[CalibrationStep] = []
    converged = False
    for it in range(iterations):
        if progress:
            progress(it / iterations, f"calibration step {it + 1}: stiffness x{fE:.3f}, "
                                      f"strength x{fS:.3f}")
        rep, _ = crush(cad, with_factors(fE, fS), material, sim_params, ruleset, seed=seed,
                       options=_screening(fidelity), replay=False, fidelity=fidelity,
                       units=units, up=up, span=span)
        F = float(rep.Fu_kgf)
        sl = np.array(rep.curve.load_kgf)
        sx = np.array(rep.curve.jack_travel_mm if jack else rep.curve.deflection_mm)
        if len(sl) and F > 0:
            i_fu = int(np.argmax(sl >= 0.999 * F))
            sl, sx = sl[: i_fu + 1], sx[: i_fu + 1]
        k_s = _secant(sl, sx, F) if F > 0 else math.nan
        eF = (F - peak_t) / peak_t
        ek = (k_s - k_t) / k_t if math.isfinite(k_s) and math.isfinite(k_t) else None
        major = next((e for e in rep.failures if e.kind != "joint_cracking"), None)
        steps.append(CalibrationStep(
            stiffness_factor=round(fE, 4), strength_factor=round(fS, 4), Fu_kgf=round(F, 2),
            stiffness_kgf_per_mm=round(k_s, 2) if math.isfinite(k_s) else None,
            Fu_error=round(eF, 4), stiffness_error=round(ek, 4) if ek is not None else None,
            first_failure=f"{e_kind(major)}: {major.what}" if major else rep.failure_mode))
        if abs(eF) <= tol and (ek is None or abs(ek) <= tol):
            converged = True
            break
        if it == iterations - 1 or F <= 0:
            break
        if ek is not None:
            fE *= float(np.clip(1.0 / (1.0 + ek), 0.5, 2.0))
        fS *= float(np.clip(peak_t / F, 0.5, 2.0))
    last = steps[-1]
    best = with_factors(last.stiffness_factor, last.strength_factor)
    return BridgeCalibration(
        file=Path(cad).name, test=Path(test_csv).name, fidelity=fidelity, seed=seed,
        test_Fu_kgf=round(peak_t, 2),
        test_stiffness_kgf_per_mm=round(k_t, 2) if math.isfinite(k_t) else None,
        test_failure=(t["fail_parts"][0] if t["fail_parts"] else ""),
        stiffness_factor=last.stiffness_factor, strength_factor=last.strength_factor,
        converged=converged, steps=steps, overrides=best,
        note=("Global factors on every stiffness and every strength: they make this "
              "bridge's simulated crush match its test, not each material value right (a "
              "strength factor far from 1 says the governing mechanism is off: compare "
              "first_failure with the test's broken part, and measure that material). "
              "This test is now used up for validation: check the fitted model blind on "
              "another test." + ("" if math.isfinite(k_t) else
                                 " The record has no curve, so only the strength was "
                                 "fitted.")
              + (" The loading plate is a large share of this peak load, so the stiffness "
                 "(10-50 % of the peak) is poorly defined." if peak_t < 10 * float(
                     sp["crusher.plate_mass_kg"]) else "")))


def compare_designs(cad_a: str | Path, cad_b: str | Path, n: int = 3, overrides_a=None,
                    overrides_b=None, material=None, sim_params=None, ruleset=None,
                    fidelity: str = "standard", processes: int | None = None,
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
