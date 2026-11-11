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
import dataclasses
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from bridgesim.cad import CadBody, load_cad, orientation_check
from bridgesim.cad.rules_cad import measure_cad
from bridgesim.materials import load_material
from bridgesim.rules import RuleSet, evaluate_values
from bridgesim.schema import Material
from bridgesim.sim import lab as labmod
from bridgesim.sim.crush import KGF, MINOR_SHARE, CrushOptions, CrushResult, run_crush
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


_CODE_SHA: list[str] = []


def _code_sha() -> str:
    """Fingerprint of the bridgesim code (every module), so a run made by other code
    is never taken for this code's."""
    if not _CODE_SHA:
        h = hashlib.sha256()
        root = Path(__file__).resolve().parents[1]
        for f in sorted(root.rglob("*.py")):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes())
        _CODE_SHA.append(h.hexdigest()[:16])
    return _CODE_SHA[0]


def _values_sha(mat: Material, sp: SimParams) -> str:
    """Fingerprint of the material and simulation values as resolved (16 hex digits)."""
    doc = {"material": mat.model_dump(mode="json"), "params": sp.values}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()
                          ).hexdigest()[:16]


def _provenance(cad, ov_raw: dict, mat: Material, sp: SimParams, fidelity: str) -> dict:
    """Hashes of what a prediction was made from (MonteCarloReport.provenance)."""
    import datetime as _dt

    def canon(x) -> str:
        return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()
                              ).hexdigest()

    return {"cad_sha256": _sha(Path(cad)), "overrides_sha256": canon(ov_raw or {}),
            "values_sha256": canon({"material": mat.model_dump(mode="json"),
                                    "params": sp.values}),
            "code_sha": _code_sha(), "fidelity": fidelity,
            "created_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def _inputs_sha(path: Path, ov_raw: dict, mat: Material, sp: SimParams, seed,
                fidelity: str, mesh_scale) -> str:
    doc = {"cad": _sha(path), "overrides": ov_raw, "material": mat.model_dump(mode="json"),
           "params": sp.values, "seed": seed, "fidelity": fidelity,
           "mesh_scale": mesh_scale, "code": _code_sha()}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()
                          ).hexdigest()[:16]


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
    # a value set under params stays a placeholder (assumed: still drawn by montecarlo
    # around it, still asked for) unless it is listed as measured (calibrate's overrides)
    meas = set(ov.measured)
    sim_over = {k: v for k, v in ov.params.items() if not k.startswith("material.")}
    if sim_over:
        try:
            for src, part in (("measured", {k: v for k, v in sim_over.items() if k in meas}),
                              ("assumed", {k: v for k, v in sim_over.items()
                                           if k not in meas})):
                if part:
                    sp = sp.updated(part, source=src)
        except KeyError as exc:
            raise ValueError(str(exc).strip("'\"")) from None
    mat = _crush_material(mat, sp)
    mat_over = {k.split(".", 1)[1]: v for k, v in ov.params.items() if k.startswith("material.")}
    if mat_over:
        known = set(mat.props())
        bad = [k for k in mat_over if k not in known]
        if bad:
            raise ValueError(f"unknown material parameter(s) {bad}; known: "
                             f"{sorted('material.' + k for k in known)}")
        g_scaled = None
        if "E_MPa" in mat_over and "G_MPa" not in mat_over and mat.E_MPa.value > 0:
            # wood's shear modulus follows its modulus (E/G kept): an E alone would
            # otherwise leave G, and E/G drift (11 to 23 over a montecarlo's draws)
            g_scaled = mat.G_MPa.value * float(mat_over["E_MPa"]) / mat.E_MPa.value
        m_meas = {k: v for k, v in mat_over.items() if f"material.{k}" in meas}
        m_ass = {k: v for k, v in mat_over.items() if f"material.{k}" not in meas}
        if m_meas:
            mat = _material_with(mat, m_meas)
        if m_ass:
            mat = _material_with(mat, m_ass, source="assumed")
        if g_scaled is not None:
            mat = _material_with(mat, {"G_MPa": g_scaled},
                                 {"G_MPa": "assumed: scaled with E (E/G kept)"},
                                 source="assumed")
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
    by_body: dict[int, list] = {}
    for p in model.patches:
        by_body.setdefault(p.body_a, []).append(p)
        by_body.setdefault(p.body_b, []).append(p)
    weak_seen: set[str] = set()
    weak: list[tuple[float, str, float, float]] = []  # ratio, member, glue, member
    for bi, info in enumerate(model.bodies):
        if info.kind != "bar" or info.f_t_eff_MPa <= 0:
            continue
        cap_member = info.f_t_eff_MPa * info.section_mm[0] * info.section_mm[1]
        # the load goes in at one end and out at the other: the weaker end governs
        ends = [0.0, 0.0]
        ax = np.asarray(info.axis if info.axis is not None else (1.0, 0.0, 0.0), float)
        for p in by_body.get(bi, []):
            # (one piece drawn as two bodies counts too: a floor beam with its tenons
            # 'solid' read 230 kgf against 24,473, the adversarial review of round 26)
            if p.kind in ("glue", "solid"):
                s_ = float((np.asarray(p.centroid) - np.asarray(info.centroid_mm)) @ ax)
                # what its glue points hold (end grain, crossing, size, fit-up)
                ends[int(s_ >= 0)] += patch_capacity_N(model, p, mat)
        glue = min(e for e in ends if e > 0) if any(ends) else 0.0
        if glue and glue < 0.5 * cap_member and _group(info.name) not in weak_seen:
            weak_seen.add(_group(info.name))
            weak.append((glue / cap_member, info.name, glue, cap_member))
    if weak:  # one question for all of them: most are members that never see that force
        weak.sort()
        listed = "; ".join(f"'{n}' {g / KGF:.0f} kgf vs {c / KGF:.0f} kgf"
                           for _, n, g, c in weak[:6])
        qs.append(Question(
            id="weak_joints", topic="design", body=weak[0][1],
            question=f"{len(weak)} member(s) (with their copies) have much less glued area "
                     "at their weaker end than the member could carry in tension (the glue "
                     f"in shear vs the member): {listed}"
                     + (f" (and {len(weak) - 6} more)" if len(weak) > 6 else "")
                     + ". A member in compression, or lightly loaded, needs far less: "
                       "`bridgesim screen` shows which joints are actually loaded. Is a "
                       "gusset or lap missing from the CAD on any of them?",
            assumed="the joints are exactly as drawn",
            answer_with={"cad": "add the missing gusset or lap and crush again; if the "
                                "joints are as drawn, acknowledge weak_joints",
                         "bodies": {weak[0][1]: {
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
                                "my_material.yaml -o my_material.overrides.yaml (the "
                                "measured simulation values and scatter; merge it with "
                                "your own overrides)"},
            severity="important"))
    if sp.sources.get("crusher.test_rate_mm_per_min", "assumed") == "assumed" and \
            sp.values.get("glue.rate_exponent", 0.0) > 0:
        # the glue's strength follows its time to failure, which the crusher's pace sets:
        # nobody has timed it (the adversarial review of round 23: a pace 3.8x slower read
        # -10 % on Bridge_v10)
        qs.append(Question(
            id="crusher:pace", topic="material",
            question="How fast does the competition's jack move? Nobody has timed the "
                     "Troitsky crusher, and the glue (white glue holds ~26 % more when it "
                     "fails ten times faster) is taken at the time the test would reach "
                     f"the peak at {sp['crusher.test_rate_mm_per_min']:g} mm/min. Time "
                     "the press head over a few seconds of a video of the competition "
                     "against something of known size, or ask the organisers.",
            assumed=f"{sp['crusher.test_rate_mm_per_min']:g} mm/min (a placeholder, "
                    "drawn 50 % by montecarlo)",
            answer_with={"params": {"crusher.test_rate_mm_per_min": "<measured, mm/min>"},
                         "measured": ["crusher.test_rate_mm_per_min"]},
            severity="check"))
    if sp.sources.get("crusher.stiffness_N_per_mm", "assumed") == "assumed":
        # the ambitious review of round 24: the press's stiffness moved Bridge_v10's F_u by
        # 10 % over 2,000-20,000 N/mm, and its note said it showed in the stroke alone
        qs.append(Question(
            id="crusher:stiffness", topic="material",
            question="How stiff is the competition's press? When a joint cracks, the "
                     "energy stored in the press (its oil, hose and frame) drives on into "
                     "the bridge: a soft press turns cracks a stiff one lets the bridge "
                     "survive into the collapse (Bridge_v10: -3 to -10 % from 20,000 to "
                     "2,000 N/mm). Ask the organisers for the ram's stroke against the load on a "
                     "steel block, or its ram size, pump and hose.",
            assumed=f"{sp['crusher.stiffness_N_per_mm']:g} N/mm (estimated from Appendix "
                    "B's press; drawn 60 % by montecarlo)",
            answer_with={"params": {"crusher.stiffness_N_per_mm": "<measured, N/mm>"},
                         "measured": ["crusher.stiffness_N_per_mm"]},
            severity="check"))
    return qs


# ----------------------------------------------------------------------------- inspect


def _open_questions(qs: list[Question], ov: Overrides, model: Model, strict: bool = False
                    ) -> tuple[list[Question], list[str]]:
    """Questions still open, and the ids the overrides acknowledged (``acknowledge:``).
    Acknowledge patterns that match no question become warnings (with ``strict``, an
    input error, as every other override problem: the adversarial review of round 24)."""
    open_, acked = [], []
    for q in qs:
        (acked if ov.acknowledged(q.id) else open_).append(q)
    ids = [q.id for q in qs]
    bad = []
    for a in ov.acknowledge:
        if not any(i == a or fnmatch.fnmatchcase(i, a) for i in ids):
            bad.append(f"overrides: acknowledge {a!r} matches no question (ids: "
                       f"{', '.join(sorted(ids)[:8])}{' ...' if len(ids) > 8 else ''})")
    if bad and strict:
        raise ValueError("; ".join(bad))
    model.notes += bad
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
            butt_zones_mm=list(getattr(b, "butt_zones", []) or []), why=why))
    return out


def _time_at_peak(curve, sp: SimParams, fallback: float, Fu_kgf: float | None = None
                  ) -> float:
    """When the real test reaches the reported F_u: the jack's travel where the reported
    curve first reaches F_u (``Fu_kgf``; its highest point without it: a load climbing
    again after the failure call, on the platform, is not F_u) at the real jack speed
    (crusher.test_rate_mm_per_min); the time the glue's strength was taken at moment by
    moment (solver.glue_rate)."""
    try:
        L = np.asarray(curve.load_kgf, float)
        x = np.asarray(curve.jack_travel_mm, float)
        rate = float(sp.values.get("crusher.test_rate_mm_per_min", 0.0)) / 60.0
        if len(L) and rate > 0:
            i = int(np.argmax(L))
            if Fu_kgf:
                hit = np.where(L >= 0.995 * Fu_kgf)[0]
                if len(hit):
                    i = int(hit[0])
            return float(x[i]) / rate
    except (AttributeError, TypeError, ValueError):
        pass
    return float(fallback)


def patch_capacity_N(model: Model, p, mat: Material) -> float:
    """What a glued face carries in shear, as the crush's glue points hold it: the sum of
    their areas x their shear strengths (with the end-grain share, the fit-up gaps, the
    seed's scatter and the glue factors), over the local peak factor, dry points left
    out (the adversarial review of round 22: the face's average strength alone read the
    diagonal's end-grain foot at 872 N where its points hold 520). The face's average
    strength x its area where it has no glue points."""
    idx = np.asarray(p.springs, int)
    d0 = model.params.get("c_d0")
    if len(idx) and len(model.c_tau) > int(idx.max()):
        ok = np.ones(len(idx), bool) if d0 is None or len(d0) != len(model.c_tau) \
            else np.asarray(d0)[idx] < 0.5
        return float((model.c_area[idx] * model.c_tau[idx] * ok).sum()
                     / max(p.peak_factor, 1e-9))
    tau = mat.glue.tau_g_MPa.value
    return p.area_mm2 * (p.strength_MPa if p.strength_MPa > 0 else
                         mat.f_v_MPa.value / 1.5 if p.kind == "solid" else tau)


def _joint_reports(model: Model, mat: Material) -> list[JointReport]:
    return [JointReport(a=model.bodies[p.body_a].name, b=model.bodies[p.body_b].name,
                        kind=p.kind, area_mm2=round(p.area_mm2, 1),
                        lamination=p.lamination,
                        shear_capacity_N=round(patch_capacity_N(model, p, mat), 1),
                        exposed_perimeter_mm=round(float(getattr(p, "exposed_mm", 0.0)), 1))
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
                                + design_questions(bodies, model, mat, sp, rules), ov, model,
                                strict)
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
    rep.next_steps = _with_context(
        _next_inspect(rep, path, _ov_ref(overrides, ov_raw)),
        _context(material, sim_params, ruleset, units, up, span))
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
    # the standard crush's mesh, so the screen's stiffness and limits are those of the
    # model the crush runs (at x1.0 it was 14 % softer on Bridge_v10)
    model = _build(bodies, mat, sp, ov, None, 0.75, strict)
    rules = _rules(bodies, mat, rs, model)
    F = load_kgf * KGF
    sol = scr.solve(model, F)
    lims = scr.utilisations(model, sol, F,
                            peak_over_avg=sp.values.get("glue.tau_peak_over_avg", 1.065))
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
        n_limits=len(lims), mass_kg=round(mass, 3), modes=scr.modes(model, sol),
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
    at = {}
    for lm in lims:
        at.setdefault(lm.part, list(lm.where_mm))
    rep.next_steps = [
        NextStep(action="crush", why="the screen ranks the weak spots; the crush follows "
                                     "the bridge to its collapse and gives F_u",
                 parts=weak, where_mm=[at[p] for p in weak],
                 cli=f"bridgesim crush {f}{ov_arg} --fidelity quick --stop-at-failure "
                     f"--out {_results_dir(path)} --json --brief"),
        NextStep(action="strengthen", parts=weak[:3], where_mm=[at[p] for p in weak[:3]],
                 why="the most loaded parts, where damage starts: whether strengthening "
                     "them raises F_u depends on what governs the collapse; crush shows "
                     "it, then test a what-if (overrides: glue_factor, section_scale) "
                     "with ab")]
    _with_context(rep.next_steps, _context(material, sim_params, ruleset, units, up, span))
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
    unread = next((b.file_unread for b in bodies if b.file_unread), ())
    if unread:
        count, names = unread[0], list(unread[1:])
        model.notes.insert(0, f"{count} bodies named in the CAD file could not be read and "
                              f"are not in the model; not matched: {', '.join(names)}")
        model.questions.insert(0, {
            "id": "cad_unread", "topic": "geometry", "severity": "important",
            "body": names[0] if names else "",
            "question": f"{count} bodies the CAD file names could not be read, so they "
                        "are missing from the model (a body read without its name is "
                        "listed as a '(mirror copy)'). Names not matched to a body read: "
                        + ", ".join(names) + ". Do they carry load?",
            "assumed": "the bodies read are the whole bridge",
            "answer_with": {"cad": "save the part again (or export it as Parasolid "
                                   ".x_b, or as STL with all bodies in one file) and "
                                   "check that inspect counts every body; acknowledge "
                                   "cad_unread if the missing ones carry no load"}})
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
                "member_crushing": "member crushing",
                "bearing_crushing": "bearing face crushing",
                "member_erosion": "member element removed (bent past 1 rad)",
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
    from bridgesim.sim.crush import FIDELITY, dt_target_for

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
    if scale < 0.5:
        model.notes.append(f"mesh_scale {scale:g} is finer than any mesh studied (x0.5 on "
                           "Bridge_v6 stayed within the same few % as standard and fine): "
                           "plate elements may be thinner than their plates, and a crack "
                           "may start earlier at a sharp stress concentration (local "
                           "damage; ASSUMPTIONS-SIM.md)")
    rules = _rules(bodies, mat, rs, model)
    # the fine mesh needs a smaller time step, or too much mass scaling would be added
    opt = options or CrushOptions(speed_mm_s=speed, dt_target=dt_target_for(scale),
                                  slow_bridge_speed_mm_s=min(speed, STANDARD_SPEED))
    if stop_at_failure:
        import dataclasses

        opt = dataclasses.replace(opt, after_failure_mm=min(opt.after_failure_mm, 2.0))
    if progress is not None:
        opt.progress = progress
    res = run_crush(model, opt)
    rc = None
    if rate_check:
        import dataclasses

        slow = dataclasses.replace(opt, speed_mm_s=opt.speed_mm_s / 2,
                                   slow_bridge_speed_mm_s=opt.slow_bridge_speed_mm_s / 2)
        if progress is not None:
            slow.progress = lambda f, msg: progress(f, "rate check (half speed): " + msg)
        res2 = run_crush(_build(bodies, mat, sp, ov, seed, scale, False), slow)
        rc = {"speed_mm_s": res2.speed_mm_s, "Fu_kgf": round(res2.Fu_kgf, 2),
              "change": round((res.Fu_kgf - res2.Fu_kgf) / max(res2.Fu_kgf, 1e-9), 4),
              "failure_mode": res2.failure_mode, "governing": res2.governing,
              # the jack's inertia effects grow with the square of its speed (the
              # kinetic energy): F_u extrapolated to a jack at rest along v^2
              "Fu_extrapolated_kgf": round(max(0.0, (4 * res2.Fu_kgf - res.Fu_kgf) / 3),
                                           2)}
    qs, acked = _open_questions([Question(**q) for q in model.questions]
                                + design_questions(bodies, model, mat, sp, rules), ov, model,
                                strict)
    c = res.curve
    step = max(1, len(c["t"]) // 400)
    keep = set(range(0, len(c["t"]), step))
    if c["force"] and res.Fu_N > 0:  # always keep the point where F_u was reached
        keep.add(next((i for i, f in enumerate(c["force"]) if f >= 0.999999 * res.Fu_N),
                      len(c["force"]) - 1))
    keep = sorted(keep)
    raw = c.get("force_raw", c["force"])
    curve = Curve(jack_travel_mm=[round(c["travel"][i], 3) for i in keep],
                  load_kgf=[round(c["force"][i] / KGF, 2) for i in keep],
                  load_raw_kgf=[round(raw[i] / KGF, 2) for i in keep],
                  deflection_mm=[round(c["deflection"][i], 3) for i in keep],
                  joints_broken=[c["glue_broken"][i] for i in keep],
                  members_broken=[c["members_broken"][i] for i in keep])
    mass = rules.measurements.get("mass_kg", model.bridge_mass_kg)
    rep = CrushReport(
        file=path.name, sha256=_sha(path), seed=seed, overrides=ov_raw,
        inputs_sha256=_inputs_sha(path, ov_raw, mat, sp, seed, fidelity, mesh_scale),
        code_sha=_code_sha(), values_sha=_values_sha(mat, sp),
        Fu_kgf=round(res.Fu_kgf, 2), Fu_N=round(res.Fu_N, 1),
        failure_mode=res.failure_mode,
        failure_detail=res.failure_detail, governing=res.governing,
        competing=list(getattr(res, "competing", [])),
        deflection_at_Fu_mm=round(res.deflection_at_Fu_mm, 2),
        test_time_at_Fu_s=round(_time_at_peak(curve, sp, getattr(res, "test_time_at_Fu_s",
                                                                   0.0), res.Fu_kgf), 1),
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
    if any(q.id == "cad_unread" for q in qs):
        # bodies missing from the model: the F_u is of another bridge until the designer
        # says they carry no load (the adversarial review of round 24: a missing chord
        # would still have read trustworthy)
        rep.quality.trustworthy = False
        rep.quality.reasons.append(CAD_UNREAD_REASON)
    rep.next_steps = _with_context(
        _next_crush(rep, path, _ov_ref(overrides, ov_raw), out_dir),
        _context(material, sim_params, ruleset, units, up, span, seed=seed,
                 fidelity=fidelity))
    if out_dir:
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        with (out / "curve.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "jack_travel_mm", "load_N", "load_kgf", "deflection_mm",
                        "joints_broken", "members_broken", "load_raw_kgf"])
            raw = c.get("force_raw", c["force"])
            for i in range(len(c["t"])):
                w.writerow([f"{c['t'][i]:.6f}", f"{c['travel'][i]:.4f}",
                            f"{c['force'][i]:.2f}", f"{c['force'][i] / KGF:.3f}",
                            f"{c['deflection'][i]:.4f}", c["glue_broken"][i],
                            c["members_broken"][i], f"{raw[i] / KGF:.3f}"])
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


MAJOR = ("bearing_crushing", "joint_cracking", "joint_failure", "member_fracture",
         "member_shear",
         "member_crushing", "member_erosion", "plate_crack", "floss_break")


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


# the options each suggested command takes, of those that describe the run's context
_CTX_OPTS = {
    "crush": ("material", "sim-params", "rules", "units", "up-axis", "span-axis", "seed"),
    "screen": ("material", "sim-params", "rules", "units", "up-axis", "span-axis"),
    "inspect": ("material", "sim-params", "rules", "units", "up-axis", "span-axis"),
    "sensitivity": ("material", "units", "up-axis", "span-axis", "seed"),
    "montecarlo": ("material", "units", "up-axis", "span-axis"),
    "ab": ("material", "units", "up-axis", "span-axis", "fidelity"),
    "diagnose": ("material", "units", "up-axis", "span-axis"),
}


def _ref(x, default: str | None = None) -> str | None:
    """A material, simulation-parameter or rules reference as a command-line value: an
    existing file as an absolute path, a bundled name as itself, the default as None."""
    if x is None or not isinstance(x, (str, Path)) or str(x) == (default or ""):
        return None
    p = Path(x)
    return str(p.resolve()) if p.exists() else str(x)


def _context(material=None, sim_params=None, ruleset=None, units=None, up: str = "y",
             span: str = "x", seed: int | None | bool = False,
             fidelity: str | None = None) -> dict[str, str]:
    """The options that make a suggested command run on the same model as this report:
    the material, parameters and rules files, the file's axes and units, the seed and
    (for ab) the fidelity."""
    ctx = {"material": _ref(material, "popsicle_birch"), "sim-params": _ref(sim_params),
           "rules": _ref(ruleset, "troitsky_2027"), "units": units or None,
           "up-axis": None if up == "y" else up, "span-axis": None if span == "x" else span,
           "seed": None if seed is False else str(-1 if seed is None else seed),
           "fidelity": fidelity}
    return {k: v for k, v in ctx.items() if v is not None}


def _with_context(steps: list[NextStep], ctx: dict[str, str]) -> list[NextStep]:
    """Add the run's context to every suggested command that takes it (and does not set
    it already), so following a step never silently changes the model."""
    for st in steps:
        if not st.cli or not st.cli.startswith("bridgesim "):
            continue
        cmd = st.cli.split()[1]
        extra = "".join(f" --{k} {shlex.quote(v)}" for k, v in ctx.items()
                        if k in _CTX_OPTS.get(cmd, ()) and f"--{k}" not in st.cli)
        if extra:
            head, sep, tail = st.cli.partition(" --json")
            st.cli = head + extra + sep + tail
    return steps


def _results_dir(path: Path, out_dir=None) -> str:
    """Where a suggested crush writes: the given folder, or one next to the CAD file
    (absolute either way)."""
    d = Path(out_dir) if out_dir else path.parent / f"{path.stem}_results"
    return shlex.quote(str(d.resolve()))


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
                          cli=f"bridgesim crush {f}{ov} --out {_results_dir(path)} "
                              "--json --brief"))
    return steps + _measure_step(rep.assumed_parameters, f, ov)


def _copies(name: str) -> str:
    """A glob matching a part and its copies on the other sides ('Top Chord - Far' ->
    'Top Chord*', 'End Post L Near' -> 'End Post [LR] [NF]*r'): a design change is made
    on every copy, not on the one this seed broke."""
    from bridgesim.sim.idealize import copies_glob

    return copies_glob(name)


def _whatif(gov: list[EventOut], crushed: set | frozenset = frozenset()) -> dict:
    """Overrides that strengthen the parts that broke at the peak, on every copy of them:
    a joint's glue (a bigger glued area or a gusset), a member's section or a plate's
    thickness. A joint whose face crushed first (``crushed``) gets none: it bears, and
    more glue does not change what its grain can take."""
    bodies: dict[str, dict] = {}
    glue: dict[str, float] = {}
    for e in gov:
        if e.kind in ("joint_failure", "joint_cracking") and len(e.parts) == 2 and \
                e.what not in crushed:
            glue[f"{_copies(e.parts[0])} | {_copies(e.parts[1])}"] = 1.5
        elif e.kind in ("member_shear", "member_fracture", "member_crushing",
                        "member_erosion", "plate_crack"):
            for p in e.parts or [e.what]:
                bodies[_copies(p)] = {"section_scale": 1.25}
    return {k: v for k, v in (("bodies", bodies), ("glue", glue)) if v}


STANDARD_SPEED = 100.0  # mm/s, crush.FIDELITY["standard"]: a slowly ringing bridge's cap


CAD_UNREAD_REASON = ("bodies the CAD file names are missing from the model (cad_unread): "
                     "export the part again so that every body is read, or acknowledge "
                     "cad_unread if they carry no load")


def _next_crush(rep: CrushReport, path: Path, overrides, out_dir) -> list[NextStep]:
    f, ov = shlex.quote(str(path.resolve())), _ov_arg(overrides)
    out = _results_dir(path, out_dir)
    # the what-if (with the user's overrides) is written there by crush(out_dir=...)
    wy = (shlex.quote(str(Path(out_dir).resolve() / "whatif.yaml")) if out_dir
          else "whatif.yaml")
    steps: list[NextStep] = []
    q = rep.quality
    order = ("quick", "standard", "fine")
    # the studies run on a mesh fine enough for the members that fail: quick for
    # diagnose (and standard for montecarlo) unless the bridge needs finer
    need = order.index(q.min_fidelity) if q.min_fidelity in order else 0
    # diagnose at the crush's own fidelity, at least standard: quick's label can name
    # another joint than standard's, and its what-ifs read a joint's gain half again too
    # high (the adversarial review of round 26: the stub +3.6 % at quick where the
    # gusset laps gave +30 %)
    have = order.index(rep.fidelity) if rep.fidelity in order else 1
    dfid, mfid = order[max(need, 1, have)], order[max(need, 1)]
    # (a reason no finer fidelity cures, at fine, gets no rerun: it would loop)
    if not q.trustworthy and any(r != CAD_UNREAD_REASON and not (
            "no fidelity resolves it" in r and rep.fidelity == "fine") for r in q.reasons):
        # a mesh too coarse for the failing members is cured by a finer one, not by the
        # rate check (the adversarial review of round 24: the step looped at quick)
        rfid = rep.fidelity
        if rfid in order and need > order.index(rfid):
            rfid = order[need]
        steps.append(NextStep(
            action="rerun", why="not trustworthy: " + "; ".join(
                r for r in q.reasons if r != CAD_UNREAD_REASON),
            cli=f"bridgesim crush {f}{ov} --fidelity {rfid} --rate-check "
                f"--out {out}_check --json --brief"))
    broke = [e for e in rep.failures if e.kind != "joint_cracking"]
    # the governing failure first (the break that started the fall of the load), then
    # the others that broke within 10 % of F_u
    pool = list(rep.failures) + list(rep.events or [])
    lead = [e for e in pool if f"{e.kind.replace('_', ' ')}: {e.what}" == rep.governing][:1]
    if not lead and rep.governing.startswith("joint failure: "):  # let go during the fall
        part = rep.governing.split(": ", 1)[1]
        lead = [e.model_copy(update={"kind": "joint_failure"}) for e in pool
                if e.what == part and e.kind in ("joint_cracking", "glue_damage")][:1]
    near = [e for e in broke if e.load_kgf >= 0.9 * rep.Fu_kgf and e not in lead]
    # every joint of the governing part that broke with it (a diagonal's foot and both
    # its gusset laps: strengthening one leaves the others), then the others, up to 3
    lead_parts = set(lead[0].parts) if lead else set()
    same = [e for e in near if e.kind == "joint_failure" and lead_parts & set(e.parts)]
    gov = (lead + same + [e for e in near if e not in same][:max(0, 3 - len(lead)
                                                                - len(same))]) \
        or broke[:3]
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
            action="strengthen", parts=[p for w in soft if at[w] for p in w.split(" | ")],
            where_mm=[at[w] for w in soft if at[w] for _p in w.split(" | ")],
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
        where: list[list[float]] = []  # one per part: where its failure happened
        for e in gov:
            for p in e.parts or [e.what]:
                if p not in parts and e.where_mm:
                    parts.append(p)
                    where.append(e.where_mm)
        # faces of the parts that failed whose wood crushed first, before the peak: they
        # bear, and their first non-linearity was the bearing, not the glue
        gp = {p for e in gov for p in e.parts}
        pre = [e for e in rep.failures if e.kind == "bearing_crushing"
               and set(e.parts) & gp and e.load_kgf < rep.Fu_kgf]
        crushed = {e.what for e in pre}
        bear = ("; first the wood under " + ", ".join(
            f"{e.what} crushed (from {e.load_kgf:.0f} kgf)" for e in pre[:3])
            + ": those faces bear more than their grain takes, which more glue does not "
              "change: give them a bigger bearing face, or the grain along the force (end "
              "grain on end grain)") if pre else ""
        wi = _whatif(gov, crushed)
        ova = ov.replace(" -o ", " --overrides-a ")
        steps.append(NextStep(
            action="strengthen", parts=parts,
            where_mm=where,
            why="; ".join(f"{e.what}: {e.kind.replace('_', ' ')} at {e.load_kgf:.0f} kgf "
                          f"({e.load_kgf / max(rep.Fu_kgf, 1e-9):.0%} of F_u)" for e in gov)
            + bear
            + ". The bridge failed there: strengthen it (glued area, a gusset, a deeper "
              "member). Try it first as a what-if (`overrides`: 1.5x the glue of a joint "
              "that separated, 1.25x the section of a part that broke, on every copy of "
              "it; with your own overrides: whatif.yaml in the output folder) with ab on "
              "the same seeds; if it helps, make the change in the CAD and compare the two "
              "files",
            overrides=wi,
            cli=(f"bridgesim ab {f} {f}{ova} --overrides-b {wy} -n 3 --json" if wi
                 else f"bridgesim ab {f} <changed design> -n 3 --json")))
        steps.append(NextStep(
            action="diagnose", parts=parts,
            why="which of these parts really limits the bridge (several often break "
                "within a few % of each other, and the label names the first in a "
                "cascade): diagnose strengthens each in turn on the same seeds and ranks "
                "them by the rise of F_u",
            cli=f"bridgesim diagnose {f}{ov} -n 2 --fidelity {dfid} --out {out}_diagnose "
                "--json"))
        close = {e.what for e in gov if e.load_kgf <= 1.03 * gov[0].load_kgf}
        if rep.fidelity == "quick":
            steps.append(NextStep(
                action="confirm", why="quick fidelity is for ranking designs on one "
                "mesh: its F_u was -7 to +17 % from standard on Bridge_v10 (rounds 25-26; "
                "1-12 % above it on Bridge_v6 and Bridge_v10 in rounds 18-20), but about "
                "half of it on a light truss of 6 mm members, and the part that breaks "
                "first can differ at standard (round 26: quick named the diagonal's "
                "foot, standard the gusset laps that carry the load)",
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
        sample = " --sample-placeholders" if rep.assumed_parameters else ""
        steps.append(NextStep(
            action="predict", why="one seed is one possible build: the spread over "
                                  "imperfections and scatter gives recommended_prediction_kgf "
                                  "for the Load Prediction Form"
                                  + (" (--sample-placeholders: the range also covers the "
                                     "material values not yet measured)" if sample else "")
                                  + f"; after the real test, `bridgesim compare "
                                    f"{out}_montecarlo/montecarlo.json my_test.csv` says "
                                    "whether it fell inside the predicted range",
            cli=f"bridgesim montecarlo {f}{ov} -n 8{sample}"
                + (f" --fidelity {mfid}" if mfid != "standard" else "")
                + f" --out {out}_montecarlo --json"))
    steps.append(NextStep(
        action="compare", why="after a real crush test (a prototype or the competition): "
                              "the reality gap and what to measure next",
        cli=f"bridgesim compare {out}/crush.json my_test.csv --json"))
    return steps


def select(rep: InspectReport, body: str | list[str] | None = None,
           joint: str | list[str] | None = None) -> InspectReport:
    """The inspect report with only the bodies whose name matches ``body``, the joints
    with one of those bodies or one of whose parts (or whose 'A | B' name) matches
    ``joint`` (glob patterns, case ignored), and only the questions about them: to look
    up a few parts of a big bridge."""
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
    if bp or jp:
        sel_b = set(names)
        upd["joints"] = [j for j in rep.joints if j.a in sel_b or j.b in sel_b
                         or (jp and (hit(f"{j.a} | {j.b}", jp) or hit(j.a, jp)
                                     or hit(j.b, jp)))]
        if jp:
            names |= {p for j in upd["joints"] for p in (j.a, j.b)}
    else:
        upd["joints"] = rep.joints
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
    d["questions"] = [q for q in d["questions"] if q["severity"] == "important"]
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
    """The strict reading: the highest load up to the first glued joint separating or
    member splitting in shear, what a judge who stops the test there reads on the load
    cell (its peak), not the load the moment it happens (on a falling branch when the
    joint goes after the peak)."""
    # a joint worth a few % of the load (MINOR_SHARE) letting go is not what a judge
    # stops the test for: it carried next to nothing
    hit = [e for e in res.events if e.kind in ("joint_failure", "member_shear")
           and e.capacity_N >= MINOR_SHARE * res.Fu_N]
    if not hit:
        return None
    tt = np.asarray(res.curve.get("t", []), float)
    f = np.asarray(res.curve.get("force", []), float)
    upto = f[tt <= hit[0].t] if len(tt) == len(f) and len(tt) else np.zeros(0)
    return round(max(float(upto.max()) if len(upto) else 0.0, hit[0].force_N) / KGF, 2)


KE_LIMIT = 0.05  # kinetic energy / jack work while the load rises: quasi-static below
RELOAD_RATIO = 1.1  # a load after the failure call above this x F_u: flagged
# a failing (or competing) joint narrower than its plate's elements by more than this:
# 1.5 since round 27 (3 before; the literature review of round 26, bridgesim's own plate
# element: peel off by up to 1.9x at 2.6x, within 10 % at 1.5 elements across a patch)
JOINT_MESH_RATIO = 1.5
LIGHT_MESH_RATIO = 3.0  # elements longer than this x the failing members' depth: too coarse
# light members (shallower than LIGHT_DEPTH_MM) need elements no longer than
# LIGHT_MESH_RATIO_THIN x their depth: the light truss's 6 mm members read 46.2 kgf at
# standard (12 mm elements) and 35.8 at fine (8 mm), failing elsewhere (round 21);
# round 18's 22 mm elements 34.5 against fine's 44.6
LIGHT_DEPTH_MM = 10.0
LIGHT_MESH_RATIO_THIN = 1.5
# ... and, since round 25, a mesh no coarser than fine (LIGHT_MESH_SCALE) whatever their
# elements: with them at most 1.33x their depth at quick and standard alike, the light
# truss read 18.5 / 37.8 / 44.8 kgf at quick / standard / fine (its plates and glued
# joints, sampled at the plates' element size, set it)
LIGHT_MESH_SCALE = 0.5


def _quality(res: CrushResult, mat: Material, sp: SimParams, rc: dict | None = None
             ) -> Quality:
    reasons = []
    # quasi-static: kinetic energy a few % of the work at most (Abaqus/Explicit's
    # guideline for quasi-static analyses, Getting Started 13.5)
    if res.kinetic_ratio_max > KE_LIMIT and not (rc and abs(rc["change"]) < 0.05):
        reasons.append(f"loading not quasi-static (peak kinetic/work {res.kinetic_ratio_max:.2f}"
                       f" > {KE_LIMIT:g}): run slower or use --rate-check")
    rl = float(getattr(res, "reload_kgf", 0.0) or 0.0)
    fu = float(getattr(res, "Fu_kgf", 0.0) or 0.0)
    if fu > 0 and rl > RELOAD_RATIO * fu:
        # the adversarial review of round 25: a 10 mm truss read 14.7 kgf at its first
        # 30 % dip (a deck edge touching a vertical let go) and carried 39.7 later
        reasons.append(f"the load rose again to {rl:.1f} kgf ({rl / fu:.1f}x F_u) "
                       "after the failure call: a first break (often a small joint or a "
                       "contact) let the load fall, and the bridge then carried more. "
                       "Judges who let the test go on would read the higher load; find "
                       "what broke first (failures) and check that it is really glued")
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
    # a mesh too coarse for a light bridge: elements several member depths long, small
    # glued joints on one or two nodes; on a light truss (members 6 mm deep) quick read
    # 17.9, standard 34.5 and fine 44.6 kgf (validation/README.md 3.10)
    gov = getattr(res, "governing", "") or ""
    scale = float(res.model.params.get("mesh_scale", 1.0))
    bars = {b.name: min(b.section_mm) for b in res.model.bodies
            if b.kind == "bar" and b.section_mm and min(b.section_mm) > 0}
    # the members in the failure (the governing parts), else the bridge's typical member
    gov_parts = [x.strip() for x in gov.split(": ", 1)[-1].split(" | ")] if gov else []
    in_path = [bars[g] for g in gov_parts if g in bars]
    depth = (min(in_path) if in_path else
             float(np.median(list(bars.values()))) if bars else 0.0)
    # the members' elements: 30 mm at x1, but no longer than element_depth_cap
    from bridgesim.sim.idealize import element_depth_cap

    def coarse(sc: float) -> bool:
        el = min(30.0 * sc, element_depth_cap(depth, sc))
        return depth > 0 and (el > LIGHT_MESH_RATIO * depth or (
            depth < LIGHT_DEPTH_MM and (el > LIGHT_MESH_RATIO_THIN * depth
                                        or sc > LIGHT_MESH_SCALE + 1e-9)))

    from bridgesim.sim.crush import FIDELITY

    # the coarsest fidelity whose mesh is fine enough for these members: what diagnose,
    # montecarlo and the other studies of this bridge need
    min_fid = next((k for k, (sc, _) in FIDELITY.items() if not coarse(sc)), "fine")
    elem = min(30.0 * scale, element_depth_cap(depth, scale))
    if coarse(scale):
        reasons.append(f"the mesh is too coarse for the members that fail (elements ~"
                       f"{elem:.0f} mm, members {depth:.0f} mm deep): F_u is not "
                       "converged on such a light bridge (its plates and glued joints "
                       "too); use a finer --fidelity (fine)")
    # a failing joint narrower than the elements of a plate it hangs on (its glue points
    # on long rigid arms to the plate's nodes): any bridge, not only a light one (the
    # adversarial review of round 25: a 10 mm truss read 14.7 kgf at quick against 48
    # at standard and fine, at a 40 mm^2 joint on a deck of 48 mm elements, unflagged)
    # the governing joint and those competing with it (the adversarial review of round
    # 26: a 3.8x deck joint competing at fine escaped while the label went elsewhere)
    gov_j = getattr(res, "governing", "") or ""
    names = [b.name for b in res.model.bodies]
    pairs = [{x.strip() for x in g.split(": ", 1)[-1].split(" | ")}
             for g in [gov_j, *list(getattr(res, "competing", []) or [])]
             if g.startswith("joint")]
    ratio = max([p.mesh_ratio for p in res.model.patches
                 if {names[p.body_a], names[p.body_b]} in pairs] or [0.0])
    if ratio > JOINT_MESH_RATIO:
        need_fid = next((k for k, (sc, _) in FIDELITY.items()
                         if ratio * sc / max(scale, 1e-9) <= JOINT_MESH_RATIO), None)
        reasons.append(f"a joint that fails or competes at the peak is {ratio:.1f}x "
                       "narrower than the elements of the plate it is glued to (its glue "
                       "hangs on far nodes): F_u is not converged there; "
                       + (f"use --fidelity {need_fid}" if need_fid else
                          "no fidelity resolves it (even fine's elements are wider): read "
                          "F_u as uncertain by ~30 %, and widen the joint or the plate"))
        order_f = list(FIDELITY)
        need_fid = need_fid or "fine"
        if order_f.index(need_fid) > order_f.index(min_fid):
            min_fid = need_fid
    # a plate cracking where its elements are larger than the crack band its fracture
    # energy allows, in the mode they cracked in: when such a crack starts depends on the
    # mesh (their strength is lowered to dissipate G_f)
    gov = getattr(res, "governing", "") or ""
    snap = np.asarray(res.model.params.get("s_snap", np.zeros(0)), bool)
    snap3 = np.asarray(res.model.params.get("s_snap3", np.zeros((0, 3))), bool)
    mode = np.asarray(res.model.params.get("s_crack_mode", np.zeros(0)), int)
    if gov.startswith("plate crack: ") and len(snap) == len(res.model.s_body):
        names = [b.name for b in res.model.bodies]
        part = gov.split(": ", 1)[1]
        if part in names:
            sel = res.model.s_body == names.index(part)
            cr = sel & (mode >= 0) if len(mode) == len(sel) else np.zeros_like(sel)
            if cr.any() and len(snap3) == len(sel):
                frac = float(snap3[cr, mode[cr]].mean())
                kinds = ("fibre tension", "across the grain", "shear")
                how = kinds[int(np.bincount(mode[cr], minlength=3).argmax())]
            else:
                frac, how = (float(snap[sel].mean()) if sel.any() else 0.0), "any mode"
            if frac > 0.5:
                reasons.append(f"the failure starts as a crack in '{part}' ({how}), whose "
                               f"cracked elements are larger than the crack band of that "
                               f"mode ({frac:.0%} of them): when it starts depends on the "
                               "mesh; confirm with --fidelity fine")
    return Quality(dt_s=res.dt, steps=res.steps, wall_s=round(res.wall_s, 1),
                   added_mass_fraction=round(res.added_mass_frac, 4),
                   kinetic_to_work_max=round(res.kinetic_ratio_max, 4),
                   settle_sag_mm=round(res.settle_sag_mm, 3), notes=res.notes,
                   trustworthy=not reasons, reasons=reasons,
                   speed_mm_s=getattr(res, "speed_mm_s", 0.0), rate_check=rc,
                   fundamental_period_ms=round(1000 * float(
                       res.model.params.get("period_s", 0.0)), 2),
                   holds=getattr(res, "holds", 0),
                   hold_time_s=round(getattr(res, "hold_time_s", 0.0), 4),
                   min_fidelity=min_fid)


# ----------------------------------------------------------------------------- compare


def _secant(load, x, pk) -> float:
    """Secant stiffness on the rising branch between the first points where the load
    reaches 10 % and 40 % of the peak (interpolated: the sampling does not matter), as
    EN 12512 and EN 26891 take a timber joint's or structure's stiffness (Debertolis et
    al. 2025 for glued trusses); a glue line stays straight to ~65 % of its peak, so the
    range sees the elastic structure, not the bend before the peak."""
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

    d10, d40 = cross(0.1 * pk), cross(0.4 * pk)
    if not (math.isfinite(d10) and math.isfinite(d40)) or d40 <= d10:
        return math.nan
    return 0.3 * pk / (d40 - d10)


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
    # the part that broke first: the body name (or a glob, or 'A | B' for a joint) in the
    # part column, else the description in the notes
    ev = [(r.get("event") or "", (r.get("part") or "").strip() or (r.get("notes") or ""))
          for r in rows]
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
                      meaning=f"secant of load vs {what} between 10 % and 40 % of the "
                              "peak (EN 12512)"),
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
                      "(supports sliding?)" + (". The simulated jack stroke includes the "
                      "machine's stiffness, crusher.stiffness_N_per_mm (10 kN/mm unless "
                      "measured)." if jack else "."))
    if abs(e_peak) > 0.15:
        advice.append("Ultimate load is off by more than 15 %: compare the first thing that "
                      "broke. If it was a joint, calibrate tau_g with lap tests of that joint; "
                      "if a member, calibrate f_t/f_c with stick tests "
                      "(bridgesim sensitivity says which values matter most).")
    loc = next((m_ for m_ in metrics if m_.name == "failure location"), None)
    if loc is not None and loc.error and loc.error > 0:
        advice.append("The test's first break does not match the simulation's: give the "
                      "body name of the part that broke (as inspect lists it, or 'A | B' for "
                      "a joint) in the test's part column, to be sure; if it really broke "
                      "elsewhere, look for that part in the simulated failure sequence: the "
                      "parts that broke in the simulation may be the weaker of the two in the "
                      "model only (glue_factor, strength_factor in the overrides), and "
                      "diagnose says which one the load depends on.")
    return RealityGap(subject=f"{rep.file} vs {Path(test_csv).name}", metrics=metrics,
                      score_0_100=round(score, 1),
                      competition_prediction_error=round(abs(peak_t - peak_s) / peak_t, 4),
                      advice=advice)


_STOP = {"the", "a", "of", "and", "at", "side", "ply", "plate", "broke", "failed", "joint",
         "glue", "ud", "mirror", "copy", "x", "member", "foot", "end", "node", "centre",
         "center", "truss", "between", "on", "to", "its", "in", "where", "first", "glued"}
_SIDES = {"near", "far", "left", "right"}


def _words(s: str) -> set[str]:
    import re

    low = s.lower()
    w = set(re.findall(r"[a-z]+", low))
    side = {"l": "left", "r": "right"}
    w |= {side[x] for x in re.findall(r"\b([lr])\b", low)}
    return w


def _part_match(observed: str, simulated: str) -> float:
    """1 if the observed part ('top chord, far side, left', or a body name or glob from
    the part column: 'Knee-Diagonal R Far*') is the simulated one ('Top Chord (10-ply
    UD) - L Far'; a joint 'A | B' if it names either side), 0.5 if it is the same part
    on another side, else 0."""
    import fnmatch

    obs = observed.strip().lower()
    names = [simulated.lower()] + [p.strip() for p in simulated.lower().split(" | ")]
    pats = [p.strip() for p in obs.split(" | ")]
    if all(any(fnmatch.fnmatchcase(n, p) or fnmatch.fnmatchcase(n, p + "*") for n in names)
           for p in pats if p):
        return 1.0
    o, s_ = _words(observed) - _STOP, _words(simulated) - _STOP
    o_sides, s_sides = o & _SIDES, s_ & _SIDES
    o_part = {w for w in o - _SIDES if len(w) > 1}  # not a node's letter ('node M')
    if not o_part or len(o_part & s_) / len(o_part) < 0.6:
        return 0.0
    return 1.0 if o_sides <= s_sides else 0.5


def _same_part(observed: str, simulated: str) -> bool:
    return _part_match(observed, simulated) == 1.0


# --------------------------------------------------------------------------- lab tests


SHIPPED_E = 10000.0  # materials/popsicle_birch.yaml's E_MPa as shipped


def _crush_material(mat: Material, sp: SimParams) -> Material:
    """The material the crush (and its coupon models) uses: while the stick modulus is a
    placeholder, the crush's own value (wood.E_placeholder_MPa, 10.5 GPa: dry craft sticks
    by vibration), not the material file's."""
    e_ph = sp.values.get("wood.E_placeholder_MPa", 0.0)
    # only while the material file still holds its shipped value: a team that typed its
    # own (assumed) E there gets its own
    if e_ph and mat.E_MPa.source == "assumed" and abs(mat.E_MPa.value - SHIPPED_E) < 1e-6:
        mat = _material_with(mat, {"E_MPa": e_ph, "G_MPa": e_ph / 16}, {
            "E_MPa": "placeholder for the crush: dry craft sticks are 11-15 GPa "
                     "(wood.E_placeholder_MPa); measure yours",
            "G_MPa": "E/16"}, source="assumed")
    return mat


def _wood_law(sp: SimParams) -> dict:
    """The fibre law's compression parameters of the bridge, for the coupon models."""
    return {"fc_prop": sp.values.get("wood.fc_proportional", 1.0),
            "crush_softening": sp["wood.crush_softening"],
            "crush_band_mm": sp.values.get("wood.crush_band_mm", 0.0)}


def coupon_K_s(sp: SimParams, t: float, G: float) -> float:
    """Shear stiffness of a lap of two sticks ``t`` thick along each other's grain: the
    glue line (glue.K_s_N_per_mm3) in series with each stick's shear through its
    thickness, t / (3 G) (as idealize.shear_compliance gives every joint of the bridge)."""
    return 1.0 / (1.0 / sp["glue.K_s_N_per_mm3"] + 2.0 * t / (3.0 * max(G, 1e-9)))


def glue_line_K_s(sp: SimParams, k_lap: float, t: float, G: float) -> float:
    """The glue line's shear stiffness from a lap coupon's (``k_lap``, per mm^2): the two
    sticks' shear through their thickness taken out (coupon_K_s inverted); the coupon's
    own value if the sticks alone are softer than it (a record too stiff to separate)."""
    rest = 1.0 / max(k_lap, 1e-12) - 2.0 * t / (3.0 * max(G, 1e-9))
    return 1.0 / rest if rest > 0.1 / max(k_lap, 1e-12) else k_lap


def _lap(sp: SimParams, tau_peak: float, overlap: float, width: float, E: float,
         t: float = 2.0, G: float | None = None, clamp_MPa: float = 0.0,
         exposed_mm: float | None = None):
    """The simulated single-lap joint with the bridge model's glue: its peel strength the
    lower of peel_to_shear x the shear strength and the sticks' tension across the grain
    (as every glue line of the bridge); its shear stiffness the glue line's in series with
    the sticks' shear through their thickness (coupon_K_s; G: E / 16 unless given); the
    lap size effect and the squeeze-out fillets of its exposed outline (``exposed_mm``,
    default its two ends, 2 x width: two sticks of one width; glue.fillet_mm); pressed
    by ``clamp_MPa`` while it is pulled, it holds more (glue.pressure_coeff)."""
    G = E / 16.0 if G is None else G
    from bridgesim.sim.idealize import fillet_factor, lap_size_factor

    tau_peak = tau_peak * lap_size_factor(overlap * width,
                                          sp.values.get("glue.size_ref_mm2", 200.0),
                                          sp.values.get("glue.weibull_m", 10.0),
                                          sp.values.get("glue.size_cap_mm2", 3000.0))
    fil = float(sp.values.get("glue.fillet_mm", 0.0))
    ff = 1.0
    if fil > 0:  # (its toughness with it, as the bridge's joints: the same slip at let-go)
        ff = fillet_factor(2.0 * width if exposed_mm is None else exposed_mm,
                           overlap * width, fil, sp.values.get("glue.size_ref_mm2", 200.0),
                           sp.values.get("glue.fillet_ref_perimeter_mm", 20.0))
        tau_peak *= ff
    peel = min(sp["glue.peel_to_shear"] * tau_peak, sp["wood.f_t90_MPa"]) / max(tau_peak,
                                                                                1e-12)
    return labmod.lap_shear(tau_peak, peel, sp["glue.K_n_N_per_mm3"],
                            coupon_K_s(sp, t, G), sp["glue.G_Ic_N_per_mm"] * ff,
                            sp["glue.G_IIc_N_per_mm"] * ff, overlap=overlap, width=width, t=t,
                            E=E, **glue_law(sp),
                            pressure_coeff=float(sp.values.get("glue.pressure_coeff", 0.0)),
                            clamp_MPa=clamp_MPa,
                            friction=float(sp.values.get("glue.friction", 0.4)))


def glue_law(sp: SimParams) -> dict:
    """The glue law's pre-peak shape of the bridge model, for the coupon models."""
    return {"onset": sp.values.get("glue.prepeak_onset", 1.0),
            "peak_slip": sp.values.get("glue.peak_slip_ratio", 1.0)}


def lab(material=None, sim_params=None, csv_dir: str | Path | None = None) -> LabReport:
    """Simulate the standard coupon tests with the current material (and, where the CSVs
    in ``csv_dir`` hold data, the team's own specimens) and report the reality gap. The
    sticks are those of the crush: while E is a placeholder, the crush's (10.5 GPa)."""
    mat = material if isinstance(material, Material) else load_material(
        material or "popsicle_birch")
    sp = sim_params if isinstance(sim_params, SimParams) else SimParams.load(sim_params)
    mat = _crush_material(mat, sp)
    E, ft, fc = mat.E_MPa.value, mat.f_t_MPa.value, mat.f_c_MPa.value
    st = mat.stick
    tests: list[dict] = []
    r = labmod.three_point_bending(E, ft, fc, span=100.0, b=st.width_mm, t=st.thickness_mm,
                                   Gf=sp["wood.G_f_N_per_mm"], **_wood_law(sp))
    I = st.width_mm * st.thickness_mm**3 / 12
    mor = r.peak * 100 / 4 / (st.width_mm * st.thickness_mm**2 / 6)
    tests.append({"test": "stick 3-point bending, span 100 mm", "peak_N": round(r.peak, 2),
                  "stiffness_N_per_mm": round(r.stiffness, 3),
                  "theory_stiffness_N_per_mm": round(48 * E * I / 100**3, 3),
                  "modulus_of_rupture_MPa": round(mor, 1),
                  "material_f_b_MPa": mat.f_b_MPa.value})
    ov, wd = mat.glue.overlap_mm.value, st.width_mm
    tau_peak = mat.glue.tau_g_MPa.value * sp.values.get("glue.tau_peak_over_avg", 1.065)
    g = _lap(sp, tau_peak, ov, wd, E, t=st.thickness_mm)
    tests.append({"test": f"single-lap glue joint {ov:g} x {wd:g} mm", "peak_N": round(g.peak, 1),
                  "tau_avg_MPa": round(g.peak / (ov * wd), 3),
                  "material_tau_g_MPa": mat.glue.tau_g_MPa.value})
    # the same lap clamped while it is pulled (glue.pressure_coeff: a pressed line holds
    # more; the share of it already let go slides at glue.friction x the pressure)
    p_c = 1.0
    gc = _lap(sp, tau_peak, ov, wd, E, t=st.thickness_mm, clamp_MPa=p_c)
    tests.append({"test": f"single-lap glue joint {ov:g} x {wd:g} mm clamped at {p_c:g} MPa",
                  "peak_N": round(gc.peak, 1), "tau_avg_MPa": round(gc.peak / (ov * wd), 3),
                  "rise_over_unclamped_MPa": round((gc.peak - g.peak) / (ov * wd), 3),
                  "pressure_coeff": sp.values.get("glue.pressure_coeff", 0.0),
                  "friction": sp.values.get("glue.friction", 0.4)})
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


def _lap_key(r: dict, ov: float, w: float) -> tuple[float, float, float, float]:
    """A lap coupon's kind: its overlap and width (mm), the pressure it was clamped at
    while pulled (clamp_pressure_MPa; blank: 0) and its exposed outline, where a fillet
    can form (exposed_perimeter_mm; blank: its two ends, 2 x width, two sticks of one
    width lapped side by side)."""
    p = _float(r, "clamp_pressure_MPa") or 0.0
    pe = _float(r, "exposed_perimeter_mm")
    return (float(ov), float(w), max(float(p), 0.0),
            float(pe) if pe is not None and pe >= 0 else 2.0 * float(w))


def fit_size_and_fillet(laps: list[tuple[float, float, float]], ref_mm2: float = 200.0,
                        ref_exposed_mm: float = 20.0, cap_mm2: float = 3000.0
                        ) -> tuple[float, float, float] | None:
    """The Weibull modulus of the lap size law and the fillet width that together
    explain laps of three sizes or more ((area mm^2, exposed outline mm, strength MPa)
    per lap): tau = tau_ref (A / A_ref)^(-1/m) (1 + f P_e / A) / (1 + f P_ref / A_ref)
    (idealize.lap_size_factor, fillet_factor), least squares on log tau, 1/m in [0, 1]
    and f in [0, 5] mm. The size law sees a lap's area, the fillet its exposed outline
    over its area, so laps with different perimeter-to-area ratios at one area (a wide
    20 x 30 mm lap beside a 20 x 10) tell them apart. Returns (m, f, tau_ref): m = 0 when
    the data show no size effect (the law's off value); None with fewer than three
    sizes."""
    from scipy.optimize import least_squares

    from bridgesim.sim.idealize import fillet_factor, lap_size_factor

    if len({(round(a, 1), round(p, 1)) for a, p, _t in laps}) < 3:
        return None
    A = np.array([x[0] for x in laps], float)
    P = np.array([x[1] for x in laps], float)
    y = np.log(np.array([x[2] for x in laps], float))

    def model(th):
        k, f = th[1], th[2]
        s = np.array([lap_size_factor(a, ref_mm2, 1.0 / k if k > 1e-9 else 0.0, cap_mm2)
                      for a in A])
        fl = np.array([fillet_factor(p, a, f, ref_mm2, ref_exposed_mm)
                       for a, p in zip(A, P, strict=True)])
        return th[0] + np.log(s) + np.log(fl)

    best = None
    for k0 in (0.02, 0.1, 0.3):  # a few starts: the problem is small but not convex
        for f0 in (0.0, 0.5, 2.0):
            r = least_squares(lambda th: model(th) - y, [float(np.mean(y)), k0, f0],
                              bounds=([-np.inf, 0.0, 0.0], [np.inf, 1.0, 5.0]))
            if best is None or r.cost < best.cost - 1e-12:
                best = r
    t0, k, f = (float(v) for v in best.x)
    # (m above 100, the parameter's range, is no size effect worth the name: off)
    return (1.0 / k if k >= 0.01 else 0.0), f, float(np.exp(t0))


def _fit_lap_laws(joints: list, sp: SimParams, sp_fit: SimParams, E: float,
                  params: dict, log: list) -> SimParams:
    """What the lap coupons (``joints``: (lap kind, load N), _lap_key) tell besides tau_g,
    fitted before it: with laps of three or more sizes pulled unclamped, the size law's
    Weibull modulus and the fillet width together (fit_size_and_fillet); with laps of one
    size pulled at two or more clamp pressures, the pressure coefficient, through the
    simulated clamped lap: the share of a lap already let go at its peak slides at
    glue.friction x the pressure, so a clamped lap's strength rises faster than the
    coefficient alone (with the shipped values ~0.28 per MPa at a coefficient of 0, ~0.4
    more per unit of it), and the slope of the measured strengths over the pressure is
    matched by the simulated one. Returns ``sp_fit`` with what was fitted (also put in
    ``params``)."""
    from bridgesim.sim.idealize import fillet_factor, lap_size_factor

    upd: dict[str, float] = {}
    free = [(kd[0] * kd[1], kd[3], F_ / (kd[0] * kd[1])) for kd, F_ in joints if kd[2] == 0]
    sizes = {(round(a, 1), round(p, 1)) for a, p, _t in free}
    sf = fit_size_and_fillet(free, sp.values.get("glue.size_ref_mm2", 200.0),
                             sp.values.get("glue.fillet_ref_perimeter_mm", 20.0),
                             sp.values.get("glue.size_cap_mm2", 3000.0))
    if sf is not None:
        m_w, f_w, _t = sf
        upd["glue.weibull_m"], upd["glue.fillet_mm"] = round(m_w, 3), round(f_w, 3)
        log.append(f"glue.weibull_m = {m_w:.1f} (0: no size effect) and glue.fillet_mm = "
                   f"{f_w:.2f} fitted together to {len(free)} unclamped laps of "
                   f"{len(sizes)} sizes (the size law sees a lap's area, the fillet its "
                   "exposed outline over its area)")
    elif len(sizes) == 2:
        log.append("laps of two sizes: the size law and the fillets are not told apart "
                   "(three sizes or more with different perimeter-to-area ratios, "
                   "glue.fillet_mm): glue.weibull_m and glue.fillet_mm kept")
    sp_fit = sp_fit.updated(upd, source="measured") if upd else sp_fit
    # the pressure: the lap kind pulled at the most clamp pressures (then the most laps)
    groups: dict[tuple, list] = {}
    for kd, F_ in joints:
        groups.setdefault((kd[0], kd[1], kd[3]), []).append((kd[2], F_ / (kd[0] * kd[1])))
    cand = [g for g in groups.items() if len({p for p, _t in g[1]}) >= 2]
    if cand:
        (ov, w, pe), rows = max(cand, key=lambda g: (len({p for p, _t in g[1]}), len(g[1])))
        x = np.array([p for p, _t in rows], float)
        y = np.array([t for _p, t in rows], float)
        b, a = (float(v) for v in np.polyfit(x, y, 1))
        p0, p1 = float(x.min()), float(x.max())
        ratio = sp.values.get("glue.tau_peak_over_avg", 1.065)
        # the coupon's own size law and fillet out of its level (a, at the clamp p0)
        lvl = lap_size_factor(ov * w, sp_fit.values.get("glue.size_ref_mm2", 200.0),
                              sp_fit.values.get("glue.weibull_m", 10.0),
                              sp_fit.values.get("glue.size_cap_mm2", 3000.0))
        if sp_fit.values.get("glue.fillet_mm", 0.0) > 0:
            lvl *= fillet_factor(pe, ov * w, sp_fit["glue.fillet_mm"],
                                 sp_fit.values.get("glue.size_ref_mm2", 200.0),
                                 sp_fit.values.get("glue.fillet_ref_perimeter_mm", 20.0))
        tp = max(a, 1e-3) * ratio / max(lvl, 1e-9)

        def slope(mu: float) -> float:
            s2 = sp_fit.updated({"glue.pressure_coeff": mu})
            lo = _lap(s2, tp, ov, w, E, clamp_MPa=p0, exposed_mm=pe).peak
            hi = _lap(s2, tp, ov, w, E, clamp_MPa=p1, exposed_mm=pe).peak
            return (hi - lo) / (ov * w) / (p1 - p0)

        s0, s1 = slope(0.0), slope(0.5)
        d_s = (s1 - s0) / 0.5
        mu = min(max((b - s0) / d_s, 0.0), 1.0) if d_s > 1e-9 else 0.0
        if 0.0 < mu < 1.0:
            mu = min(max(mu + (b - slope(mu)) / d_s, 0.0), 1.0)
        upd_p = {"glue.pressure_coeff": round(mu, 4)}
        sp_fit = sp_fit.updated(upd_p, source="measured")
        upd.update(upd_p)
        log.append(f"glue.pressure_coeff = {mu:.3f} from {len(rows)} {ov:g} x {w:g} mm laps "
                   f"clamped at {p0:g}-{p1:g} MPa: their strength rose {b:.3f} MPa per MPa; "
                   f"the simulated lap rises {s0:.3f} per MPa at a coefficient of 0 (the "
                   f"share already let go slides at glue.friction "
                   f"{sp.values.get('glue.friction', 0.4):g} x the pressure)"
                   + (": the data rise no faster than that friction alone gives "
                      "(glue.friction lower?)" if b <= s0 else ""))
    params.update(upd)
    return sp_fit


def at_coupon_time(F: float, r: dict, sp: SimParams) -> float:
    """A glue test's failure load brought to the time to failure tau_g describes
    (glue.coupon_time_s), from the row's time_to_failure_s: tau(t_ref) = tau(t) (t /
    t_ref)^n (glue.rate_exponent), so a lap pulled apart in 10 s does not read 20 % strong;
    unchanged without a time."""
    t = _float(r, "time_to_failure_s")
    n = float(sp.values.get("glue.rate_exponent", 0.0))
    tref = float(sp.values.get("glue.coupon_time_s", 60.0))
    if not t or t <= 0 or n <= 0 or tref <= 0:
        return F
    return F * (t / tref) ** n


def _lab_gaps(d: Path, mat: Material, sp: SimParams) -> list[RealityGap]:
    gaps = []
    memo: dict = {}

    def bending(L, b, t):  # specimens of the same size share one simulation
        key = (round(L, 3), round(b, 3), round(t, 3))
        if key not in memo:
            memo[key] = labmod.three_point_bending(mat.E_MPa.value, mat.f_t_MPa.value,
                                                   mat.f_c_MPa.value, span=L, b=b, t=t,
                                                   Gf=sp["wood.G_f_N_per_mm"], **_wood_law(sp))
        return memo[key]

    rows = _coupon_rows(d, "stick_bending")
    if rows:
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
    rows = _coupon_rows(d, "glue_joint")
    if rows:
        mets = []
        laps: dict = {}
        for r in rows:
            ov, w, Ff = _float(r, "overlap_mm"), _float(r, "width_mm"), _float(
                r, "failure_load_N")
            faces = _float(r, "faces") or 1
            if not (ov and w and Ff):
                continue
            Ff = at_coupon_time(Ff, r, sp)
            tau_peak = mat.glue.tau_g_MPa.value * sp.values.get("glue.tau_peak_over_avg",
                                                                 1.065)
            # each lap simulated at its own size, exposed outline and clamp (laps of one
            # kind share one simulation)
            key = _lap_key(r, ov, w)
            if key not in laps:
                laps[key] = _lap(sp, tau_peak, ov, w, mat.E_MPa.value, clamp_MPa=key[2],
                                 exposed_mm=key[3])
            sim = laps[key]
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
    if _coupon_rows(d, "stick_bending"):
        Es, fails, pairs = [], [], []
        for r in _coupon_rows(d, "stick_bending"):
            L, b, t = _float(r, "span_L_mm"), _float(r, "width_b_mm"), _float(r,
                                                                             "thickness_t_mm")
            F, dl, Ff = _float(r, "load_F_N"), _float(r, "deflection_mm"), _float(
                r, "failure_load_N")
            if L and b and t and F and dl:
                Es.append(F * L**3 / (48 * dl * b * t**3 / 12))
            if L and b and t and Ff:
                fails.append((L, b, t, Ff))
                if F and dl:
                    pairs.append((Es[-1], Ff * L / 4 / (b * t * t / 6)))
        # the simulated sticks: the measured E, else the crush's placeholder
        E = float(np.mean(Es)) if Es else _crush_material(mat, sp).E_MPa.value
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
        if len(pairs) >= 5:  # stiffer sticks are stronger: the correlation of E and f_b
            pe, pf = np.log(np.array(pairs)).T
            if np.std(pe) > 0 and np.std(pf) > 0:
                params["scatter.E_strength_corr"] = float(
                    np.clip(np.corrcoef(pe, pf)[0, 1], 0.0, 0.95))
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
                                                 Gf=sp["wood.G_f_N_per_mm"], **_wood_law(sp))
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
    # the direct coupons first: a lap's measured shear stiffness and toughness set the
    # glue law the lap strength is then fitted through (fitted with the placeholders' and
    # replaced after, a lap's simulated strength no longer matched its test)
    _direct_coupons(d, mat, sp, changes, notes, params, log)
    sp_fit = sp.updated({k: v for k, v in params.items() if k in sp.values
                         and k.startswith("glue.")}, source="measured") \
        if any(k.startswith("glue.") and k in sp.values for k in params) else sp
    if _coupon_rows(d, "glue_joint"):
        joints = []  # (lap kind: overlap, width, clamp MPa, exposed mm; load N)
        for r in _coupon_rows(d, "glue_joint"):
            ov, w, Ff = _float(r, "overlap_mm"), _float(r, "width_mm"), _float(
                r, "failure_load_N")
            faces = _float(r, "faces") or 1
            if ov and w and Ff:
                joints.append((_lap_key(r, ov, w), at_coupon_time(Ff, r, sp) / faces))
        if joints:
            E = changes.get("E_MPa", mat.E_MPa.value)
            sp_fit = _fit_lap_laws(joints, sp, sp_fit, E, params, log)
            tau = float(np.mean([F_ / (kd[0] * kd[1]) for kd, F_ in joints]))
            ratio = sp.values.get("glue.tau_peak_over_avg", 1.065)
            # inverse fit through the simulated lap joint of each measured geometry and
            # clamp (laps of one kind share one simulation)
            for _ in range(3):
                sims = {kd: _lap(sp_fit, tau * ratio, kd[0], kd[1], E, clamp_MPa=kd[2],
                                 exposed_mm=kd[3]).peak for kd in dict.fromkeys(
                                     kd for kd, _F in joints)}
                errs = [F_ / max(sims[kd], 1e-9) for kd, F_ in joints]
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
                # the scatter about each lap's own kind (its size and clamp taken out)
                tj = [F_ / (kd[0] * kd[1]) for kd, F_ in joints] \
                    if len({kd for kd, _F in joints}) == 1 else errs
                params["scatter.glue_cov"] = float(np.std(tj, ddof=1) / np.mean(tj))
    log += coupon_files(d)[1] if d.is_dir() else []
    if not changes and not params:
        if not d.is_dir():
            raise FileNotFoundError(f"no such folder: {d}")
        raise ValueError(
            f"no coupon test results in {d.name}: fill one or more of the templates of "
            "validation/physical/ (stick_bending, glue_joint, stick_tension, "
            "stick_compression, crush_across, end_grain, lap_slip; any file name with "
            "their columns, a row per specimen) and run calibrate on that folder"
            + ("; " + "; ".join(coupon_files(d)[1]) if coupon_files(d)[1] else ""))
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


#: coupon tests: kind -> (words its file name may contain, columns it needs), in the
#: order a name is matched (the more specific first)
COUPONS: dict[str, tuple[tuple[str, ...], set[str]]] = {
    "lap_slip": (("slip",), {"overlap_mm", "width_mm", "load_N", "slip_mm"}),
    "short_beam": (("short_beam", "sbs", "interlaminar"),
                   {"plies", "span_mm", "width_mm", "thickness_mm", "peak_load_N"}),
    "end_grain": (("end_grain", "endgrain", "end-grain", "butt"),
                  {"width_mm", "thickness_mm", "failure_load_N"}),
    "crush_across": (("crush_across", "across", "c90", "perpendicular"),
                     {"contact_length_mm", "width_mm", "plateau_load_N"}),
    "stick_compression": (("compression", "compress"),
                          {"length_mm", "width_mm", "thickness_mm", "failure_load_N"}),
    "stick_tension": (("tension", "tensile"), {"width_mm", "thickness_mm", "failure_load_N"}),
    "stick_bending": (("bending", "bend", "3pt", "three_point", "flex"),
                      {"span_L_mm", "width_b_mm", "thickness_t_mm"}),
    "glue_joint": (("glue", "lap_shear", "lap_joint", "joint"),
                   {"overlap_mm", "width_mm", "failure_load_N"}),
}


def coupon_files(d: Path) -> tuple[dict[str, list[Path]], list[str]]:
    """The coupon test files in folder ``d``, by kind, whatever they are called: by a
    word of their name (``my_tension_tests.csv``) and their columns, or by their columns
    alone when they fit one kind only; and a warning for each CSV that is not
    recognised. Bridge crush records (a load_kgf column) are left to compare."""
    found: dict[str, list[Path]] = {k: [] for k in COUPONS}
    warn: list[str] = []
    for f in sorted(Path(d).glob("*.csv")):
        try:
            head = f.read_text(encoding="utf-8").splitlines()[0]
        except (IndexError, UnicodeDecodeError, OSError):
            warn.append(f"skipped {f.name}: empty or unreadable")
            continue
        cols = {c.strip() for c in head.split(",")}
        if "load_kgf" in cols:  # a bridge crush record: for compare
            continue
        name = f.stem.lower()
        kind = next((k for k, (words, need) in COUPONS.items()
                     if need <= cols and any(w in name for w in words)), None)
        if kind is None:
            fit = [k for k, (_w, need) in COUPONS.items() if need <= cols]
            best = max((len(COUPONS[k][1]) for k in fit), default=0)
            fit = [k for k in fit if len(COUPONS[k][1]) == best]
            if len(fit) == 1:
                kind = fit[0]
            elif fit:
                warn.append(f"skipped {f.name}: its columns fit {' and '.join(fit)}; put "
                            f"one of {', '.join(repr(w) for k in fit for w in COUPONS[k][0][:1])} "
                            "in its name")
                continue
            else:
                warn.append(f"skipped {f.name}: not a coupon test file (columns of "
                            "validation/physical/*_template.csv expected)")
                continue
        found[kind].append(f)
    return found, warn


def _coupon_rows(d: Path, kind: str) -> list[dict]:
    """Every row of the coupon files of ``kind`` in ``d``."""
    return [r for f in coupon_files(d)[0][kind] for r in _rows(f)]


def _direct_coupons(d: Path, mat: Material, sp: SimParams, changes: dict, notes: dict,
                    params: dict, log: list) -> None:
    """Strength and glue values measured directly (no inverse fit): stick tension and
    short-stack compression (f_t, f_c, replacing the ratio-scaled pair from bending), a
    stick block pressed flat (f_c90), sticks glued end to end and pulled (the end-grain
    factor) and the load-slip record of a lap joint (glue shear stiffness and
    toughness)."""
    vals = [_float(r, "failure_load_N") / (_float(r, "width_mm") * _float(r, "thickness_mm"))
            for r in _coupon_rows(d, "stick_tension")
            if _float(r, "failure_load_N") and _float(r, "width_mm") and _float(r, "thickness_mm")]
    if vals:
        changes["f_t_MPa"] = float(np.mean(vals))
        notes["f_t_MPa"] = f"measured: {len(vals)} stick tension tests (calibrate)"
        log.append(f"f_t = {changes['f_t_MPa']:.1f} MPa from {len(vals)} tension tests")
    comp = []
    for r in _coupon_rows(d, "stick_compression"):
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
           for r in _coupon_rows(d, "crush_across")
           if _float(r, "plateau_load_N") and _float(r, "contact_length_mm")
           and _float(r, "width_mm")]
    if c90:
        params["wood.f_c90_MPa"] = float(np.mean(c90))
        log.append(f"wood.f_c90_MPa = {params['wood.f_c90_MPa']:.2f} from {len(c90)} blocks "
                   "pressed flat")
    ends = [_float(r, "failure_load_N") / (_float(r, "width_mm") * _float(r, "thickness_mm"))
            for r in _coupon_rows(d, "end_grain")
            if _float(r, "failure_load_N") and _float(r, "width_mm") and _float(r, "thickness_mm")]
    if ends:
        # sticks glued end to end and pulled apart: an end-grain line in tension, which
        # holds a share of the wood's tension strength along the grain
        ft = changes.get("f_t_MPa", mat.f_t_MPa.value)
        fac = float(np.mean(ends)) / max(ft, 1e-9)
        params["glue.end_grain_tension_factor"] = float(np.clip(fac, 0.005, 1.0))
        log.append(f"glue.end_grain_tension_factor = "
                   f"{params['glue.end_grain_tension_factor']:.3f} from {len(ends)} "
                   "end-to-end joints (their strength over the sticks' tension strength)")
    # short-beam shear of a laminate (ASTM D2344): tau = 0.75 P / (b h) on its glue lines
    sbs = [0.75 * at_coupon_time(_float(r, "peak_load_N"), r, sp)
           / (_float(r, "width_mm") * _float(r, "thickness_mm"))
           for r in _coupon_rows(d, "short_beam")
           if _float(r, "peak_load_N") and _float(r, "width_mm") and _float(r, "thickness_mm")]
    if sbs:
        params["glue.interlaminar_MPa"] = float(np.mean(sbs))
        log.append(f"glue.interlaminar_MPa = {params['glue.interlaminar_MPa']:.2f} from "
                   f"{len(sbs)} short-beam shear tests (0.75 P / (b h); a UD laminate that "
                   "splits in the wood next to its glue lines gives f_v)")
    ks, gs = [], []
    by_spec: dict[str, list] = {}
    for r in _coupon_rows(d, "lap_slip"):
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
        # the toughness only from a record that follows the lap down after its peak (a
        # record that stops at the peak, or drops in one step, has no softening branch:
        # its area is the elastic energy, not the work to break the glue)
        ip = int(np.argmax(F))
        tail = F[ip + 1:]
        if len(tail) >= 2 and tail.min() <= 0.2 * F[ip] and (tail > 0.2 * F[ip]).any():
            gs.append(float(np.trapezoid(F, x)) / A)
        else:
            log.append(f"{rows[0].get('specimen', '')}: no softening branch recorded after "
                       "the peak (at least one reading between the peak and 20 % of it): "
                       "its toughness is not taken")
    if ks:
        # the records' slope is the whole lap's: the glue line is what is left once the
        # two sticks' shear through their thickness is taken out (the crush adds each
        # joint's own wood back, idealize.shear_compliance)
        t_st = mat.stick.thickness_mm
        params["glue.K_s_N_per_mm3"] = float(np.mean(
            [glue_line_K_s(sp, k, t_st, mat.G_MPa.value) for k in ks]))
        log.append(f"glue.K_s_N_per_mm3 = {params['glue.K_s_N_per_mm3']:.1f} from {len(ks)} "
                   f"lap load-slip records (secant 10-40 % of the peak / glued area, "
                   f"{float(np.mean(ks)):.1f} N/mm^3 on average, with the sticks' shear "
                   "through their thickness taken out; it includes their stretching over "
                   "the overlap)")
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
    (lognormal, ``PLACEHOLDER_PRIORS``, a Latin hypercube over the n runs), so the spread
    includes what they leave unknown.
    ``total_p05_kgf`` / ``total_p95_kgf`` widen the runs' spread by the mesh error and the
    model-form error (``uncertainty.mesh_cov``, ``uncertainty.model_cov``)."""
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
    args = [(str(cad), draw_placeholders(ov_raw, drawn, s, n) if drawn else overrides,
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
        # a run flagged only because its loading was not quasi-static is crushed again
        # at half the jack speed, as the flag advises, not left out: such runs are those
        # where a weak joint snaps (the adversarial review of round 24: B's three flagged
        # runs were its three lowest, and leaving them out lifted its median 20 %)
        slow = [r for r in res if not r["trustworthy"] and r.get("reasons") and all(
            x.startswith("loading not quasi-static") for x in r["reasons"])]
        if slow:
            base = options or _screening(fidelity)
            sargs = []
            for r in slow:
                a = list(args[r["seed"] - 1])
                a[6] = dataclasses.replace(
                    base, speed_mm_s=base.speed_mm_s / 2,
                    slow_bridge_speed_mm_s=base.slow_bridge_speed_mm_s / 2)
                a[8] = str(Path(out_dir) / f"seed_{r['seed']}_slow") if out_dir else None
                sargs.append(tuple(a))
            again = {x["seed"]: x for x in ex.map(_mc_one, sargs)}
            for i, r in enumerate(res):
                if r["seed"] in again:
                    res[i] = {**again[r["seed"]], "slowed_from_kgf": r["Fu_kgf"]}
    # untrustworthy runs (not quasi-static, a crack in elements past the crack band,
    # ...) are listed but left out of the range and the prediction, while enough remain
    ok = [r for r in res if r["trustworthy"]]
    use = ok if len(ok) >= max(3, (n + 1) // 2) else res
    F = sorted(r["Fu_kgf"] for r in use)
    modes: dict[str, int] = {}
    parts: dict[str, int] = {}
    for r in res:
        key = r.get("governing") or r["first_failure"]
        modes[key] = modes.get(key, 0) + 1
        g = governing_part(key)
        parts[g] = parts.get(g, 0) + 1
    p = np.percentile(F, [5, 50, 95])
    # the mesh and model-form error (lognormal, independent of the runs): in the total
    # range, and in the prediction, which it pulls down (a wider spread of the true
    # F_u weighs the low side more in the §12.7 error |F - p| / F)
    # (the mesh's scatter at quick is wider: quick read -6 to +8 % from fine on Bridge_v10
    # over rounds 22-26, against fine's -1 to -5 % from standard; the realism review of
    # round 26: one bias for quick of either sign misstates it)
    m_cov = float(sp_.values.get(f"uncertainty.mesh_cov_{fidelity}",
                                 sp_.values.get("uncertainty.mesh_cov", 0.0)))
    s_tot = math.hypot(m_cov, sp_.values.get("uncertainty.model_cov", 0.0))
    # the mesh's and the jack speed's bias, not noise (uncertainty.mesh_bias_*: against
    # fine with a jack at rest): the prediction and the total range are taken toward
    # it, the runs listed as they came
    bias = float(sp_.values.get(f"uncertainty.mesh_bias_{fidelity}", 0.0))
    Fr = F
    F = [x * (1.0 + bias) for x in Fr]
    rep = MonteCarloReport(file=Path(cad).name, n=n, Fu_kgf=[round(x, 2) for x in Fr],
                           p05_kgf=round(float(p[0]), 2), p50_kgf=round(float(p[1]), 2),
                           p95_kgf=round(float(p[2]), 2), failure_modes=modes,
                           governing_parts=dict(sorted(parts.items(),
                                                       key=lambda kv: -kv[1])),
                           recommended_prediction_kgf=round(best_prediction(F, s_tot), 1),
                           fidelity=fidelity,
                           untrustworthy_runs=sum(1 for r in res if not r["trustworthy"]))
    rep.mesh_bias = bias
    rep.provenance = _provenance(cad, ov_raw, mat_, sp_, fidelity)
    rep.recommended_prediction_runs_kgf = round(best_prediction(Fr), 1)
    # how much the judgement of the model-form error moves it (the adversarial review of
    # round 24: from 1443 to 848 kgf on Bridge_v10 over 0.15-0.76)
    mcov = sp_.values.get("uncertainty.model_cov", 0.0)
    rep.prediction_by_model_cov = {
        f"{c:.2f}": round(best_prediction(F, math.hypot(m_cov, c)), 1)
        for c in sorted({0.15, round(float(mcov), 2), 0.76})}
    rep.slowed_runs = sum(1 for r in res if "slowed_from_kgf" in r)
    # the published tests' sign as well as their width (the adversarial review of round
    # 25: they set model_cov, which lowers the prediction, while the sign they show, which
    # would raise it, was left out): shown, not applied
    rep.published_check = PUBLISHED_CHECK
    pb = float(sp_.values.get("uncertainty.published_bias", 0.0) or 0.0)
    if pb > 0:
        rep.prediction_if_biased_kgf = round(best_prediction([x / pb for x in F], s_tot), 1)
    rep.recommended_interval_kgf = prediction_interval(F, s_tot)
    if len(use) < len(res):
        rep.Fu_untrustworthy_kgf = sorted(round(r["Fu_kgf"], 2) for r in res
                                          if not r["trustworthy"])
    Fs = sorted(r["Fu_strict_kgf"] for r in use)
    rep.Fu_strict_kgf = [round(x, 2) for x in Fs]
    rep.p50_strict_kgf = round(float(np.percentile(Fs, 50)), 2)
    rep.recommended_prediction_strict_kgf = round(best_prediction(
        [x * (1.0 + bias) for x in Fs], s_tot), 1)
    rep.min_kgf, rep.max_kgf = round(Fr[0], 2), round(Fr[-1], 2)
    if s_tot > 0:
        rep.total_p05_kgf, rep.total_p95_kgf = total_range(F, s_tot)
    rep.assumed_parameters = assumed
    rep.sampled_placeholders = sorted(drawn)
    small = (f" With {n} runs the 5 % and 95 % points are interpolated between the "
             "extreme runs: read min_kgf and max_kgf instead, and use n >= 20 for real "
             "percentiles." if n < 20 else "")
    if drawn:
        rep.note = ("The spread covers member bow, stick-to-stick and joint-to-joint "
                    "scatter, plate placement and the placeholder material values "
                    f"({', '.join(sorted(drawn))}), each drawn per run around its "
                    "placeholder (lognormal, with the spread of the literature or "
                    "craft-stick range of each: 20-50 %; a Latin hypercube, so each "
                    "value covers its range evenly over the runs). total_p05_kgf / total_p95_kgf "
                    "add the mesh error and the model-form error (uncertainty.*), which "
                    "only a validation test can narrow." + small)
    else:
        rep.note = ("The spread covers member bow, stick-to-stick and joint-to-joint "
                    "scatter and plate placement. It does not cover the material values "
                    "themselves: "
                    + (f"{len(assumed)} are still placeholders, which can move F_u by tens "
                       "of % (bridgesim sensitivity shows which); measure them, or add "
                       "--sample-placeholders to include them in the range."
                       if assumed else "they are all measured.") + small)
    if rep.slowed_runs:
        rep.note += (f" {rep.slowed_runs} run(s) flagged as not quasi-static were crushed "
                     "again at half the jack speed (slowed_runs).")
    if len(use) < len(res):
        rep.note += (f" {len(res) - len(use)} untrustworthy run(s) (Fu_untrustworthy_kgf) "
                     "are left out of the range and the prediction.")
    elif len(ok) < len(res):
        why = sorted({x.split(":")[0].split(" (")[0] for r in res if not r["trustworthy"]
                      for x in r.get("reasons", [])})
        rep.note = (f"NOT RELIABLE: only {len(ok)} of {len(res)} runs are trustworthy "
                    "(" + "; ".join(why) + "), too few to leave the others out: the range "
                    "and the prediction use them all. Re-run at the fidelity the crush "
                    "report's quality.min_fidelity names (or slower). " + rep.note)
    yours = sorted(k for k in (ov_raw.get("params") or {}) if k in drawn)
    if yours:
        # a value typed under params stays a placeholder unless listed as measured (the
        # adversarial review of round 23: a team's own number was redrawn 50 %)
        rep.note += (" Drawn although you set them under params (list them under "
                     "measured: if you measured them): " + ", ".join(yours) + ".")
    rigid = sorted({b for r in res for b in r.get("rigid_under_plate", [])})
    if rigid:
        rep.note += (" The part(s) under the loading plate " + ", ".join(rigid)
                     + " are rigid blocks in the model: they cannot bend, shear or crush, "
                     "so a failure there is missing from every run (check them by hand, "
                     "or model them as members: kind: bar, with axis: the direction each spans).")
    if out_dir:
        (Path(out_dir) / "montecarlo.json").write_text(rep.model_dump_json(indent=2),
                                                      encoding="utf-8")
    return rep


def governing_part(label: str) -> str:
    """A governing label with every copy of its parts together: 'joint failure: End Post
    (..) - L Near | T1 Cover (..) - L Near' -> 'joint failure: End Post (..) | T1 Cover
    (..)'."""
    from bridgesim.sim.idealize import _group

    kind, sep, what = label.partition(": ")
    if not sep:
        return label
    return f"{kind}: " + " | ".join(_group(p) for p in what.split(" | "))


def placeholders(mat: Material, sp: SimParams) -> dict[str, float]:
    """The values montecarlo --sample-placeholders draws (``PLACEHOLDER_PRIORS``: those
    that set where and when a bridge fails) that are still placeholders, as override keys
    with their placeholder value."""
    assumed = set(_assumed(mat, sp))
    crush_mat = _crush_material(mat, sp)  # the E the crush uses while E is a placeholder
    out = {}
    for k in (*PLACEHOLDER_PRIORS, *ADDITIVE_PRIORS):
        if k.startswith("material.") and k in assumed:
            out[k] = crush_mat.props()[k[len("material."):]].value
        elif not k.startswith("material.") and f"sim.{k}" in assumed:
            out[k] = sp[k]
    return out


def _lhs_normal(key: str, seed: int, n: int) -> float:
    """Run ``seed`` (1..n) of a Latin hypercube over n runs: the standard normal's n
    equal-probability strata, one per run, in an order fixed for the key and n, at a
    seeded point within the stratum. Over the n runs every value covers its prior
    evenly. With plain draws, seeds 1-8 all drew the glue's strength below its
    placeholder (a 1-in-256 chance), which pulls the runs' median down."""
    from statistics import NormalDist

    h = zlib.crc32(key.encode())
    order = np.random.default_rng([n, h, 7919]).permutation(n)
    u = float(np.random.default_rng([seed, n, h, 104729]).uniform(0.02, 0.98))
    return NormalDist().inv_cdf((int(order[seed - 1]) + u) / n)


def draw_placeholders(overrides: dict | None, drawn: dict[str, float], seed: int,
                      n: int | None = None) -> dict:
    """The overrides of one montecarlo run with each placeholder value drawn around its
    placeholder (lognormal, median the placeholder, sigma from ``PLACEHOLDER_PRIORS``,
    the literature or craft-stick range of that value), the same
    for the same seed; the user's own overrides are kept. With ``n`` (the number of
    runs, seed 1..n) the draws are a Latin hypercube over the runs (``_lhs_normal``)."""
    rng = np.random.default_rng([seed, 104729])
    doc = json.loads(json.dumps(overrides or {}))
    prm = dict(doc.get("params") or {})
    # the wood's strengths follow its stiffness (stiffer wood is stronger, the
    # correlation the per-body scatter uses, scatter.E_strength_corr): one draw for E,
    # the strengths partly from it
    keys = sorted(drawn)
    if n is not None and n > 1 and 1 <= seed <= n:
        z = {k: _lhs_normal(k, seed, n) for k in keys}
    else:
        z = {k: float(rng.normal()) for k in keys}
    zE = z.get("material.E_MPa", 0.0)
    for k in keys:
        if k in WOOD_STRENGTHS and "material.E_MPa" in z:
            z[k] = E_STRENGTH_CORR * zE + math.sqrt(1 - E_STRENGTH_CORR**2) * z[k]
    for k in keys:
        v = drawn[k]
        if k in ADDITIVE_PRIORS:
            sd, lo, hi = ADDITIVE_PRIORS[k]
            prm[k] = round(min(max(v + sd * z[k], lo), hi), 6)
            continue
        x = v * float(np.exp(z[k] * PLACEHOLDER_PRIORS.get(k, PLACEHOLDER_COV)))
        if k in ("glue.end_grain_factor", "glue.prepeak_onset", "glue.pressure_coeff") \
                or k.endswith("_efficiency"):
            # end grain never glues better than side grain; shares; the pressure
            # coefficient's range (0-1)
            x = min(x, 1.0)
        elif k == "glue.peak_slip_ratio":
            x = max(x, 1.0)  # the peak is never before the straight line reaches it
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
    from bridgesim.sim.crush import FIDELITY, dt_target_for

    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {', '.join(FIDELITY)}")
    return CrushOptions(speed_mm_s=FIDELITY[fidelity][1],
                        dt_target=dt_target_for(FIDELITY[fidelity][0]),
                        after_failure_mm=2.0,
                        slow_bridge_speed_mm_s=min(FIDELITY[fidelity][1],
                                                   FIDELITY["standard"][1]))


def _mc_one(a):
    cad, overrides, material, sim_params, ruleset, seed, options, fidelity, out, units, \
        up, span = a
    rep = _finished(out, cad, seed, fidelity, overrides, material, sim_params, ruleset,
                    units, up, span)
    if rep is None:
        rep, _ = crush(cad, overrides, material, sim_params, ruleset, seed=seed,
                       options=options or _screening(fidelity), replay=False,
                       fidelity=fidelity, out_dir=out, units=units, up=up, span=span)
    major = next((e for e in rep.failures if e.kind != "joint_cracking"), None)
    where = f"{e_kind(major)}: {major.what}" if major else rep.failure_mode
    strict = rep.Fu_first_joint_failure_kgf
    at_peak = [{"kind": e.kind, "what": e.what, "parts": e.parts}
               for e in rep.failures if e.kind != "joint_cracking"
               and e.load_kgf >= 0.9 * rep.Fu_kgf]
    return {"seed": seed, "Fu_kgf": rep.Fu_kgf, "first_failure": where,
            "governing": rep.governing, "at_peak": at_peak,
            "secant_deflection": _rising_secant(rep, False),
            "secant_travel": _rising_secant(rep, True),
            "Fu_strict_kgf": min(strict, rep.Fu_kgf) if strict else rep.Fu_kgf,
            "trustworthy": rep.quality.trustworthy, "reasons": rep.quality.reasons,
            "mass_kg": rep.mass_kg,
            "efficiency_kgf_per_kg": rep.efficiency_kgf_per_kg,
            "rigid_under_plate": sorted(
                {q.id.split(":", 1)[1] for q in rep.questions
                 if q.id.startswith("rigid_under_plate:")}
                | {a.split(":", 1)[1] for a in rep.acknowledged_questions
                   if a.startswith("rigid_under_plate:") and "*" not in a})}


def _finished(out, cad, seed, fidelity, overrides, material=None, sim_params=None,
              ruleset=None, units=None, up="y", span="x") -> CrushReport | None:
    """The report a run of the same study left in ``out``, so an interrupted montecarlo
    resumes instead of starting over: only if everything it depends on is the same (the
    CAD, the overrides as read now, from a file too, the material, the simulation
    parameters, seed, fidelity and the code: ``inputs_sha256``)."""
    f = Path(out) / "crush.json" if out else None
    if f is None or not f.exists():
        return None
    try:
        rep = CrushReport.model_validate_json(f.read_text(encoding="utf-8"))
        path, _b, _ov, ov_raw, mat, sp, _rs = _inputs(cad, overrides, material, sim_params,
                                                      ruleset, units, up, span)
    except Exception:  # noqa: BLE001 - a partial or old file, or inputs that fail: run
        return None
    want = _inputs_sha(path, ov_raw, mat, sp, seed, fidelity, None)
    return rep if rep.inputs_sha256 and rep.inputs_sha256 == want else None


def _rising_secant(rep: CrushReport, jack: bool) -> float:
    """Secant stiffness of a crush's rising branch (10-40 % of F_u), against the jack
    travel or the mid-span deflection, kgf/mm (nan without a curve)."""
    F = float(rep.Fu_kgf)
    sl = np.array(rep.curve.load_kgf)
    sx = np.array(rep.curve.jack_travel_mm if jack else rep.curve.deflection_mm)
    if not len(sl) or F <= 0:
        return math.nan
    i_fu = int(np.argmax(sl >= 0.999 * F))
    return float(_secant(sl[: i_fu + 1], sx[: i_fu + 1], F))


def e_kind(e: EventOut) -> str:
    return e.kind.replace("_", " ")


def _with_error(F, s_tot: float) -> np.ndarray:
    """The runs F, each times a lognormal error of sigma ``s_tot`` (median 1, as the
    placeholders are drawn: the model as likely to read high as low) at 41 equally likely
    points: the true F_u as the model sees it. (Until round 21 mean 1, which put the
    median 4.5 % low with a sigma of 0.3 and pulled the prediction down for a convention:
    the adversarial review of round 20.)"""
    F = np.asarray(F, float)
    if s_tot <= 0:
        return F
    from scipy.stats import norm

    z = norm.ppf((np.arange(41) + 0.5) / 41)
    return (F[:, None] * np.exp(s_tot * z)[None, :]).ravel()


def best_prediction(F, s_tot: float = 0.0) -> float:
    """Prediction minimising the expected §12.7 error E = |F - p| / F over the runs F:
    the median of F weighted by 1/F (lower than the plain median). With ``s_tot`` (the
    mesh and model-form error), over F times that error: the error widens the true F_u
    and pulls the prediction down."""
    F = np.sort(_with_error(F, s_tot))
    if not len(F):
        return 0.0
    w = 1.0 / np.maximum(F, 1e-9)
    c = np.cumsum(w) / w.sum()
    return float(F[int(np.searchsorted(c, 0.5))])


def prediction_interval(F, s_tot: float = 0.0, n_boot: int = 2000) -> list[float]:
    """How far the recommended prediction could move with other runs: its 5 % and 95 %
    points over bootstrap resamples of the runs (a fixed stream). With a few runs in two
    clusters (two competing failure sites) this is wide: run more."""
    F = np.asarray(F, float)
    if len(F) < 2:
        return [round(float(F[0]), 1)] * 2 if len(F) else [0.0, 0.0]
    rng = np.random.default_rng(161803)
    b = [best_prediction(rng.choice(F, len(F)), s_tot) for _ in range(n_boot)]
    return [round(float(np.percentile(b, 5)), 1), round(float(np.percentile(b, 95)), 1)]


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


def renamed_bodies(names_a: list[str], names_b: list[str]) -> list[str]:
    """The bodies of design B with no namesake in design A when B also lost some of A's
    (bodies renamed, not only added): their seeded scatter is drawn afresh (the adversarial
    review of round 21: renaming one diagonal moved its joints from 1.69 to 2.66 MPa)."""
    a, b = set(names_a), set(names_b)
    if not (a - b):
        return []
    return sorted(b - a)


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

    def group(label: str) -> str:
        """A governing failure by part with its copies ('member shear: Deck Stringer
        (..) - R Far' -> 'member shear: Deck Stringer (..)')."""
        kind, sep, what = label.partition(": ")
        if not sep:
            return label
        return kind + ": " + " | ".join(_group(p) for p in what.split(" | "))

    def gov(rs):  # most frequent governing failure by part group; ties: the first seed's
        labels = [group(r.get("governing") or r["first_failure"]) for r in rs]
        if not labels:
            return ""
        top = max(labels.count(x) for x in labels)
        return next(x for x in labels if labels.count(x) == top)

    moved = [f"seed {a['seed']}: A {a.get('governing') or a['first_failure']} -> B "
             f"{b.get('governing') or b['first_failure']}"
             for a, b in zip(ra, rb, strict=True)
             if group(a.get("governing") or a["first_failure"])
             != group(b.get("governing") or b["first_failure"])]

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
    if moved:
        verdict += (f". The failure moved in {len(moved)} of {n} seeds (failure_moved): "
                    "there B is limited by another part, so strengthening that one too may "
                    "be what it takes")
    bad = sum(1 for r in ra + rb if not r["trustworthy"])
    if bad:
        verdict += (f". {bad} of the {2 * n} crushes are not trustworthy (runs_a/runs_b"
                    "[].trustworthy; crush one to see why): confirm the comparison at a "
                    "finer fidelity before relying on it")
    runs = [[DesignRun(seed=r["seed"], Fu_kgf=r["Fu_kgf"], mass_kg=r["mass_kg"],
                       efficiency_kgf_per_kg=r["efficiency_kgf_per_kg"],
                       first_failure=r["first_failure"], governing=r.get("governing", ""),
                       trustworthy=r["trustworthy"])
             for r in rs] for rs in (ra, rb)]
    return DesignComparison(
        a=name_a, b=name_b, n=n, fidelity=fidelity, runs_a=runs[0], runs_b=runs[1],
        Fu_mean_a_kgf=round(float(np.mean([r["Fu_kgf"] for r in ra])), 2),
        Fu_mean_b_kgf=round(float(np.mean([r["Fu_kgf"] for r in rb])), 2),
        Fu_change=round(dF, 4), Fu_change_ci90=ciF, efficiency_change=round(dE, 4),
        efficiency_change_ci90=ciE, significant=sig, verdict=verdict,
        governing_a=gov(ra), governing_b=gov(rb), failure_moved=moved,
        untrustworthy_runs=sum(1 for r in ra + rb if not r["trustworthy"]))


#: spread of a placeholder material value drawn by montecarlo --sample-placeholders
#: (lognormal sigma): textbook values for birch sticks and PVA glue differ by about this
def total_range(F, s_tot: float) -> tuple[float, float]:
    """The 5 % and 95 % points of F_u with the mesh and model-form error: each run's
    F_u times a lognormal error of sigma ``s_tot`` (median 1, independent of the runs),
    sampled with a fixed stream so the same runs give the same range."""
    g = np.random.default_rng(271828).normal(0.0, s_tot, (len(F), 400))
    tot = (np.asarray(F, float)[:, None] * np.exp(g)).ravel()
    return (round(float(np.percentile(tot, 5)), 1), round(float(np.percentile(tot, 95)), 1))


PLACEHOLDER_COV = 0.3  # for a placeholder without its own prior below
CALIBRATED_MODEL_COV = 0.15  # model-form error left after calibrate-bridge on a test
#: the recommended prediction's §12.7 error on the published tests (validation README
#: 3.14, round 26's runs; D semi-blind, on round 26's code), restated each round
PUBLISHED_CHECK = (
    "On the published craft-stick truss tests a prediction like this one missed by: A "
    "82 % (superglue; its rebuild's load path is a known mismatch), B 21 % (its putty "
    "joints a x5 stand-in), C 55 % (it chose the end-grain law), D 77 % (semi-blind: "
    "Boon's Basic Warren, 10 jig-built white-glue bridges, the model reading 0.32 of "
    "them, its compression diagonals buckling at about a third of the load the real ones "
    "held), all but B low: small trusses outside a team's bridge's range, so a judgement, "
    "not a correction (validation README 3.14)")
# drawn partly from the stick modulus's draw (materials/simulation.yaml,
# scatter.E_strength_corr: r = 0.6-0.8 between MOE and MOR, USDA Wood Handbook)
WOOD_STRENGTHS = ("material.f_t_MPa", "material.f_c_MPa", "material.f_v_MPa",
                  "wood.f_c90_MPa")
E_STRENGTH_CORR = 0.7
# how far each placeholder may be from the truth for a team's own sticks and glue
# (lognormal sigma around the placeholder, ~ the half-width of the literature or
# craft-stick range at 90 %): hand-made PVA laps on birch sticks ~0.9-4.5 MPa (nothing
# published; ~2.3 from pressed craft-stick laminates' 3.7 less hand pressure and unsanded
# faces; materials/popsicle_birch.yaml), glue toughness 0.5-2.5 N/mm, craft-stick E
# 7-16 GPa (dry sticks by vibration 11.6-14.8, veneer 14.6, a table-edge test 7.1), f_t
# 25-80 MPa, f_v 5-12 MPa, f_c90
# 4-7.5 MPa, the end-grain factor 0.2-0.6 (USDA Wood Handbook ch. 10: end-grain butt
# joints reach about a quarter of the wood's strength)
PLACEHOLDER_PRIORS = {
    "material.E_MPa": 0.30, "material.f_t_MPa": 0.30, "material.f_c_MPa": 0.25,
    "material.f_v_MPa": 0.25, "material.glue.tau_g_MPa": 0.50,
    "glue.G_IIc_N_per_mm": 0.45, "wood.f_c90_MPa": 0.25, "wood.G_f_N_per_mm": 0.40,
    "glue.end_grain_factor": 0.35,
    # round 18 (the adversarial review: these set where Bridge_v10 fails, and f_v draws
    # above the bond line's strength change nothing in members): the sticks' tension
    # across the grain (veneer 1.1-2.0 MPa), the pressed bond line (6-12), the plies'
    # shear and across-grain fracture energies (0.5-2, 0.2-0.5 N/mm), peel (0.5-0.8 x
    # shear), rolling shear (birch 3.5-4.6 MPa, birch plywood's means ~2.5-4: 30 % since
    # round 25, when it began to cap every lap across a gusset's face), the fit-up of cut
    # ends (0.1-0.6 mm)
    "wood.f_t90_MPa": 0.25, "glue.interlaminar_MPa": 0.25, "wood.G_v_N_per_mm": 0.45,
    "wood.G_f90_N_per_mm": 0.35, "glue.peel_to_shear": 0.25, "wood.f_rolling_MPa": 0.30,
    "scatter.bearing_gap_mm": 0.50,
    # round 19 (the ambitious and realism reviews): the glue lines' stiffness decides where
    # Bridge_v10 fails (halved, its failure moved from node A to node M at -1.7 % F_u);
    # nothing measures it yet: a fully bonded line between 2 mm sticks ~350-550 N/mm^3, a
    # hand-made lap with gaps far softer
    "glue.K_s_N_per_mm3": 0.60, "glue.K_n_N_per_mm3": 0.50,
    "glue.weibull_m": 0.35,  # the lap size effect's modulus: 6-20 (literature, round 19)
    # round 20 (the literature review): the glue law's pre-peak bend (pressed PVAc bends
    # from 60-70 % of its peak, peaks at 0.05-0.07 mm), butt joints in thin plies in
    # tension (0.53-0.8 of the statics) and in compression (0.94-0.96 when they bear)
    "glue.prepeak_onset": 0.15, "glue.peak_slip_ratio": 0.40,
    "wood.butt_tension_efficiency": 0.15, "wood.butt_bearing_efficiency": 0.05,
    # round 21 (the literature review of round 20): the glue law's shape from 0.45-0.75
    # and 1.1-3.8 (Serrano 2000, Wernersson 1994), hand-made laminations 2-4.5 MPa, the
    # glue's rate exponent 0.065-0.22 and the crusher's pace (nobody has timed it), which
    # together set how much stronger the glue is in the crush than in its coupons
    "glue.lamination_K_s_N_per_mm3": 0.35, "glue.rate_exponent": 0.37,
    "crusher.test_rate_mm_per_min": 0.50,
    # how much hand-made joints vary joint to joint (0.17-0.53: pressed lab PVAc ~10 %,
    # clamped gusset coupons 27 %, Debertolis et al. 2025; a node's weakest joint lets go
    # first, so it moves F_u 6-16 %, the adversarial review of round 20)
    "scatter.glue_cov": 0.35,
    # round 22 (the reviews of round 21): the end-grain glue line's share of the wood's
    # tension (nothing measures it: 0.05-0.25), the size effect across the grain (m 5-11,
    # Kruger et al. 2025, Barrett 1974); the wood's moisture is drawn additively
    # (ADDITIVE_PRIORS)
    "glue.end_grain_tension_factor": 0.50, "wood.t90_weibull_m": 0.30,
    # round 24: the modulus across the grain sets every bearing face's stiffness (veneer
    # 214-273 MPa pulled across, Kruger et al. 2025; solid birch radially ~600-900)
    "wood.E_T_MPa": 0.35,
    # round 25 (the realism review of round 24: the judgement constants were never drawn):
    # how steeply a face's stress spreads across the grain (0.23-0.42 with the rolling
    # modulus), how far the bearing faces bed in, the supports out of level, the lean,
    # friction on the plates
    "wood.bearing_spread_across": 0.35, "wood.G_rolling_MPa": 0.40,
    "contact.seating_mm": 0.50,
    "scatter.support_level_mm": 0.50, "scatter.lean_rad": 0.50,
    "contact.friction_plate": 0.30, "contact.friction_platform": 0.30,
    # the press's stiffness (the ambitious review of round 24: 2,000-20,000 N/mm moved
    # Bridge_v10's F_u by 10 %, the energy it lets go into the bridge at a crack)
    "crusher.stiffness_N_per_mm": 0.60,
    # round 27 (the adversarial review of round 26: these move F_u and were never drawn,
    # so the total range left them out): the glue's peel toughness, the gap a glue line
    # fills at full strength, the coupons' time (the rate law's reference), how much a
    # whole bridge's glue varies from one build to the next and its sticks' stiffness and
    # strength stick to stick, how fast crushed wood softens
    "glue.G_Ic_N_per_mm": 0.45, "glue.gap_fill_mm": 0.40, "glue.coupon_time_s": 0.40,
    "scatter.glue_bridge_cov": 0.40, "scatter.E_cov": 0.30, "scatter.strength_cov": 0.30,
    "wood.crush_softening": 0.30,
    # round 28: the gauge length of the crushing fall (a kink band's width, millimetres to
    # a few centimetres in wood; nothing measured on sticks)
    "wood.crush_band_mm": 0.50,
    # round 28: how much more a glue line pressed together holds in shear (nothing
    # measures it for PVA laps on sticks: 0.13-0.69 around the assumed 0.3, at most 1).
    # Not drawn: the within-joint field and the fillet (scatter.glue_point_cov,
    # glue.fillet_mm), off by default as alternatives to the size law that glue.weibull_m
    # fits; drawn around 0 they would stay 0, and switched on without refitting m they
    # count the same observation twice
    "glue.pressure_coeff": 0.50,
}
# placeholders drawn additively, value + sd x z within (low, high): the wood's moisture
# above its coupons' (percentage points; members laminated in the last days before the
# test hold several points more, half the runs at 0, a sixth above 2)
ADDITIVE_PRIORS = {"wood.moisture_excess_pct": (2.0, 0.0, math.inf)}

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
    "scatter.glue_cov": ("how much the glued joints vary joint to joint (workmanship: "
                         "the weakest of a node's joints lets go first)",
                         "ten single laps glued as you glue the bridge "
                         "(glue_joint_template.csv): their coefficient of variation"),
    "glue.K_s_N_per_mm3": ("shear stiffness of a glue line (how a node shares its load "
                           "between its joints)",
                           "the load-slip record of a lap test (lap_slip_template.csv)"),
    "glue.end_grain_factor": ("glue strength on end grain / on side grain",
                              "glue two sticks end to end and pull them apart "
                              "(end_grain_template.csv)"),
    "wood.f_t90_MPa": ("tension strength across the grain (it caps a glue line's peel "
                       "and sets when a 0/90 stick plate cracks)",
                       "pull two sticks glued edge to edge, or a stick across its width"),
    "glue.interlaminar_MPa": ("shear strength of the pressed glue lines between plies "
                              "(where members split, and how load passes a butt joint)",
                              "short-beam shear of a 3-5 ply laminate "
                              "(short_beam_shear_template.csv)"),
    # round 22 (the adversarial review: these moved F_u most and were never ranked)
    # round 25: a lap on a gusset holds no more than the gusset's face sheared across
    # its fibres (it replaced round 23's wood-failure share)
    "wood.f_rolling_MPa": ("rolling shear: a stick sheared across its fibres (what a "
                           "gusset's face passes under a member lapped across it)",
                           "laps of a stick glued across another at 90 degrees "
                           "(glue_joint_template.csv, grain_angle_deg 90): where the wood "
                           "fails, the load over the area"),
    "glue.rate_exponent": ("how much stronger the glue is when it fails faster",
                           "laps failing in ~10 s, ~60 s and ~600 s "
                           "(glue_joint_template.csv, time_to_failure_s)"),
    "crusher.test_rate_mm_per_min": ("how fast the competition's jack moves (the glue's "
                                     "time to failure)",
                                     "time the press head in a video of the competition"),
    "crusher.stiffness_N_per_mm": ("how stiff the competition's press is (the energy "
                                   "it lets go into the bridge when a joint cracks)",
                                   "the jack stroke against the load on a steel block, or "
                                   "the press's ram size and pump from the organisers"),
    "glue.weibull_m": ("how much weaker a large lap is than the coupon",
                       "laps of 200 and ~2000 mm^2 glued the same way"),
    "wood.t90_weibull_m": ("how much more a small joint's wood holds across the grain",
                           "T-pulls of a stick's end on another's face against a stick "
                           "pulled across its width"),
    "glue.end_grain_tension_factor": ("an end-grain glue line pulled apart",
                                      "two sticks glued end to end and pulled "
                                      "(end_grain_template.csv)"),
    # round 25 (the adversarial review of round 24: what sets how stiff a face is, never
    # ranked, so measure_first never asked for the test that pins it)
    "wood.E_T_MPa": ("stiffness of a stick across its grain (how stiff a glued or bearing "
                     "face pressed across the grain is)",
                     "press a stack of sticks flat (crush_across_template.csv): the slope "
                     "below 1 % strain"),
    "wood.bearing_spread_across": ("how the stress under a face pressed across the grain "
                                   "spreads into the wood (how stiff the faces of a node "
                                   "are)",
                                   "a block pressed onto a member's side, the indentation "
                                   "read beside it (crush_across_template.csv)"),
    "glue.K_n_N_per_mm3": ("stiffness of a glue line pressed or pulled across it",
                           "the slope of a T-pull of two glued sticks"),
}
# other values worth a --param (meaning, how to measure); any key of
# materials/simulation.yaml or material.<key> works
SENSITIVITY_MORE = {
    "glue.peel_to_shear": ("peel strength of a glue line / its shear strength",
                           "T-peel test of two glued sticks"),
    "glue.G_Ic_N_per_mm": ("toughness of a glue line in peel", "T-peel test"),
    "wood.G_f_N_per_mm": ("fracture energy of a stick in tension",
                          "an edge-notched stick pulled apart (K_Ic, G = K^2 / E)"),
    "wood.densification_strain": ("how far crushed wood compacts",
                                  "press a stick stack flat and record the curve"),
    "wood.crush_band_mm": ("how much shortening a member crushing in compression takes to "
                           "lose its strength",
                           "a short glued stack crushed past its peak "
                           "(stick_compression_template.csv): the shortening to its plateau"),
    "contact.seating_mm": ("how far the bearing faces bed in",
                           "the soft start of your crush-test curve"),
    "scatter.support_level_mm": ("how far the support corners are out of level",
                                 "the gap under the corner that rocks on a flat table"),
    "floss.strand_break_N": ("breaking load of one floss strand",
                             "hang weights on a strand"),
    "glue.pressure_coeff": ("how much more a glue line holds in shear while a thrust "
                            "presses it together (knee and heel laps)",
                            "laps pulled while clamped at 0, 1 and 2 MPa "
                            "(glue_joint_template.csv, clamp_pressure_MPa)"),
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
                 "wood.f_t90_MPa", "wood.f_rolling_MPa", "glue.interlaminar_MPa",
                 "floss.strand_break_N")


#: the strengths a test governed by each mechanism measures (calibrate-bridge)
MECHANISM_KEYS = {
    "joint": ("material.glue.tau_g_MPa", "wood.f_rolling_MPa", "wood.f_t90_MPa",
              "glue.interlaminar_MPa"),
    "member": ("material.f_t_MPa", "material.f_c_MPa", "material.f_v_MPa",
               "material.f_b_MPa"),
    "plate": ("material.f_t_MPa", "material.f_c_MPa", "wood.f_rolling_MPa"),
    "bearing": ("wood.f_c90_MPa",),
    "floss": ("floss.strand_break_N",),
}


def _mechanism(governing: str) -> str:
    """The mechanism of a governing label (MECHANISM_KEYS' keys, or '')."""
    g = (governing or "").lower()
    if "floss" in g:
        return "floss"
    for k in ("joint", "bearing", "plate", "member"):
        if g.startswith(k) or f" {k}" in g.split(":", 1)[0]:
            return k
    return "joint" if "glue" in g else ""


def calibrate_bridge(cad: str | Path, test_csv: str | Path, overrides=None, material=None,
                     sim_params=None, ruleset=None, fidelity: str = "standard",
                     seed: int | None = 0, iterations: int = 4, tol: float = 0.03,
                     progress=None, units=None, up: str = "y", span: str = "x",
                     seeds: int = 3, processes: int | None = None):
    """Fit a global stiffness factor (E, G, E_T, glue stiffness) and a global strength
    factor (every strength) so the simulated crush of this bridge reproduces a physical
    crush test of it: the secant stiffness (10-40 % of the peak, if the record has a
    curve) and the ultimate load. A fixed-point iteration: each step crushes the bridge
    with seeds 1..``seeds`` (in parallel) and fits their median, so the factors do not
    absorb one seed's imperfections (``seeds=1``: the single ``seed``). The fit is
    model updating: a test used here cannot validate the model any more."""
    from bridgesim.sim.report import BridgeCalibration, CalibrationStep

    t = _read_test_csv(test_csv)
    peak_t = float(t["peak"])
    if not math.isfinite(peak_t) or peak_t <= 0:
        raise ValueError(f"{test_csv}: no load_kgf value")
    if iterations < 1 or seeds < 1:
        raise ValueError("iterations and seeds must be at least 1")
    jack = t["x_kind"] == "jack_travel_mm"
    seed_list = [seed] if seeds == 1 else list(range(1, seeds + 1))
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
        args = [(sd, (str(cad), with_factors(fE, fS), material, sim_params, ruleset, sd,
                      None, fidelity, None, units, up, span)) for sd in seed_list]
        if len(args) == 1:
            runs = [_mc_one(args[0][1])]
        else:
            runs = _pool_runs(args, len(args), processes, None, "crush")
        runs.sort(key=lambda r: (r["seed"] is None, r["seed"] or 0))
        F = float(np.median([r["Fu_kgf"] for r in runs]))
        ks = [r["secant_travel" if jack else "secant_deflection"] for r in runs]
        ks = [k for k in ks if math.isfinite(k)]
        k_s = float(np.median(ks)) if ks and F > 0 else math.nan
        eF = (F - peak_t) / peak_t
        ek = (k_s - k_t) / k_t if math.isfinite(k_s) and math.isfinite(k_t) else None
        gov = Counter(r["governing"] for r in runs).most_common(1)[0][0]
        steps.append(CalibrationStep(
            stiffness_factor=round(fE, 4), strength_factor=round(fS, 4), Fu_kgf=round(F, 2),
            stiffness_kgf_per_mm=round(k_s, 2) if math.isfinite(k_s) else None,
            Fu_error=round(eF, 4), stiffness_error=round(ek, 4) if ek is not None else None,
            first_failure=gov, Fu_kgf_runs=[round(r["Fu_kgf"], 2) for r in runs]))
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
    # the fitted values are no longer placeholders (montecarlo stops redrawing them around
    # the fit), and what is left of the model-form error is a calibrated model's: the
    # scatter from one build of a bridge to the next (hobby stick bridges 6-19 %, Boon
    # 2024; calibrated timber-truss predictions log-sd 0.11, Debertolis et al. 2025; JCSS
    # 10-15 % for calibrated connection models): 0.15 (the adversarial review of round
    # 20: calibrate-bridge left every value a placeholder and the model error at 30 %)
    # only what the test can tell (the adversarial review of round 26: a glue-governed
    # test said nothing of f_c or f_c90, yet every strength was marked measured and the
    # next design's range collapsed): the strengths of the mechanism that governed, the
    # wood's moduli when there was a curve, and only if the fit converged; the model-form
    # error stays until another test, compared blind, shows the fitted model within it
    if converged:
        tested = set(MECHANISM_KEYS.get(_mechanism(last.first_failure), ()))
        if math.isfinite(k_t):
            tested |= {"material.E_MPa", "material.G_MPa"}
        best["measured"] = sorted(set(best.get("measured") or [])
                                  | {k for k in tested if k in best["params"]})
    return BridgeCalibration(
        file=Path(cad).name, test=Path(test_csv).name, fidelity=fidelity,
        seed=seed if seeds == 1 else None, seeds=seed_list, test_Fu_kgf=round(peak_t, 2),
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
                 "(10-40 % of the peak) is poorly defined." if peak_t < 10 * float(
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
    ba = _inputs(cad_a, overrides_a, material, sim_params, ruleset, units, up, span)[1]
    bb = _inputs(cad_b, overrides_b, material, sim_params, ruleset, units, up, span)[1]
    renamed = renamed_bodies([b.name for b in ba], [b.name for b in bb])
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
    if renamed:
        rep.note = (f"{len(renamed)} bodies of B have no namesake in A ({', '.join(renamed[:4])}"
                    + (", ..." if len(renamed) > 4 else "") + "): each body's and each "
                    "joint's random scatter is drawn from a stream named after its bodies, "
                    "so theirs is drawn afresh and part of the difference may be that luck "
                    "(a joint's glue +-30 %). Keep the body names from one version to the "
                    "next (SolidWorks renames unnamed bodies after a feature edit). "
                    + rep.note).strip()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "comparison.json").write_text(rep.model_dump_json(indent=2),
                                                      encoding="utf-8")
    return rep


def diagnose(cad: str | Path, n: int = 2, overrides=None, material=None, sim_params=None,
             ruleset=None, fidelity: str = "quick", candidates: int = 3,
             processes: int | None = None, out_dir: str | Path | None = None,
             progress=None, units=None, up: str = "y", span: str = "x"):
    """What really limits the bridge: crush it with seeds 1..n, take the parts that
    broke near the peak (within 10 % of F_u) in those runs, strengthen each in turn (the
    what-if of next_steps: glue x1.5 for a joint, section x1.25 for a member or plate, on
    every copy) and crush again with the same seeds. The parts are ranked by the paired
    change of F_u: the label 'governing failure' of one run is the first thing that
    broke in a cascade; this measures which part the load really depends on."""
    from bridgesim.sim.report import DiagnosisItem, DiagnosisReport, EventOut

    if n < 1 or candidates < 1:
        raise ValueError("n and candidates must be at least 1")
    path = Path(cad)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    _, _, _, ov_raw, _, _, _ = _inputs(cad, overrides, material, sim_params, ruleset, units,
                                       up, span)

    def run_args(tag, ov):
        return [(tag, (str(cad), ov, material, sim_params, ruleset, sd, None, fidelity,
                       str(Path(out_dir) / f"{tag}_seed_{sd}") if out_dir else None,
                       units, up, span)) for sd in range(1, n + 1)]

    base = _pool_runs(run_args("base", ov_raw or None), n, processes, progress, "base crush")
    base.sort(key=lambda r: r["seed"])
    # candidates: what broke near the peak, by part with its copies, most frequent first
    seen: dict[str, tuple[int, dict]] = {}
    for r in base:
        for e in {(_copies_key(e)): e for e in r["at_peak"]}.values():
            k = _copies_key(e)
            cnt, ev = seen.get(k, (0, e))
            seen[k] = (cnt + 1, ev)
    cands = sorted(seen.items(), key=lambda kv: -kv[1][0])[:candidates]
    if not cands:
        raise ValueError("nothing broke near the peak in the base runs: no part to "
                         "strengthen (see failure_mode: a deflection or contact limit, "
                         "or buckling)")
    args, whatifs, tags = [], {}, []
    for _k, (_cnt, e) in cands:
        ev = EventOut(t_s=0.0, jack_travel_mm=0.0, load_N=0.0, load_kgf=0.0,
                      kind=e["kind"], what=e["what"], parts=e["parts"])
        wi = _whatif([ev])
        # the runs of each what-if in a folder named after the part it strengthens
        tag = "whatif_" + (re.sub(r"[^A-Za-z0-9]+", "_", _k.split(": ", 1)[-1])
                           .strip("_")[:48] or "part")
        tag = tag if tag not in tags else f"{tag}_{len(tags)}"
        tags.append(tag)
        whatifs[tag] = wi
        args += run_args(tag, merge_whatif(ov_raw, wi))
    res = _pool_runs(args, len(args), processes, progress, "what-if crush")
    items = []
    for tag, (k, (cnt, e)) in zip(tags, cands, strict=True):
        rb = sorted((r for r in res if r["tag"] == tag), key=lambda r: r["seed"])
        ch = [(b["Fu_kgf"] - a["Fu_kgf"]) / max(a["Fu_kgf"], 1e-9)
              for a, b in zip(base, rb, strict=True)]
        govs = [b.get("governing") or b["first_failure"] for b in rb]
        items.append(DiagnosisItem(
            part=k.split(": ", 1)[1], kind=e["kind"].replace("_", " "), seen=cnt,
            overrides=whatifs[tag], Fu_change=round(float(np.mean(ch)), 4),
            Fu_changes=[round(c, 4) for c in ch], stronger_with=int(sum(c > 0 for c in ch)),
            governing_after=max(set(govs), key=govs.count),
            runs=(str((Path(out_dir) / f"{tag}_seed_<n>").resolve()) if out_dir
                  else None)))
    items.sort(key=lambda it: -it.Fu_change)
    top = items[0]
    helps = top.Fu_change > 0.02 and top.stronger_with == n
    govs0 = [r.get("governing") or r["first_failure"] for r in base]
    rep = DiagnosisReport(
        file=path.name, fidelity=fidelity, n=n, Fu_base_kgf=[r["Fu_kgf"] for r in base],
        governing_base=max(set(govs0), key=govs0.count), candidates=items,
        limiting=top.part if helps else "none",
        verdict=(f"Strengthen {top.part} ({top.kind}) first: F_u {top.Fu_change:+.1%} "
                 f"(stronger with {top.stronger_with} of {n} seeds); "
                 + "; ".join(f"{it.part}: {it.Fu_change:+.1%}" for it in items[1:])
                 if helps else
                 "No single part limits the bridge: strengthening any one of them moves "
                 "the failure to another at about the same load ("
                 + "; ".join(f"{it.part}: {it.Fu_change:+.1%}" for it in items)
                 + "). Strengthen them together (one what-if with all of them), or look "
                   "for what they share (the load path they are on)."))
    f = shlex.quote(str(path.resolve()))
    ovs = _ov_arg(_ov_ref(overrides, ov_raw))
    wi_all = {}
    for it in (items[:1] if helps else items):
        wi_all = merge_whatif(wi_all, it.overrides)
    rep.next_steps = [NextStep(
        action="strengthen", parts=[it.part for it in (items[:1] if helps else items)],
        overrides=wi_all,
        why=("the measured limiting part: make it stronger in the CAD (glued area, a "
             "gusset, a deeper member) and compare the two files with ab" if helps else
             "strengthen these parts together: test the combined what-if (`overrides`) "
             "with ab on more seeds"),
        cli=f"bridgesim ab {f} {f}{ovs.replace(' -o ', ' --overrides-a ')} --overrides-b "
            + (shlex.quote(str(Path(out_dir).resolve() / "whatif.yaml")) if out_dir
               else "<your overrides plus `overrides`>") + " -n 3 --json")]
    _with_context(rep.next_steps, _context(material, sim_params, ruleset, units, up, span,
                                           fidelity=fidelity))
    if out_dir:
        import yaml

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "whatif.yaml").write_text(
            "# your overrides plus the what-if of diagnose\n"
            + yaml.safe_dump(merge_whatif(ov_raw, wi_all), sort_keys=False,
                             allow_unicode=True), encoding="utf-8")
        (Path(out_dir) / "diagnosis.json").write_text(rep.model_dump_json(indent=2),
                                                     encoding="utf-8")
    return rep


def _copies_key(e: dict) -> str:
    """A failure by its kind and its part(s) with their copies."""
    parts = e.get("parts") or [e["what"]]
    return f"{e['kind']}: " + " | ".join(_group(p) for p in parts)


def dump(model_obj) -> str:
    return json.dumps(model_obj.model_dump(mode="json"), indent=2)
