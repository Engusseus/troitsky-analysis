"""Typed, JSON-serialisable reports of the crush simulation (for people and AI agents).

Every command of ``bridgesim inspect / crush / compare / lab / calibrate / montecarlo``
returns one of these models, and ``--json`` prints it. ``bridgesim schema <name>`` prints
the JSON Schema, so an agent can rely on the field names and units. Units are in the
field names (``_mm``, ``_kgf``, ``_N``, ``_g``, ``_s``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

SCHEMA_VERSION = "1.0"


class Question(BaseModel):
    """Something the file does not say and the simulation had to assume. Answer it by
    adding ``answer_with`` (edited) to the overrides YAML and running again."""

    id: str = Field(description="Stable identifier, e.g. 'layup:Body 21531'.")
    topic: Literal["material", "connection", "geometry", "rules", "loading", "design"]
    body: str | None = None
    question: str
    assumed: str = Field(description="What the simulation assumed meanwhile.")
    answer_with: dict[str, Any] = Field(
        default_factory=dict, description="Overrides YAML fragment that answers it.")
    severity: Literal["info", "check", "important"] = "check"


class BodyReport(BaseModel):
    name: str
    centroid_mm: list[float] = Field(default_factory=list, description="X along the span, "
                                                                       "Y up, Z across")
    bbox_min_mm: list[float] = Field(default_factory=list)
    bbox_max_mm: list[float] = Field(default_factory=list)
    axis: list[float] | None = Field(None, description="Long axis (bars) or plate normal "
                                                       "(shells), unit vector")
    kind: Literal["bar", "shell", "frame", "block", "cable", "ignore"] = Field(
        description="bar = fibre beam member, shell = plate, frame = plate of narrow arms "
                    "modelled as members along its arms, block = rigid piece, "
                    "cable = floss, ignore = left out.")
    layup: str
    layup_source: Literal["name", "override", "default"]
    mass_g: float
    volume_mm3: float
    length_mm: float = 0.0
    section_mm: tuple[float, float] = (0.0, 0.0)
    thickness_mm: float = 0.0
    E_eff_MPa: float = 0.0
    f_t_eff_MPa: float = 0.0
    f_c_eff_MPa: float = 0.0
    butt_zones_mm: list[float] = Field([], description=(
        "A laminated member's butt joints that stay open (their gap does not close: "
        "rounded stick ends, butt_gap_mm), at their places: each a short element of the "
        "net section, centred this far from the member's end (butt_stagger_mm and a "
        "seeded offset, or the designer's butt_at). Empty: joints that bear, the member's "
        "factors (E_eff, f_t_eff, f_c_eff) over its length"))
    why: str = ""


class JointReport(BaseModel):
    a: str
    b: str
    kind: Literal["glue", "tie", "bonded", "solid", "dry"] = Field(description=(
        "glue (a glued joint), tie (a floss knot), bonded (bodies that overlap in the "
        "CAD), solid (two bodies of one piece of wood, glue override 'solid': the face "
        "between them breaks at the wood's shear strength f_v / 1.5 averaged, as a "
        "notched end, and its tension across the grain), dry (two faces drawn apart, "
        "0.8-3 mm, or left unglued by the overrides: no glue; they bear on each other, "
        "with friction, once the gap has closed)"))
    area_mm2: float
    lamination: bool = Field(False, description=(
        "Two plates glued face to face (a deck's plies drawn as separate bodies): glued "
        "as the plies inside one laminated body, at glue.interlaminar_MPa and "
        "glue.lamination_K_s_N_per_mm3, with no size effect"))
    shear_capacity_N: float = Field(description=(
        "What the joint's glue points carry in pure shear, as the crush holds them: the "
        "sum of their areas x their strengths (tau_g times the lap size factor, the "
        "crossed-lap factor, the end-grain share, the fit-up gaps, the seed's scatter and "
        "glue_factor; glue.interlaminar_MPa for a lamination, f_v / 1.5 for a solid "
        "face), dry points left out"))
    exposed_perimeter_mm: float = Field(0.0, description=(
        "The length of the joint's outline where one part's face goes on past the "
        "other's edge, where squeeze-out can form a fillet (glue.fillet_mm: the joint "
        "then holds as a lap of area + fillet x this); flush edges not counted"))


class RuleLine(BaseModel):
    section: str
    title: str
    passed: bool | None
    measured: str = ""
    penalty: float = 0.0
    bans: list[str] = []
    cause: str = ""
    ambiguous: bool = False


class RulesSummary(BaseModel):
    total_penalty: float
    bans: list[str]
    lines: list[RuleLine]
    measurements: dict[str, float]


class NextStep(BaseModel):
    """One thing to do next, most useful first: the loop an agent drives."""
    action: Literal["ask_designer", "fix_cad", "screen", "crush", "rerun", "strengthen",
                    "diagnose", "confirm", "measure", "predict", "compare"] = Field(
                        description=(
        "ask_designer: put the open questions to the designer; fix_cad: change the "
        "SolidWorks model (a failing rule); screen: the linear pre-screen (seconds: "
        "stiffness and the most loaded parts); crush: run the crushing test; rerun: the "
        "result is not trustworthy, run it again as `cli` says; strengthen: the parts "
        "that broke at the peak, then check the change with `ab`; diagnose: measure which "
        "of the parts that broke near the peak limits the bridge; confirm: check the "
        "answer at a finer fidelity; measure: test sticks and glue (placeholders); "
        "predict: the value for the Load Prediction Form; compare: after a real test"))
    why: str
    parts: list[str] = Field([], description="Body or joint names concerned (as inspect "
                                             "lists them)")
    where_mm: list[list[float]] = Field([], description="Where each of those parts failed "
                                                        "(x, y, z), one per part, in the "
                                                        "same order")
    ids: list[str] = Field([], description="Question ids concerned")
    cli: str | None = Field(None, description="The command to run (MCP: the matching tool)")
    overrides: dict[str, Any] = Field({}, description=(
        "A what-if to try before changing the CAD (strengthen): merge it into a copy of "
        "your overrides, save it as the file the `cli` names (whatif.yaml), and run it"))


class InspectReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["inspect"] = "inspect"
    file: str
    sha256: str
    n_bodies: int
    mass_kg: float
    bodies: list[BodyReport]
    joints: list[JointReport]
    loading: dict[str, Any]
    rules: RulesSummary
    questions: list[Question]
    acknowledged_questions: list[str] = Field(
        [], description="Question ids the overrides acknowledged (acknowledge:), not asked")
    assumed_parameters: list[str] = Field(
        description="Material/simulation values still marked 'assumed' (not measured).")
    model: dict[str, Any] = Field(description="Size of the structural model.")
    warnings: list[str] = []
    next_steps: list[NextStep] = Field([], description="What to do next, most useful first")


EventKind = Literal["member_fracture", "member_shear", "member_crushing", "member_erosion",
                    "plate_crack", "bearing_crushing",
                    "glue_damage",
                    "joint_cracking", "joint_failure", "floss_break", "criterion",
                    "instability", "numerical"]
FailureMode = Literal["collapse", "deflection", "contact", "instability", "none"]


class EventOut(BaseModel):
    t_s: float
    jack_travel_mm: float
    load_N: float
    load_kgf: float
    kind: EventKind = Field(description=(
        "member_fracture (fibres cracked through), member_shear (split along the glue "
        "lines or grain), member_crushing (fibres folding in compression past f_c), "
        "member_erosion (an element bent or twisted past 1 rad removed, its section not "
        "cracked through), "
        "plate_crack, bearing_crushing (the wood under a face pressed together, a member "
        "end bearing on a knee or a tenon on its sill, has crushed by 0.1 mm on average: "
        "the face bears more than its grain can take), "
        "glue_damage (a joint's glue starts to soften), "
        "joint_cracking (>= 30 % of a joint's glued area has let go), joint_failure "
        "(>= 90 %: separated), floss_break, criterion (§12.5), instability, numerical"))
    what: str = Field(description="The part, or 'A | B' for a glued joint")
    parts: list[str] = Field([], description="Body names involved (two for a joint)")
    detail: str = ""
    where_mm: list[float] | None = Field(None, description="Where it happened (centroid "
                                                           "of the part or glued patch)")
    fraction_separated: float | None = Field(
        None, description="Joints: share of the glued area that has let go")

    def model_post_init(self, _ctx) -> None:
        if not self.parts and self.kind not in ("criterion", "instability", "numerical"):
            self.parts = [p.strip() for p in self.what.split(" | ")]


class CriterionHit(BaseModel):
    t_s: float
    jack_travel_mm: float | None = None
    load_kgf: float | None = None
    what: str = ""


class Criteria(BaseModel):
    """When each §12.5 failure criterion was first met (null: never)."""

    deflection: CriterionHit | None = None
    contact: CriterionHit | None = None
    joint_or_shear: CriterionHit | None = None
    instability: CriterionHit | None = None


class Curve(BaseModel):
    jack_travel_mm: list[float]
    load_kgf: list[float] = Field(description=(
        "The reported load: the jack force smoothed along the travel between breaks "
        "(stress waves removed, no lag, the peak kept where a break cuts the curve)"))
    deflection_mm: list[float]
    joints_broken: list[int]
    members_broken: list[int]
    load_raw_kgf: list[float] = Field(default_factory=list, description=(
        "The jack force as computed, before smoothing (stress waves included)"))


class Quality(BaseModel):
    """How much to trust the numbers of this run (numerics, not physics)."""

    dt_s: float
    steps: int
    wall_s: float
    added_mass_fraction: float = Field(description="Mass added for numerical stability "
                                                   "(<= 0.05 is fine).")
    kinetic_to_work_max: float = Field(description="Peak kinetic energy of the bridge "
                                                   "(debris excluded) / work done by the "
                                                   "jack, while the load rises to 95 % of "
                                                   "its peak; < 0.05 means the loading was "
                                                   "quasi-static.")
    settle_sag_mm: float
    notes: list[str] = []
    trustworthy: bool = Field(True, description="False if any of `reasons` applies: then "
                                                "treat F_u with suspicion")
    reasons: list[str] = []
    speed_mm_s: float = Field(0.0, description=(
        "The simulated jack speed: the fidelity's (quick 200, standard 100, fine 50 mm/s) "
        "for a bridge whose lowest mode under the plate rings in up to 10 ms; for a softer "
        "or lighter one (fundamental_period_ms), the speed at the start, slower in "
        "proportion to its period (at most 16x), rising back toward the fidelity's once "
        "the jack travel passes 6 mm x period / 10 ms"))
    fundamental_period_ms: float = Field(0.0, description=(
        "Period of the bridge's lowest mode under the loading plate (the plate's mass and "
        "half the bridge's on its linear stiffness). A load drop must last this long to "
        "count as the moment of failure"))
    holds: int = Field(0, description="Times the jack was slowed to 2 % while something "
                                      "was breaking, until it stopped (quasi-static failure)")
    hold_time_s: float = 0.0
    min_fidelity: str = Field("quick", description=(
        "The coarsest fidelity whose mesh is fine enough for the members that fail "
        "(elements at most 3x their depth, 1.5x below 10 mm): run diagnose, montecarlo, "
        "sensitivity and ab of this bridge at least there"))
    rate_check: dict | None = Field(None, description=(
        "With --rate-check: the same crush at half the jack speed (speed_mm_s, Fu_kgf, "
        "change = relative change of F_u, governing). |change| < 0.05 shows the simulated "
        "test is slow enough to stand for the real one. Fu_extrapolated_kgf = (4 F(v/2) - "
        "F(v)) / 3: F_u extrapolated to a jack at rest along the square of the speed (the "
        "inertia effects grow with the kinetic energy; on Bridge_v10 F_u fell 2.4 % from "
        "200 to 100 mm/s and 0.5 % from 100 to 50), the closest the simulation gets to the "
        "real test's mm/min"))


class CrushReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["crush"] = "crush"
    file: str
    sha256: str
    seed: int | None
    overrides: dict[str, Any]
    inputs_sha256: str = Field("", description=(
        "Fingerprint of everything the run depends on: the CAD, the overrides as read, "
        "the material, the simulation parameters, seed, fidelity and the code"))
    code_sha: str = Field("", description=(
        "Fingerprint of the bridgesim code alone (every module): two runs with the same "
        "code_sha ran the same code, whatever else changed"))
    values_sha: str = Field("", description=(
        "Fingerprint of the material and simulation values as resolved (the yaml files "
        "and the overrides' params): with code_sha, what physics the run had (the "
        "adversarial review of round 26: code_sha alone left the yaml out)"))
    Fu_kgf: float = Field(description="Predicted ultimate load (rulebook §12.5), kgf.")
    Fu_N: float
    failure_mode: FailureMode = Field(description="collapse (the load fell 30 % below "
                                                  "its peak), deflection (> 50 mm), contact "
                                                  "(with the platform other than at the "
                                                  "supports, or with the press uprights), "
                                                  "instability (collapsed under "
                                                  "self-weight, F_u = 0), "
                                                  "none (no failure within the jack travel)")
    failure_detail: str
    governing: str = Field("", description=(
        "The break that started the fall of the load (the reason the bridge failed): the "
        "first break from where the load last stood within 5 % of F_u before the "
        "failure call (breaks before it were survived). '<kind>: <part>', 'glue giving "
        "way: A | B', 'buckling or sway', or the failure mode"))
    competing: list[str] = Field([], description=(
        "Other parts that broke within 10 % of F_u before the failure call: on another "
        "seed, mesh or measured material the failure can start at one of them (diagnose "
        "measures which one the load depends on). Empty: the governing failure stands "
        "alone"))
    deflection_at_Fu_mm: float
    test_time_at_Fu_s: float = Field(0.0, description=(
        "When the real test would reach F_u (s): the jack's travel to the peak at "
        "crusher.test_rate_mm_per_min. The glue's strength is taken at this time to "
        "failure (glue.rate_exponent, against its coupons' glue.coupon_time_s)"))
    mass_kg: float
    efficiency_kgf_per_kg: float = Field(description="§12.6: Fu / mass")
    criteria: Criteria = Field(description="First time each §12.5 failure criterion "
                                           "was met (null if never).")
    failures: list[EventOut] = Field([], description=(
        "What broke, in order (joint cracking or separation, member and plate failures, "
        "floss), first time per part, up to the collapse: the structured form of "
        "failure_sequence"))
    events: list[EventOut] = Field(description="First event of each kind per part (full log: "
                                               "events.csv)")
    n_events_total: int = 0
    Fu_first_joint_failure_kgf: float | None = Field(
        None, description="The highest jack load up to the moment the first glued joint "
                          "separated (>= 90 % of its area let go) or a member split in "
                          "shear: what a strict judge who stops the test there reads (§12.5 "
                          "counts joint failure as failure). It equals F_u when the first "
                          "joint goes at or after the peak.")
    warnings: list[str] = []
    failure_sequence: list[str] = Field(description="Plain-language order in which things "
                                                    "broke.")
    curve: Curve
    quality: Quality
    rules: RulesSummary
    questions: list[Question]
    acknowledged_questions: list[str] = Field(
        [], description="Question ids the overrides acknowledged (acknowledge:), not asked")
    assumed_parameters: list[str]
    fidelity: str = Field("standard", description="quick (screening), standard or fine")
    next_steps: list[NextStep] = Field([], description="What to do next, most useful first")
    artifacts: dict[str, str] = {}


class GapMetric(BaseModel):
    name: str
    simulated: float | str
    measured: float | str
    error: float | None = Field(description="Relative error (sim - test) / test, or None")
    unit: str = ""
    meaning: str = ""


class RealityGap(BaseModel):
    """How far a simulation is from a physical test of the same thing."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["reality_gap"] = "reality_gap"
    subject: str
    metrics: list[GapMetric]
    score_0_100: float = Field(description="100 = indistinguishable from the test. "
                                           "100 x (1 - weighted mean |error|), floored at 0.")
    competition_prediction_error: float | None = Field(
        None, description="§12.7 E = |Fu - Fu,p| / Fu if the test reached failure.")
    advice: list[str] = []


class LabReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["lab"] = "lab"
    tests: list[dict[str, Any]]
    gaps: list[RealityGap] = []


class MonteCarloReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["montecarlo"] = "montecarlo"
    file: str
    n: int
    provenance: dict[str, str] = Field({}, description=(
        "What the prediction was made from, to show later that it was made before a "
        "test (keep montecarlo.json, or these hashes, with a date the test cannot have "
        "influenced: a commit, an e-mail to the organisers, a timestamping service): "
        "the CAD file's sha256, the overrides' and the material and simulation values' "
        "sha256 (canonical JSON), the code (code_sha), the fidelity and the UTC time the "
        "report was made (the ambitious review of round 26)."))
    Fu_kgf: list[float]
    p05_kgf: float
    p50_kgf: float
    p95_kgf: float
    failure_modes: dict[str, int]
    governing_parts: dict[str, int] = Field({}, description=(
        "Where the collapse started, counted over the runs by part (every copy of a part "
        "together, near and far, left and right): the failure site as a distribution. "
        "Several entries with similar counts mean failure sites competing within the "
        "scatter: strengthen them together (diagnose ranks them)"))
    recommended_prediction_kgf: float = Field(
        description="Value for the Load Prediction Form (§12.7): the median weighted by "
                    "1/F_u, which minimises the expected prediction error "
                    "E = |F_u - F_p| / F_u, over the runs times the mesh and model-form "
                    "error (uncertainty.*), which pulls it below the runs' own value "
                    "(recommended_prediction_runs_kgf)")
    mesh_bias: float = Field(0.0, description=(
        "The mesh's and the simulated jack speed's bias at this fidelity against fine "
        "with a jack at rest, taken out of the prediction and the total range "
        "(uncertainty.mesh_bias_*); the runs are listed as they came"))
    recommended_prediction_runs_kgf: float | None = Field(
        None, description="The same over the runs alone (no mesh or model-form error)")
    prediction_by_model_cov: dict[str, float] = Field({}, description=(
        "recommended_prediction_kgf with other model-form errors (uncertainty.model_cov, "
        "a judgement until a test of your own): 0.15 a model calibrated on a test, the "
        "shipped value, and 0.76 the scatter of the published white-glue tests about the "
        "model (validation README 3.14). The wider the error, the lower the value that "
        "minimises the expected §12.7 error"))
    prediction_if_biased_kgf: float | None = Field(None, description=(
        "recommended_prediction_kgf if the model reads this bridge as low as it reads the "
        "published white-glue tests on average (uncertainty.published_bias: their runs' "
        "median over the test), the runs divided by it: what the only external evidence "
        "would suggest, not applied (small trusses, outside a team's bridge's range)"))
    slowed_runs: int = Field(0, description=(
        "Runs flagged as not quasi-static that were crushed again at half the jack speed "
        "(their first F_u in each run's slowed_from_kgf)"))
    recommended_interval_kgf: list[float] = Field([], description=(
        "5 % and 95 % points of recommended_prediction_kgf over bootstrap resamples of "
        "the runs: how far other runs could move it (wide with few runs or two clusters "
        "of failure loads: run more). Not the uncertainty of F_u: that is the total "
        "range (total_p05_kgf, total_p95_kgf)"))
    published_check: str = Field("", description=(
        "How far this model's recommended prediction was from the published white-glue "
        "truss tests it was rebuilt for (the §12.7 error |F - p| / F, validation README "
        "3.14): what a prediction like this one missed by, on small trusses outside a "
        "team's bridge's range (the adversarial review of round 26)"))
    Fu_untrustworthy_kgf: list[float] = Field([], description=(
        "F_u of the runs whose quality.trustworthy is false, left out of Fu_kgf, the "
        "percentiles and the prediction (unless fewer than half the runs are "
        "trustworthy)"))
    fidelity: str = "standard"
    Fu_strict_kgf: list[float] = Field([], description=(
        "The same runs judged strictly: the load when the first glued joint separated or a "
        "member split (§12.5 'shear, including joint failure'), if that came before the "
        "peak"))
    p50_strict_kgf: float | None = None
    recommended_prediction_strict_kgf: float | None = Field(
        None, description="The Load Prediction Form value if the judges stop the test at "
                          "the first visible joint failure; recommended_prediction_kgf is "
                          "for judges who stop at the collapse (the usual practice)")
    untrustworthy_runs: int = Field(0, description="Runs whose quality.trustworthy is "
                                                   "false")
    total_p05_kgf: float | None = Field(None, description=(
        "The 5 % point of F_u including the mesh error and the model-form error "
        "(uncertainty.mesh_cov, uncertainty.model_cov in materials/simulation.yaml) on top "
        "of the runs' spread: the range to quote until a validation test narrows it"))
    total_p95_kgf: float | None = Field(None, description="The 95 % point of the same")
    min_kgf: float | None = Field(None, description="Lowest F_u of the runs")
    max_kgf: float | None = Field(None, description="Highest F_u of the runs (with fewer "
                                                    "than 20 runs, read these rather than "
                                                    "p05/p95)")
    assumed_parameters: list[str] = Field([], description="Material values still "
                                                         "placeholders")
    sampled_placeholders: list[str] = Field([], description=(
        "Placeholder values drawn per run (--sample-placeholders), so the spread includes "
        "them"))
    note: str = Field("", description="What the spread does and does not cover")


class DesignRun(BaseModel):
    seed: int
    Fu_kgf: float
    mass_kg: float
    efficiency_kgf_per_kg: float
    first_failure: str = Field(description="The first part that let go (often long before "
                                           "the peak)")
    governing: str = Field("", description="The break that started the fall of the load: "
                                           "why this run failed")
    trustworthy: bool


class DesignComparison(BaseModel):
    """Paired comparison of two designs (or one design with two sets of overrides):
    both are crushed with the same seeds, so each pair shares imperfections and material
    scatter wherever the designs share bodies, and the difference is the design's."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["design_comparison"] = "design_comparison"
    a: str
    b: str
    n: int
    fidelity: str
    runs_a: list[DesignRun]
    runs_b: list[DesignRun]
    Fu_mean_a_kgf: float
    Fu_mean_b_kgf: float
    Fu_change: float = Field(description="Mean of (F_u,B - F_u,A) / F_u,A over the seeds")
    Fu_change_ci90: tuple[float, float] = Field(
        description="90 % confidence interval of Fu_change (paired t; wide with few seeds)")
    efficiency_change: float = Field(description="Same for F_u / mass (§12.6)")
    efficiency_change_ci90: tuple[float, float]
    significant: bool = Field(description="True if the F_u confidence interval excludes 0")
    verdict: str
    governing_a: str = Field(description="Most frequent governing failure of A (what broke "
                                         "at the peak), by part with its copies")
    governing_b: str = Field(description="The same for B")
    failure_moved: list[str] = Field([], description=(
        "Seeds where B failed somewhere else than A ('seed 2: A <failure> -> B <failure>'): "
        "the change moved the failure to another part, which then sets F_u"))
    untrustworthy_runs: int = 0
    note: str = Field("", description=(
        "Caveats: bodies of B with no namesake in A (renamed), whose seeded scatter is "
        "drawn afresh, so part of the difference may be luck"))


class SensitivityItem(BaseModel):
    param: str = Field(description="Overrides key (params:) of the value varied")
    meaning: str
    how_to_measure: str
    source: str = Field(description="assumed or measured")
    base: float
    Fu_low_kgf: float = Field(description="F_u with the value lowered by `rel`")
    Fu_high_kgf: float
    elasticity: float = Field(description="Relative change of F_u per relative change of "
                                          "the value (1 = proportional)")


class SensitivityReport(BaseModel):
    """How much each material value moves F_u: what to measure first."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["sensitivity"] = "sensitivity"
    file: str
    fidelity: str
    seed: int | None
    seeds: int = Field(1, description="F_u is averaged over seeds seed, seed + 1, ...")
    rel: float = Field(description="Each value is varied by -rel and +rel")
    Fu_base_kgf: float
    items: list[SensitivityItem] = Field(description="Most influential first")
    measure_first: list[str] = Field(description="Assumed values that move F_u most, with "
                                                 "how to measure them")
    note: str = ""


class CalibrationStep(BaseModel):
    stiffness_factor: float
    strength_factor: float
    Fu_kgf: float
    stiffness_kgf_per_mm: float | None
    Fu_error: float = Field(description="(simulated - test) / test")
    stiffness_error: float | None = None
    first_failure: str = Field("", description="What governed (most often over the seeds)")
    Fu_kgf_runs: list[float] = Field([], description="F_u of each seed; Fu_kgf is their "
                                                     "median")


class BridgeCalibration(BaseModel):
    """Global stiffness and strength factors fitted so the simulated crush of a bridge
    reproduces a physical crush test of it (model updating, not validation)."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["bridge_calibration"] = "bridge_calibration"
    file: str
    test: str
    fidelity: str
    seed: int | None = Field(description="The seed, when fitted to one (seeds=1)")
    seeds: list[int | None] = Field([], description="Seeds crushed at each step: the fit is "
                                                    "to their median F_u and stiffness")
    test_Fu_kgf: float
    test_stiffness_kgf_per_mm: float | None
    test_failure: str = ""
    stiffness_factor: float = Field(description="Multiplies E, G, E_T and the glue "
                                                "stiffness")
    strength_factor: float = Field(description="Multiplies every strength: f_t, f_c, f_v, "
                                               "f_b, tau_g, f_c90, f_t90")
    converged: bool
    steps: list[CalibrationStep]
    overrides: dict[str, Any] = Field(description="Your overrides with the fitted params: "
                                                  "use them for the next crushes")
    note: str = ""


class ScreenLimit(BaseModel):
    part: str = Field(description="A body, or 'A | B' for a glued joint")
    mode: str = Field(description="glue (the joint's resultant against its capacity), "
                                  "bearing (the peak pressure on a face pressed together, "
                                  "glued or dry, against what the wood under it takes "
                                  "across or along its grain: where it starts to crush), "
                                  "tension, compression, shear, buckling (Euler, the "
                                  "member's length, pinned), plate fibre tension, plate "
                                  "fibre crushing, plate across-grain tension, plate "
                                  "across-grain crushing, plate shear")
    utilisation: float = Field(description="At the reference load (1 = at its limit)")
    load_kgf: float = Field(description="Plate load at which it reaches its limit if the "
                                        "bridge stayed linear")
    where_mm: list[float]


class ScreenReport(BaseModel):
    """Linear pre-screen (seconds): the elastic response to the plate load, and how
    close each joint, member and plate is to its limit. Ranks weak spots and compares
    design variants; F_u needs crush (load redistributes after the first joint goes)."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["screen"] = "screen"
    file: str
    sha256: str
    overrides: dict[str, Any] = {}
    load_kgf: float = Field(description="Reference plate load")
    stiffness_kgf_per_mm: float = Field(description="Plate load / plate displacement")
    first_limit_kgf: float = Field(description="Plate load at which the most utilised "
                                               "part reaches its limit: where damage "
                                               "starts (linear), far below F_u when the "
                                               "load can find another path")
    first_limit: str = Field(description="That part and how it fails")
    limits: list[ScreenLimit] = Field(description="Most utilised first (up to 25)")
    n_limits: int
    mass_kg: float
    modes: list[dict] = Field([], description=(
        "The lowest natural frequencies of the bridge on its supports with the loading "
        "plate on its deck (frequency_hz) and the parts that move most in each: what a "
        "tap test of the built bridge measures (a phone's accelerometer app on a node, a "
        "light tap). Measured frequencies against these check the sticks' and the glue's "
        "stiffness and the supports before the crush: the stick modulus moves every "
        "mode, the glue stiffness the sway modes most, the supports the lowest"))
    wall_s: float
    note: str = ""
    warnings: list[str] = []
    next_steps: list[NextStep] = []


class DiagnosisItem(BaseModel):
    part: str = Field(description="A part (or 'A | B' joint) that broke near the peak, "
                                  "with its copies")
    kind: str = Field(description="How it broke (joint failure, member shear, ...)")
    seen: int = Field(description="In how many of the base runs it broke near the peak")
    overrides: dict[str, Any] = Field(description="The what-if that strengthened it "
                                                  "(glue x1.5 for a joint, section x1.25 "
                                                  "for a member or plate)")
    Fu_change: float = Field(description="Mean paired change of F_u when it is "
                                         "strengthened")
    Fu_changes: list[float] = Field(description="Per seed")
    stronger_with: int = Field(description="Seeds where F_u rose")
    governing_after: str = Field(description="What governs once it is strengthened (most "
                                             "frequent)")
    runs: str | None = Field(None, description="With --out: the crushes of this what-if "
                                               "(<n>: the seed); the base ones are "
                                               "base_seed_<n>")


class DiagnosisReport(BaseModel):
    """What really limits the bridge, measured rather than read off one run: each part
    that broke near the peak is strengthened in turn on the same seeds, and the parts
    are ranked by how much that raises F_u. A part whose strengthening changes nothing
    is not what limits the bridge, even if it broke first."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["diagnosis"] = "diagnosis"
    file: str
    fidelity: str
    n: int = Field(description="Seeds (1..n) of every run")
    Fu_base_kgf: list[float]
    governing_base: str = Field(description="Most frequent governing failure of the base")
    candidates: list[DiagnosisItem] = Field(description="Most effective first")
    limiting: str = Field(description="The part to strengthen first, or 'none' when no "
                                      "single change helps")
    verdict: str
    next_steps: list[NextStep] = []


# ------------------------------------------------------------ what the MCP tools return


class BriefFailure(BaseModel):
    """A failure in a crush brief (an EventOut without its time, newtons and parts)."""
    model_config = ConfigDict(extra="allow")

    jack_travel_mm: float
    load_kgf: float
    kind: str
    what: str = Field(description="The part, or 'A | B' for a glued joint")
    detail: str = ""
    where_mm: list[float] | None = None
    fraction_separated: float | None = None


def _brief_model(name: str, full: type[BaseModel], drop: set[str], kinds: tuple[str, str],
                 extra: dict, doc: str) -> type[BaseModel]:
    """A report's brief form: the full report's fields minus ``drop`` (left optional, so
    the full report validates too), kind either form, plus ``extra``."""
    fields: dict = {}
    for k, f in full.model_fields.items():
        if k == "kind":
            continue
        if k in drop:
            fields[k] = (f.annotation | None, Field(None, description=f.description))
        else:
            fields[k] = (f.annotation, f)
    fields["kind"] = (Literal[kinds], Field(kinds[1]))  # type: ignore[valid-type]
    fields.update(extra)
    return create_model(name, __doc__=doc, __config__=ConfigDict(extra="allow"), **fields)


CrushBrief = _brief_model(
    "CrushBrief", CrushReport, {"curve", "events", "failure_sequence", "assumed_parameters"},
    ("crush", "crush_brief"),
    {"failures": (list[BriefFailure], Field([], description="The first failures (10 in the "
                                                           "brief) in order")),
     "n_failures_total": (int | None, None),
     "n_assumed_parameters": (int | None, None),
     "n_questions": (dict[str, int] | None, Field(None, description="Open questions by "
                                                                    "severity")),
     "peak": (dict[str, float] | None, Field(None, description="Where F_u was reached: "
                                                               "load, jack travel, "
                                                               "deflection"))},
    "A crush report cut to what an agent needs for its next step (crush --brief, the MCP "
    "default): no curve, event log or failure sequence; the first failures; the important "
    "questions in full and the others counted; next steps. crush.json has everything.")

InspectBrief = _brief_model(
    "InspectBrief", InspectReport, {"bodies", "joints"}, ("inspect", "inspect_brief"),
    {"bodies_by_kind": (dict[str, int] | None, None), "n_joints": (int | None, None)},
    "An inspect report without the per-body and per-joint lists (inspect --brief, the MCP "
    "default): counts, questions, failed rules, warnings, next steps.")


class JobStarted(BaseModel):
    status: Literal["running"]
    out_dir: str
    poll: str = ""


class JobStatus(BaseModel):
    """A background job, read from its folder (works from any server, a restarted one
    too)."""
    model_config = ConfigDict(extra="allow")

    status: Literal["queued", "running", "done", "failed", "cancelled", "unknown",
                    "finished already"]
    progress: float | None = None
    message: str | None = None
    report: dict[str, Any] | None = Field(None, description="The job's report when done "
                                                            "(its kind says which)")
    error: dict[str, Any] | None = Field(None, description="The JSON error object when "
                                                           "failed")
    out_dir: str | None = None


class MaterialCalibration(BaseModel):
    material: str = Field(description="The material YAML written (pass it as `material`)")
    changes: list[str]
    overrides_yaml: str = Field("", description="The specimens' scatter as params: to put "
                                                "in overrides_yaml (empty without 3+ "
                                                "specimens of a kind)")


SCHEMAS = {"inspect": InspectReport, "crush": CrushReport, "reality_gap": RealityGap,
           "lab": LabReport, "montecarlo": MonteCarloReport, "question": Question,
           "design_comparison": DesignComparison,
           "sensitivity": SensitivityReport, "bridge_calibration": BridgeCalibration,
           "screen": ScreenReport, "crush_brief": CrushBrief, "inspect_brief": InspectBrief,
           "job_status": JobStatus, "diagnosis": DiagnosisReport}
