"""Typed, JSON-serialisable reports of the crush simulation (for people and AI agents).

Every command of ``bridgesim inspect / crush / compare / lab / calibrate / montecarlo``
returns one of these models, and ``--json`` prints it. ``bridgesim schema <name>`` prints
the JSON Schema, so an agent can rely on the field names and units. Units are in the
field names (``_mm``, ``_kgf``, ``_N``, ``_g``, ``_s``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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
    kind: Literal["bar", "shell", "block", "cable", "ignore"] = Field(
        description="bar = fibre beam member, shell = plate, block = rigid piece, "
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
    why: str = ""


class JointReport(BaseModel):
    a: str
    b: str
    kind: Literal["glue", "tie", "bonded"]
    area_mm2: float
    shear_capacity_N: float = Field(description="Glue area x tau_g: what a lap test of "
                                                "this joint would give in pure shear.")


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
    assumed_parameters: list[str] = Field(
        description="Material/simulation values still marked 'assumed' (not measured).")
    model: dict[str, Any] = Field(description="Size of the structural model.")
    warnings: list[str] = []


class EventOut(BaseModel):
    t_s: float
    jack_travel_mm: float
    load_N: float
    load_kgf: float
    kind: str = Field(description="member_fracture, member_shear, plate_crack, "
                                  "glue_damage, joint_failure, floss_break, criterion, "
                                  "instability")
    what: str
    detail: str = ""
    where_mm: list[float] | None = Field(None, description="Where it happened (centroid "
                                                           "of the part or glued patch)")


class Curve(BaseModel):
    jack_travel_mm: list[float]
    load_kgf: list[float]
    deflection_mm: list[float]
    joints_broken: list[int]
    members_broken: list[int]


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
                                                   "its peak; < 0.1 means the loading was "
                                                   "quasi-static.")
    settle_sag_mm: float
    notes: list[str] = []
    trustworthy: bool = Field(True, description="False if any of `reasons` applies: then "
                                                "treat F_u with suspicion")
    reasons: list[str] = []
    speed_mm_s: float = 0.0
    holds: int = Field(0, description="Times the jack was slowed to 2 % while something "
                                      "was breaking, until it stopped (quasi-static failure)")
    hold_time_s: float = 0.0
    rate_check: dict | None = Field(None, description="With --rate-check: the same crush "
                                    "at half the jack speed (speed_mm_s, Fu_kgf, change = "
                                    "relative change of F_u). |change| < 0.05 shows the "
                                    "simulated test is slow enough to stand for the real one.")


class CrushReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["crush"] = "crush"
    file: str
    sha256: str
    seed: int | None
    overrides: dict[str, Any]
    Fu_kgf: float = Field(description="Predicted ultimate load (rulebook §12.5), kgf.")
    Fu_N: float
    failure_mode: str = Field(description="collapse, deflection, contact, instability, "
                                          "none")
    failure_detail: str
    deflection_at_Fu_mm: float
    mass_kg: float
    efficiency_kgf_per_kg: float = Field(description="§12.6: Fu / mass")
    criteria: dict[str, Any] = Field(description="First time each §12.5 failure criterion "
                                                 "was met (null if never).")
    events: list[EventOut] = Field(description="First event of each kind per part (full log: "
                                               "events.csv)")
    n_events_total: int = 0
    Fu_first_joint_failure_kgf: float | None = Field(
        None, description="Jack load when the first glued joint fully separated (or a member "
                          "split in shear). §12.5 counts joint failure as failure, so a "
                          "strict judge could stop the test here.")
    warnings: list[str] = []
    failure_sequence: list[str] = Field(description="Plain-language order in which things "
                                                    "broke.")
    curve: Curve
    quality: Quality
    rules: RulesSummary
    questions: list[Question]
    assumed_parameters: list[str]
    fidelity: str = Field("standard", description="quick (screening), standard or fine")
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
    Fu_kgf: list[float]
    p05_kgf: float
    p50_kgf: float
    p95_kgf: float
    failure_modes: dict[str, int]
    recommended_prediction_kgf: float = Field(
        description="Value for the Load Prediction Form (§12.7): the median of the runs "
                    "weighted by 1/F_u, which minimises the expected prediction error "
                    "E = |F_u - F_p| / F_u.")
    fidelity: str = "standard"
    untrustworthy_runs: int = Field(0, description="Runs whose quality.trustworthy is "
                                                   "false")


class DesignRun(BaseModel):
    seed: int
    Fu_kgf: float
    mass_kg: float
    efficiency_kgf_per_kg: float
    first_failure: str
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
    governing_a: str = Field(description="Most frequent first failure of A")
    governing_b: str
    untrustworthy_runs: int = 0


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
    rel: float = Field(description="Each value is varied by -rel and +rel")
    Fu_base_kgf: float
    items: list[SensitivityItem] = Field(description="Most influential first")
    measure_first: list[str] = Field(description="Assumed values that move F_u most, with "
                                                 "how to measure them")
    note: str = ""


SCHEMAS = {"inspect": InspectReport, "crush": CrushReport, "reality_gap": RealityGap,
           "lab": LabReport, "montecarlo": MonteCarloReport, "question": Question,
           "design_comparison": DesignComparison,
           "sensitivity": SensitivityReport}
