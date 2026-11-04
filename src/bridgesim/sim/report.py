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
    kinetic_to_work_max: float = Field(description="Peak kinetic energy / work done by the "
                                                   "jack after 1 J; < 0.1 means "
                                                   "quasi-static.")
    settle_sag_mm: float
    notes: list[str] = []


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
    events: list[EventOut]
    failure_sequence: list[str] = Field(description="Plain-language order in which things "
                                                    "broke.")
    curve: Curve
    quality: Quality
    rules: RulesSummary
    questions: list[Question]
    assumed_parameters: list[str]
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
        description="Value for the Load Prediction Form (§12.7): the median, which "
                    "minimises the expected relative error.")


SCHEMAS = {"inspect": InspectReport, "crush": CrushReport, "reality_gap": RealityGap,
           "lab": LabReport, "montecarlo": MonteCarloReport, "question": Question}
