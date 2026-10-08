"""Evaluate a rules YAML file (e.g. ``rules/troitsky_2027.yaml``) against a bridge.

Values are rounded as judges do (lengths to the nearest mm, mass to the nearest 0.01 kg,
half-up) before matching inclusive integer bands; the first matching band wins.
Every result cites its rulebook section. Nothing is clamped or hidden: a failing check
always appears with its penalty, bans and note.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bridgesim.mass import bridge_mass
from bridgesim.measure import Measurements, measure
from bridgesim.paths import data_dir
from bridgesim.schema import Bridge, Material
from bridgesim.textsafe import has_control_chars


def _plain_text(v: str) -> str:
    if has_control_chars(v, allow="\n"):
        raise ValueError("must not contain control characters")
    return v


class Band(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min: float | None = None
    max: float | None = None
    penalty: float = Field(0, ge=0)
    bans: list[str] = Field(default_factory=list)

    _bans = field_validator("bans")(lambda cls, v: [_plain_text(b) for b in v])

    def matches(self, v: float) -> bool:
        return (self.min is None or v >= self.min) and (self.max is None or v <= self.max)


class Consequence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    penalty: float = Field(0, ge=0)
    bans: list[str] = Field(default_factory=list)
    disqualification: bool = False

    _bans = field_validator("bans")(lambda cls, v: [_plain_text(b) for b in v])


class Steps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    free_up_to: float = Field(ge=0, allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)
    points_per_step: float = Field(ge=0, allow_inf_nan=False)
    cap: float = Field(gt=0, allow_inf_nan=False)
    over_cap_penalty: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> Steps:
        if self.cap <= self.free_up_to:
            raise ValueError("steps.cap must be greater than steps.free_up_to")
        if round(self.step * 100) < 1:
            raise ValueError("steps.step must be at least 0.01 (penalties use centigrams)")
        return self


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    section: str
    title: str
    type: Literal["bands", "check", "steps", "info"]
    measures: list[str] = Field(default_factory=list)
    measure: str | None = None
    combine: Literal["worst"] = "worst"
    show: list[str] = Field(default_factory=list)
    unit: str = ""
    limit: str = ""
    bands: list[Band] = Field(default_factory=list)
    fail: Consequence = Field(default_factory=Consequence)
    steps: Steps | None = None
    ambiguous: bool = False
    note: str = ""

    _text = field_validator("key", "section", "title", "unit", "limit", "note")(
        lambda cls, v: _plain_text(v))


class Crushing(BaseModel):
    """The competition's crushing test (§12.5)."""

    model_config = ConfigDict(extra="forbid")
    plate_length_mm: float = Field(200.0, gt=0, allow_inf_nan=False)
    plate_width_mm: float = Field(90.0, gt=0, allow_inf_nan=False)
    deflection_limit_mm: float = Field(50.0, gt=0, allow_inf_nan=False)


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    rulebook: str
    year: int
    rounding: dict[str, float]
    constants: dict[str, float] = Field(default_factory=dict)
    crushing: Crushing = Field(default_factory=Crushing)
    bans_meaning: dict[str, str] = Field(default_factory=dict)
    rules: list[Rule] = Field(max_length=500)

    _text = field_validator("name", "rulebook")(lambda cls, v: _plain_text(v))

    @field_validator("rounding", "constants")
    @classmethod
    def _positive_finite(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: x for k, x in v.items() if not (math.isfinite(x) and x > 0)}
        if bad:
            raise ValueError(f"values must be positive and finite: {bad}")
        return v

    @classmethod
    def from_yaml_str(cls, text: str) -> RuleSet:
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def load(cls, ref: str | Path = "troitsky_2027") -> RuleSet:
        path = Path(ref)
        if path.suffix not in (".yaml", ".yml"):
            path = data_dir("rules") / f"{ref}.yaml"
        return cls.from_yaml_str(path.read_text(encoding="utf-8"))


class RuleResult(BaseModel):
    key: str
    section: str
    title: str
    measured: float | None = None
    measured_text: str = ""
    limit: str = ""
    passed: bool | None = None  # None = informational / not evaluated
    penalty: float = 0
    bans: list[str] = Field(default_factory=list)
    disqualification: bool = False

    _bans = field_validator("bans")(lambda cls, v: [_plain_text(b) for b in v])
    ambiguous: bool = False
    note: str = ""


class RulesReport(BaseModel):
    ruleset: str
    results: list[RuleResult]

    @property
    def checked(self) -> list[RuleResult]:
        return [r for r in self.results if r.passed is not None]

    @property
    def info(self) -> list[RuleResult]:
        return [r for r in self.results if r.passed is None]

    @property
    def total_penalty(self) -> float:
        return sum(r.penalty for r in self.results)

    @property
    def bans(self) -> list[str]:
        return sorted({b for r in self.results for b in r.bans})

    @property
    def disqualification_risks(self) -> list[RuleResult]:
        return [r for r in self.results if r.disqualification]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.checked)

    def summary_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "ok": {True: "✓", False: "✗", None: "·"}[r.passed],
                "rule": r.title,
                "section": f"§{r.section}",
                "measured": r.measured_text,
                "limit": r.limit,
                "penalty": -r.penalty if r.penalty else 0,
                "bans": ", ".join(f"§{b}" for b in r.bans),
            }
            for r in self.results
        ]


def apply_crushing(bridge: Bridge, ruleset: RuleSet) -> Bridge:
    """Copy of ``bridge`` with the crusher plate size from the rules file (§12.5).

    The rules file describes the competition's crusher, so it overrides the plate size
    stored in the bridge file. Other load settings (P_ref, plate position) are kept.
    """
    c = ruleset.crushing
    load = bridge.load.model_copy(
        update={"plate_length_mm": c.plate_length_mm, "plate_width_mm": c.plate_width_mm})
    return bridge.model_copy(update={"load": load})


# --------------------------------------------------------------------------- rounding


def round_half_up(value: float, step: float) -> float:
    """Round to the nearest multiple of ``step`` with halves rounded up (judges' rounding)."""
    if not math.isfinite(value):
        return value
    q = Decimal(str(step))
    return float((Decimal(repr(value)) / q).quantize(Decimal(1), ROUND_HALF_UP) * q)


def mass_penalty(mass_kg: float, steps: Steps) -> float:
    """Stepped mass penalty (rulebook §8.8, Table 8).

    With m rounded to 0.01 kg: 0 if m <= free; 50 if m > cap; otherwise
    points_per_step * ceil((m - free) / step), computed in integer centigrams so that
    6.01 -> 2, 6.50 -> 2, 6.51 -> 4, 15.00 -> 36, 15.01 -> 50.
    """
    cg = int(round_half_up(mass_kg * 100.0, 1))
    free = int(round(steps.free_up_to * 100))
    cap = int(round(steps.cap * 100))
    step = int(round(steps.step * 100))
    if cg <= free:
        return 0.0
    if cg > cap:
        return steps.over_cap_penalty
    return steps.points_per_step * math.ceil((cg - free) / step)


def _fmt(v: float, unit: str) -> str:
    if not math.isfinite(v):
        return "none in the way"
    return f"{v:g} {unit}".strip()


# --------------------------------------------------------------------------- evaluation


def evaluate(
    bridge: Bridge,
    material: Material,
    ruleset: RuleSet | None = None,
    measurements: Measurements | None = None,
) -> RulesReport:
    rs = ruleset or RuleSet.load()
    meas = measurements or measure(bridge, material, rs.constants)
    values = dict(meas.values)
    values["mass_kg"] = bridge_mass(bridge, material).total_kg
    for rule in rs.rules:
        needed = list(rule.measures) + ([rule.measure] if rule.measure else []) + rule.show
        unknown = [k for k in needed if k not in values]
        if unknown:
            raise ValueError(
                f"Rule {rule.key!r} (§{rule.section}) uses unknown measurement(s) {unknown}. "
                f"Available: {sorted(values)}"
            )
    step_len = rs.rounding.get("length_mm", 1.0)
    step_mass = rs.rounding.get("mass_kg", 0.01)

    out: list[RuleResult] = []
    for rule in rs.rules:
        base = dict(key=rule.key, section=rule.section, title=rule.title, limit=rule.limit,
                    ambiguous=rule.ambiguous, note=rule.note)
        if rule.type == "info":
            out.append(RuleResult(**base))
            continue

        if rule.type == "bands":
            worst: tuple[float, list[str], float, str] | None = None
            texts = []
            for key in rule.measures:
                v = round_half_up(values[key], step_len)
                band = next((b for b in rule.bands if b.matches(v)), None)
                if band is None:
                    raise ValueError(f"Rule {rule.key}: no band matches {key} = {v}")
                texts.append(f"{key.removesuffix('_mm').replace('_', ' ')} {_fmt(v, rule.unit)}"
                             if len(rule.measures) > 1 else _fmt(v, rule.unit))
                if worst is None or band.penalty > worst[0] or (
                        band.penalty == worst[0] and len(band.bans) > len(worst[1])):
                    worst = (band.penalty, band.bans, v, key)
            assert worst is not None
            penalty, bans, v, _ = worst
            out.append(RuleResult(**base, measured=v, measured_text="; ".join(texts),
                                  passed=(penalty == 0 and not bans), penalty=penalty,
                                  bans=list(bans)))
        elif rule.type == "check":
            ok = bool(values[rule.measure or ""])
            shown = "; ".join(
                f"{k.removesuffix('_mm').removesuffix('_deg').replace('_', ' ')} "
                f"{_fmt(round_half_up(values[k], 0.1 if k.endswith('_deg') else step_len), 'deg' if k.endswith('_deg') else 'mm')}"  # noqa: E501
                for k in rule.show
            )
            fail = rule.fail
            out.append(RuleResult(
                **base, measured=float(ok), measured_text=shown or ("ok" if ok else "fails"),
                passed=ok, penalty=0 if ok else fail.penalty, bans=[] if ok else fail.bans,
                disqualification=(not ok) and fail.disqualification,
            ))
        elif rule.type == "steps":
            assert rule.steps is not None
            v = round_half_up(values[rule.measure or ""], step_mass)
            pen = mass_penalty(v, rule.steps)
            out.append(RuleResult(**base, measured=v, measured_text=f"{v:.2f} {rule.unit}",
                                  passed=pen == 0, penalty=pen))
    return RulesReport(ruleset=rs.name, results=out)
