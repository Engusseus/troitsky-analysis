"""Materials of the crush simulation: stick laminates, glue lines and floss.

A body's laminate is read from its SolidWorks name when the name says it (bridgesim
recognises ``10-ply UD``, ``2-ply 0-90``, ``4-ply +-45``, ``round toothpick``,
``waxed floss``, ``block``), from the overrides file otherwise, and is a question for the
designer when neither says (see :mod:`bridgesim.sim.idealize`).

Stiffness of a laminate comes from classical lamination theory (CLT) with each ply an
orthotropic stick layer (E_L along the grain, E_T across, G_LT, nu_LT). Its strength along
a direction comes from the max-strain criterion applied to every ply: the laminate fails
when the first ply reaches its tension, compression or shear strain limit. A UD laminate
therefore has the stick's own strengths, a 0/90 laminate about half of them and a +-45
laminate a strength set by in-plane shear.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bridgesim.paths import data_dir
from bridgesim.schema import Material
from bridgesim.yamlio import load_yaml


@dataclass
class SimParams:
    """Flat view of ``materials/simulation.yaml``: ``values['glue.K_n_N_per_mm3']``."""

    values: dict[str, float]
    sources: dict[str, str]
    notes: dict[str, str]
    name: str = ""

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimParams:
        vals, srcs, notes = {}, {}, {}
        for grp, items in d.items():
            if not isinstance(items, dict):
                continue
            for k, p in items.items():
                key = f"{grp}.{k}"
                if isinstance(p, dict):
                    vals[key] = float(p["value"])
                    srcs[key] = str(p.get("source", "assumed"))
                    notes[key] = str(p.get("note", ""))
                else:
                    vals[key] = float(p)
                    srcs[key] = "assumed"
                    notes[key] = ""
        return cls(vals, srcs, notes, str(d.get("name", "")))

    @classmethod
    def load(cls, path: str | Path | None = None) -> SimParams:
        p = Path(path) if path else _default_path()
        return cls.from_dict(load_yaml(p.read_text(encoding="utf-8")))

    def updated(self, changes: dict[str, float], source: str = "override") -> SimParams:
        unknown = [k for k in changes if k not in self.values]
        if unknown:
            raise KeyError(f"unknown simulation parameter(s) {unknown}; known: "
                           f"{sorted(self.values)}")
        bad = [msg for k, val in changes.items() if (msg := param_range_error(k, val))]
        if bad:
            raise ValueError("overrides: " + "; ".join(bad))
        v = dict(self.values)
        s = dict(self.sources)
        for k, val in changes.items():
            v[k] = float(val)
            s[k] = source
        return SimParams(v, s, dict(self.notes), self.name)

    def assumed_keys(self) -> list[str]:
        return [k for k, s in self.sources.items() if s == "assumed"]


# physical range of each simulation parameter (fnmatch pattern, low, high, low inclusive),
# checked when an override sets it; the first matching pattern applies
PARAM_RANGES: list[tuple[str, float, float, bool]] = [
    ("wood.nu_LT", 0.0, 0.5, True),
    ("wood.crush_softening", 0.0, 0.99, True),
    ("wood.densification_strain", 0.0, 0.95, False),
    ("glue.peel_to_shear", 0.0, 5.0, False),
    ("glue.end_grain_factor", 0.0, 2.0, True),
    ("glue.tau_peak_over_avg", 1.0, 2.0, True),
    ("*friction*", 0.0, 2.0, True),
    ("floss.strands", 1.0, 100.0, True),
    ("contact.seating_mm", 0.0, 10.0, True),
    ("scatter.*_cov", 0.0, 1.0, True),
    ("scatter.bow_L_over", 20.0, 1e6, True),
    ("scatter.placement_mm", 0.0, 50.0, True),
    ("scatter.support_level_mm", 0.0, 20.0, True),
    ("scatter.lean_rad", 0.0, 0.1, True),
    ("wood.E_placeholder_MPa", 0.0, 1e6, True),
    ("*", 0.0, float("inf"), False),  # moduli, strengths, energies, sizes: positive
]


def param_range_error(key: str, value) -> str | None:
    """Why ``value`` is not a physical value of the simulation parameter ``key`` (None if
    it is)."""
    import fnmatch
    import math

    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"params.{key} must be a number, got {value!r}"
    if not math.isfinite(v):
        return f"params.{key} must be finite, got {value!r}"
    for pat, lo, hi, lo_in in PARAM_RANGES:
        if fnmatch.fnmatchcase(key, pat):
            if v < lo or (v == lo and not lo_in) or v > hi:
                rng = (f"{'[' if lo_in else '('}{lo:g}, {hi:g}]" if math.isfinite(hi)
                       else f"{'>=' if lo_in else '>'} {lo:g}")
                return f"params.{key} = {v:g} is outside its physical range {rng}"
            return None
    return None


def _default_path() -> Path:
    repo = Path(__file__).resolve().parents[3] / "materials" / "simulation.yaml"
    if repo.exists():
        return repo
    return data_dir("materials") / "simulation.yaml"


# --------------------------------------------------------------------------- laminates


@dataclass
class Ply:
    E_L: float
    E_T: float
    G_LT: float
    nu_LT: float
    f_t: float
    f_c: float
    f_t90: float
    f_c90: float
    f_v: float

    @classmethod
    def from_material(cls, mat: Material, sp: SimParams) -> Ply:
        return cls(
            E_L=mat.E_MPa.value, E_T=sp["wood.E_T_MPa"], G_LT=mat.G_MPa.value,
            nu_LT=sp["wood.nu_LT"], f_t=mat.f_t_MPa.value, f_c=mat.f_c_MPa.value,
            f_t90=sp["wood.f_t90_MPa"], f_c90=sp["wood.f_c90_MPa"], f_v=mat.f_v_MPa.value)

    def Q(self) -> np.ndarray:
        nu21 = self.nu_LT * self.E_T / self.E_L
        d = 1 - self.nu_LT * nu21
        return np.array([[self.E_L / d, nu21 * self.E_L / d, 0],
                         [self.nu_LT * self.E_T / d, self.E_T / d, 0],
                         [0, 0, self.G_LT]])


def _Tstrain(theta_deg: float) -> np.ndarray:
    """Global engineering strain (ex, ey, gxy) -> ply axes (e1, e2, g12)."""
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return np.array([[c * c, s * s, c * s], [s * s, c * c, -c * s],
                     [-2 * c * s, 2 * c * s, c * c - s * s]])


def Qbar(ply: Ply, theta_deg: float) -> np.ndarray:
    T = _Tstrain(theta_deg)
    return T.T @ ply.Q() @ T


KINDS = ("laminate", "toothpick", "floss", "block")


@dataclass
class Layup:
    """Stick laminate of one body. Angles are ply directions in degrees from the body's
    reference axis (its long axis, or the direction named in ``grain``)."""

    kind: str = "laminate"  # one of KINDS
    angles: list[float] = field(default_factory=lambda: [0.0])
    grain: str = "long"  # "long" (body's long axis / local edges), "span" (global X), ...
    source: str = "default"  # "name", "override" or "default"
    text: str = ""
    orientation_assumed: bool = False  # "10-ply" without UD / 0-90 / +-45

    @property
    def label(self) -> str:
        if self.kind != "laminate":
            return self.kind
        n = len(self.angles)
        a = sorted({abs(round(x)) for x in self.angles})
        name = ("UD" if a == [0] else "0/90" if a == [0, 90] else "+-45" if a == [45]
                else "/".join(str(int(x)) for x in self.angles))
        return f"{n}-ply {name}"


_PLY = re.compile(r"(\d+)\s*-?\s*ply", re.I)


def parse_layup(name: str) -> Layup | None:
    """Layup stated in a body name, or None when the name says nothing about it."""
    t = name.lower()
    if "floss" in t and "toggle" not in t:
        return Layup("floss", [0.0], source="name", text=name)
    if "toothpick" in t or "dowel" in t:
        return Layup("toothpick", [0.0], source="name", text=name)
    m = _PLY.search(t)
    n = int(m.group(1)) if m else None
    grain = "span" if "along the span" in t else "long"
    if re.search(r"\b(ud|unidirectional)\b", t):
        return Layup("laminate", [0.0] * (n or 1), grain, "name", name)
    if re.search(r"0\s*[-/]\s*90", t):
        k = n or 2
        return Layup("laminate", [0.0 if i % 2 == 0 else 90.0 for i in range(k)], grain,
                     "name", name)
    if re.search(r"(\+-|±|\+/-)\s*45", t):
        k = n or 2
        return Layup("laminate", [45.0 if i % 2 == 0 else -45.0 for i in range(k)], grain,
                     "name", name)
    if "block" in t:
        return Layup("block", [0.0] * (n or 1), grain, "name", name)
    if n:  # "10-ply" with no orientation: sticks all along the member, the usual case
        lay = Layup("laminate", [0.0] * n, grain, "name", name)
        lay.orientation_assumed = True
        return lay
    return None


_OV_LAYUP = re.compile(
    r"^\s*(?P<n>\d+)\s*-?\s*ply\s+(?P<o>ud|unidirectional|0\s*[-/]\s*90|"
    r"(?:\+-|±|\+/-)\s*45)(?P<span>\s+along\s+the\s+span)?\s*$")
_OV_SPECIAL = re.compile(r"^\s*(block|(round\s+)?toothpick|(waxed\s+)?floss)\s*$")


def parse_override_layup(text: str) -> Layup | None:
    """A layup written in an overrides file, in a strict form: 'N-ply UD', 'N-ply 0-90',
    'N-ply +-45' (optionally followed by 'along the span'), 'block', 'toothpick' or
    'floss'. Anything else (a typo, an unfilled '<...>' placeholder) is None: unlike a
    body name, an override is never guessed at."""
    t = str(text).strip().lower()
    if _OV_SPECIAL.match(t):
        return parse_layup(t)
    m = _OV_LAYUP.match(t)
    if not m or int(m.group("n")) < 1:
        return None
    return parse_layup(t)


@dataclass
class LaminateProps:
    """Membrane (A, per unit width) and bending (D) stiffness of a laminate of total
    thickness t, plus its effective modulus and failure strains along the reference axis."""

    A: np.ndarray
    D: np.ndarray
    t: float
    E_x: float  # membrane modulus along the reference axis
    E_bx: float  # bending modulus along the reference axis
    G_xy: float
    eps_t: float  # laminate failure strain in tension along the reference axis
    eps_c: float  # ... and in compression (positive number)
    angles: list[float]
    ply: Ply
    B: np.ndarray | None = None  # membrane-bending coupling (unsymmetric layups, e.g. 0/90)

    @property
    def f_t(self) -> float:
        return self.E_x * self.eps_t

    @property
    def f_c(self) -> float:
        return self.E_x * self.eps_c


def laminate(ply: Ply, angles: list[float], t: float) -> LaminateProps:
    n = len(angles)
    h = t / n
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, th in enumerate(angles):
        z0 = -t / 2 + k * h
        z1 = z0 + h
        Qb = Qbar(ply, th)
        A += Qb * h
        B += Qb * (z1**2 - z0**2) / 2
        D += Qb * (z1**3 - z0**3) / 3
    a = np.linalg.inv(A)
    d = np.linalg.inv(D)
    E_x = 1 / (a[0, 0] * t)
    E_bx = 12 / (d[0, 0] * t**3)
    G_xy = 1 / (a[2, 2] * t)
    nu_xy = -a[0, 1] / a[0, 0]

    # Cracks between the sticks of a ply lying across the load (transverse "matrix"
    # cracking) do not break a laminate that still has sticks along the load, so the
    # across-the-grain limits only count when no ply is within 30 degrees of the axis.
    fibre_aligned = any(abs(((th + 90) % 180) - 90) <= 30 for th in angles)

    def fail_strain(sign: float) -> float:
        ex, ey = sign, -nu_xy * sign
        lim = math.inf
        for th in set(angles):
            e1, e2, g12 = _Tstrain(th) @ np.array([ex, ey, 0.0])
            checks = [(e1, ply.f_t / ply.E_L, ply.f_c / ply.E_L)]
            if not fibre_aligned:
                checks.append((e2, ply.f_t90 / ply.E_T, ply.f_c90 / ply.E_T))
            for val, lt, lc in checks:
                if val > 1e-12:
                    lim = min(lim, lt / val)
                elif val < -1e-12:
                    lim = min(lim, lc / -val)
            if abs(g12) > 1e-12:
                lim = min(lim, ply.f_v / ply.G_LT / abs(g12))
        return lim

    return LaminateProps(A, D, t, E_x, E_bx, G_xy, fail_strain(1.0), fail_strain(-1.0),
                         list(angles), ply, B)


def ply_layers(angles: list[float], t: float, min_layers: int = 2
               ) -> list[tuple[float, float, float]]:
    """Through-thickness layers (z_bottom, z_top, angle) of a laminate for layered shell
    integration: consecutive plies with the same angle are merged, and a laminate with
    fewer than ``min_layers`` layers is split so the outer fibres are resolved."""
    n = len(angles)
    h = t / n
    groups: list[list[float]] = []
    for k, th in enumerate(angles):
        if groups and abs(((groups[-1][2] - th) + 90) % 180 - 90) < 1e-6:
            groups[-1][1] = -t / 2 + (k + 1) * h
        else:
            groups.append([-t / 2 + k * h, -t / 2 + (k + 1) * h, th])
    out = []
    split = max(1, math.ceil(min_layers / len(groups)))
    for zb, zt, th in groups:
        dz = (zt - zb) / split
        out += [(zb + i * dz, zb + (i + 1) * dz, th) for i in range(split)]
    return out


def fibre_spread(n: int) -> float:
    """Factor on the spacing of n equal fibres across a rectangle so that they give the
    rectangle's exact second moment (n equally spaced fibres give (1 - 1/n^2) of it)."""
    return math.sqrt(n * n / (n * n - 1.0)) if n > 1 else 1.0
