"""Pydantic data model for a bridge. A ``Bridge`` round-trips to YAML exactly.

Conventions: mm, N, MPa; X longitudinal, Y vertical (up, table at Y = 0), Z transverse.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bridgesim.sections import RectProps, rectangle, stick_stack_dims
from bridgesim.textsafe import has_control_chars

SCHEMA_VERSION = 1

#: Size limits for a bridge model. A detailed popsicle-stick bridge needs a few hundred
#: nodes; the limits stop a malicious or mistaken file from exhausting a shared server.
MAX_NODES = 2000
MAX_MEMBERS = 6000
MAX_SECTIONS = 200

Source = Literal["assumed", "measured"]

MemberGroup = Literal[
    "top_chord",
    "bottom_chord",
    "diagonal",
    "vertical",
    "floor_beam",
    "top_strut",
    "top_brace",
    "bottom_brace",
    "pier",
    "pier_brace",
    "other",
]

MEMBER_GROUPS: tuple[str, ...] = MemberGroup.__args__  # type: ignore[attr-defined]

Release = Literal["Rxi", "Ryi", "Rzi", "Rxj", "Ryj", "Rzj"]
DOF = Literal["DX", "DY", "DZ", "RX", "RY", "RZ"]


#: Coordinates beyond +/-100 m are certainly a units mistake (bridges are ~1.4 m long) and
#: would make geometric sampling allocate huge arrays.
MAX_COORD_MM = 100_000.0
#: Largest cross-section, stick or plate dimension accepted (10 m). Every numeric input has
#: a finite upper bound so that products such as areas, volumes and masses cannot overflow.
MAX_SIZE_MM = 10_000.0
#: Smallest stick or section dimension accepted (so volumes and areas cannot underflow).
MIN_SIZE_MM = 0.01
#: Total length of all members (a Troitsky bridge has about 20 to 40 m). Bounds the work of
#: the geometric rule checks, which sample every member at 2 mm.
MAX_TOTAL_MEMBER_LENGTH_MM = 500_000.0


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _plain_text(v: str) -> str:
    """Names and ids are shown in terminals and web pages: no control characters."""
    if has_control_chars(v):
        raise ValueError("must not contain control characters")
    return v


def _identifier(v: str) -> str:
    """Node, section and member ids: non-blank plain text (Pynite renames blank ids)."""
    if not v.strip():
        raise ValueError("must not be blank")
    return _plain_text(v)


# --------------------------------------------------------------------------- material


class Prop(_Strict):
    """A material or joint value together with where it came from."""

    value: float
    source: Source = "assumed"
    note: str = ""

    @field_validator("note")
    @classmethod
    def _plain_note(cls, v: str) -> str:
        if has_control_chars(v, allow="\n"):
            raise ValueError("must not contain control characters")
        return v


class Stick(_Strict):
    length_mm: float = Field(115.0, ge=MIN_SIZE_MM, le=MAX_SIZE_MM)
    width_mm: float = Field(10.0, ge=MIN_SIZE_MM, le=MAX_SIZE_MM)
    thickness_mm: float = Field(2.0, ge=MIN_SIZE_MM, le=MAX_SIZE_MM)

    @property
    def volume_mm3(self) -> float:
        return self.length_mm * self.width_mm * self.thickness_mm


class Glue(_Strict):
    """Glued-joint placeholder model: capacity = tau_g * overlap * width * faces."""

    tau_g_MPa: Prop
    mass_fraction: Prop
    overlap_mm: Prop
    faces: Prop
    exclude_groups: list[MemberGroup] = Field(
        default_factory=lambda: ["top_chord", "bottom_chord"],
        description="Groups treated as continuous through joints (no joint check).",
    )


class Material(_Strict):
    name: str
    description: str = ""

    _text = field_validator("name", "description")(lambda cls, v: _plain_text(v))
    stick: Stick = Field(default_factory=Stick)
    E_MPa: Prop
    G_MPa: Prop
    nu: Prop
    f_t_MPa: Prop
    f_c_MPa: Prop
    f_b_MPa: Prop
    f_v_MPa: Prop
    density_kg_m3: Prop
    glue: Glue

    @model_validator(mode="after")
    def _physical(self) -> Material:
        """Reject values that cannot describe wood and glue (and would corrupt results)."""
        # Upper bounds are far above any real material (steel: E = 2e5 MPa, 7850 kg/m3) and
        # only stop finite but absurd values from overflowing capacities and masses.
        # Lower bounds are far below any real material and keep products such as mass and
        # capacities from underflowing to zero.
        limits = {  # name: (lower, upper, lower bound inclusive?)
            "E_MPa": (1, 1e6, True), "G_MPa": (1, 1e6, True), "nu": (-1, 0.5, False),
            "f_t_MPa": (1e-3, 1e5, True), "f_c_MPa": (1e-3, 1e5, True),
            "f_b_MPa": (1e-3, 1e5, True), "f_v_MPa": (1e-3, 1e5, True),
            "density_kg_m3": (1, 1e5, True), "glue.tau_g_MPa": (1e-3, 1e5, True),
            "glue.mass_fraction": (0, 1, True),
            "glue.overlap_mm": (MIN_SIZE_MM, MAX_SIZE_MM, True), "glue.faces": (1, 100, True),
        }
        for key, prop in self.props().items():
            v = prop.value
            lo, hi, incl = limits[key]
            if not math.isfinite(v) or (v < lo if incl else v <= lo) or v >= hi:
                rng = f"{'>=' if incl else '>'} {lo} and < {hi:g}"
                raise ValueError(f"Material value {key} = {v} is not physical (must be {rng})")
        if not float(self.glue.faces.value).is_integer():
            raise ValueError(f"glue.faces must be a whole number of glued faces, not "
                             f"{self.glue.faces.value:g}")
        return self

    def props(self) -> dict[str, Prop]:
        """All value-carrying properties, keyed by a dotted name (for UI badges)."""
        out = {
            k: getattr(self, k)
            for k in ("E_MPa", "G_MPa", "nu", "f_t_MPa", "f_c_MPa", "f_b_MPa", "f_v_MPa",
                      "density_kg_m3")
        }
        for k in ("tau_g_MPa", "mass_fraction", "overlap_mm", "faces"):
            out[f"glue.{k}"] = getattr(self.glue, k)
        return out

    def assumed_keys(self) -> list[str]:
        return [k for k, p in self.props().items() if p.source == "assumed"]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Material:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- geometry


#: Node coordinates are snapped to this many decimals (1e-6 mm) so that float noise such as
#: 0.1 + 0.2 - 0.3 cannot make bridgesim and Pynite disagree on whether a member is
#: vertical or horizontal (which decides its section orientation).
COORD_DECIMALS = 6


class Node(_Strict):
    id: str
    x_mm: float = Field(ge=-MAX_COORD_MM, le=MAX_COORD_MM)
    y_mm: float = Field(ge=-MAX_COORD_MM, le=MAX_COORD_MM)
    z_mm: float = Field(ge=-MAX_COORD_MM, le=MAX_COORD_MM)

    _id = field_validator("id")(lambda cls, v: _identifier(v))

    @field_validator("x_mm", "y_mm", "z_mm")
    @classmethod
    def _snap(cls, v: float) -> float:
        return round(v, COORD_DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x_mm, self.y_mm, self.z_mm)


class Section(_Strict):
    """Either a stick stack ``{sticks, layout}`` or an explicit rectangle ``{b_mm, d_mm}``.

    ``d`` is the depth in the member's primary bending plane (local y): vertical for
    horizontal members, in the truss plane for truss members. See ``sections.py``.
    """

    id: str
    sticks: int | None = Field(None, ge=1, le=60)
    layout: Literal["flat", "on_edge"] | None = None
    b_mm: float | None = Field(None, ge=MIN_SIZE_MM, le=MAX_SIZE_MM)
    d_mm: float | None = Field(None, ge=MIN_SIZE_MM, le=MAX_SIZE_MM)

    _id = field_validator("id")(lambda cls, v: _identifier(v))

    @model_validator(mode="after")
    def _one_definition(self) -> Section:
        stack = self.sticks is not None or self.layout is not None
        rect = self.b_mm is not None or self.d_mm is not None
        if stack == rect:
            raise ValueError(
                f"Section {self.id!r}: give either sticks+layout or b_mm+d_mm (not both)"
            )
        if stack and (self.sticks is None or self.layout is None):
            raise ValueError(f"Section {self.id!r}: sticks and layout must both be set")
        if rect and (self.b_mm is None or self.d_mm is None):
            raise ValueError(f"Section {self.id!r}: b_mm and d_mm must both be set")
        return self

    def dims(self, stick: Stick) -> tuple[float, float]:
        """(b, d) in mm."""
        if self.sticks is not None:
            return stick_stack_dims(self.sticks, self.layout or "flat", stick.width_mm,
                                    stick.thickness_mm)
        return float(self.b_mm), float(self.d_mm)  # type: ignore[arg-type]

    def props(self, stick: Stick) -> RectProps:
        return rectangle(*self.dims(stick))

    def label(self) -> str:
        if self.sticks is not None:
            return f"{self.sticks} sticks {self.layout}"
        return f"{self.b_mm:g} x {self.d_mm:g} mm"


class Member(_Strict):
    id: str
    i: str
    j: str
    section: str
    group: MemberGroup = "other"
    K: float = Field(1.0, ge=0.1, le=10, description="Effective-length factor for buckling")
    releases: list[Release] = Field(default_factory=list)

    _id = field_validator("id", "i", "j", "section")(lambda cls, v: _identifier(v))


class Deck(_Strict):
    """Non-structural deck surface. Used for load placement, measurements and mass."""

    x_start_mm: float = Field(ge=-MAX_COORD_MM, le=MAX_COORD_MM)
    x_end_mm: float = Field(ge=-MAX_COORD_MM, le=MAX_COORD_MM)
    top_elevation_mm: float = Field(gt=0, le=MAX_COORD_MM)
    clear_width_mm: float = Field(gt=0, le=MAX_COORD_MM)
    thickness_mm: float = Field(2.0, gt=0, le=MAX_SIZE_MM)
    z_center_mm: float = Field(0.0, ge=-MAX_COORD_MM, le=MAX_COORD_MM)

    @model_validator(mode="after")
    def _ordered(self) -> Deck:
        if self.x_end_mm <= self.x_start_mm:
            raise ValueError("Deck x_end_mm must be greater than x_start_mm")
        return self

    @property
    def length_mm(self) -> float:
        return self.x_end_mm - self.x_start_mm


class Supports(_Strict):
    """Pier base nodes resting on the test platform (no anchorage, rulebook §8.3).

    ``pinned`` nodes restrain DX, DY, DZ; ``roller`` nodes restrain DY, DZ (free to slide
    longitudinally). Rotations are free. ``extra_restraints`` exists for validation models.
    """

    pinned: list[str]
    roller: list[str] = Field(default_factory=list)
    extra_restraints: dict[str, Annotated[list[DOF], Field(min_length=1)]] = Field(
        default_factory=dict)

    @model_validator(mode="after")
    def _no_overlap(self) -> Supports:
        both = sorted(set(self.pinned) & set(self.roller))
        if both:
            raise ValueError(f"Nodes cannot be both pinned and roller supports: {both}")
        return self


class PlateLoad(_Strict):
    """Crusher plate (rulebook §12.5): uniform over plate_length along X, centred on x."""

    P_ref_N: float = Field(1000.0, ge=1e-3, le=1e7)
    plate_length_mm: float = Field(200.0, ge=1e-3, le=MAX_SIZE_MM)
    plate_width_mm: float = Field(90.0, ge=1e-3, le=MAX_SIZE_MM)
    x_center_mm: float | None = Field(
        None, ge=-MAX_COORD_MM, le=MAX_COORD_MM,
        description="Plate centre along X; default = mid-span between supports",
    )
    deck_support_nodes: list[str] = Field(
        min_length=1,
        description="Nodes the deck bears on along its centreline (e.g. floor-beam centres)",
    )


class Bridge(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION  # bump with a migration when fields change
    name: str = "Untitled bridge"
    material: str | Material = "popsicle_birch"
    joint_fixity: Literal["rigid", "pinned"] = "rigid"
    nodes: list[Node] = Field(max_length=MAX_NODES)
    sections: list[Section] = Field(max_length=MAX_SECTIONS)
    members: list[Member] = Field(max_length=MAX_MEMBERS)
    deck: Deck
    supports: Supports
    load: PlateLoad
    metadata: dict[str, Any] = Field(default_factory=dict)

    _name = field_validator("name")(lambda cls, v: _plain_text(v))

    @field_validator("nodes", "sections", "members")
    @classmethod
    def _unique_ids(cls, items: list[Any]) -> list[Any]:
        seen: set[str] = set()
        for it in items:
            if it.id in seen:
                raise ValueError(f"Duplicate id {it.id!r}")
            seen.add(it.id)
        return items

    @model_validator(mode="after")
    def _references(self) -> Bridge:
        nodes = {n.id: n for n in self.nodes}
        sections = {s.id for s in self.sections}
        for m in self.members:
            for end in (m.i, m.j):
                if end not in nodes:
                    raise ValueError(f"Member {m.id!r} references unknown node {end!r}")
            if m.section not in sections:
                raise ValueError(f"Member {m.id!r} references unknown section {m.section!r}")
            a, b = nodes[m.i], nodes[m.j]
            if (a.x_mm, a.y_mm, a.z_mm) == (b.x_mm, b.y_mm, b.z_mm):
                raise ValueError(f"Member {m.id!r} has zero length")
        total = sum(math.dist(nodes[m.i].xyz, nodes[m.j].xyz) for m in self.members)
        if total > MAX_TOTAL_MEMBER_LENGTH_MM:
            raise ValueError(
                f"Members add up to {total / 1000:,.0f} m; the limit is "
                f"{MAX_TOTAL_MEMBER_LENGTH_MM / 1000:,.0f} m (a Troitsky bridge has 20 to 40 m)."
            )
        support_nodes = (
            self.supports.pinned + self.supports.roller + list(self.supports.extra_restraints)
        )
        for nid in support_nodes + self.load.deck_support_nodes:
            if nid not in nodes:
                raise ValueError(f"Unknown node {nid!r} in supports/load")
        translational = any(d in ("DX", "DY", "DZ")
                            for dofs in self.supports.extra_restraints.values() for d in dofs)
        if not self.supports.pinned and not translational:
            raise ValueError("At least one pinned support (or a translational extra "
                             "restraint) is required")
        self._check_no_hidden_joints(nodes)
        return self

    def _check_no_hidden_joints(self, nodes: dict[str, Node]) -> None:
        """Reject nodes lying inside a member that does not connect to them.

        Pynite splits a member at every model node on its line, which would silently add a
        rigid joint the file never declared. Split the member at that node explicitly.
        """
        if not self.members:
            return
        ids = list(nodes)
        P = np.array([nodes[n].xyz for n in ids])
        for m in self.members:
            a, b = np.array(nodes[m.i].xyz), np.array(nodes[m.j].xyz)
            v = b - a
            L = float(np.linalg.norm(v))
            lo, hi = np.minimum(a, b) - 1e-6, np.maximum(a, b) + 1e-6
            near = np.all((P >= lo) & (P <= hi), axis=1)
            if not near.any():
                continue
            d = P[near] - a
            t = d @ v / L
            perp = np.linalg.norm(d - np.outer(t, v / L), axis=1)
            inside = (t > 1e-6) & (t < L - 1e-6) & (perp < 1e-6 + 1e-9 * L)
            for k in np.flatnonzero(inside):
                nid = [n for n, keep in zip(ids, near, strict=True) if keep][k]
                raise ValueError(
                    f"Node {nid!r} lies on member {m.id!r} but the member does not connect to "
                    f"it. Split {m.id!r} at {nid!r} into two members, or move the node."
                )

    # ------------------------------------------------------------------ helpers
    def node_map(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def section_map(self) -> dict[str, Section]:
        return {s.id: s for s in self.sections}

    def member_length_mm(self, m: Member, nodes: dict[str, Node] | None = None) -> float:
        nodes = nodes or self.node_map()
        a, b = nodes[m.i], nodes[m.j]
        return ((b.x_mm - a.x_mm) ** 2 + (b.y_mm - a.y_mm) ** 2 + (b.z_mm - a.z_mm) ** 2) ** 0.5

    def support_nodes(self) -> list[str]:
        """Nodes the bridge stands on: pinned, roller, and any extra restraint in DY."""
        vertical = [n for n, dofs in self.supports.extra_restraints.items() if "DY" in dofs]
        return list(dict.fromkeys(self.supports.pinned + self.supports.roller + vertical))

    def reaction_nodes(self) -> list[str]:
        """Supports plus any node restrained in translation through extra_restraints."""
        extra = [n for n, dofs in self.supports.extra_restraints.items()
                 if any(d in ("DX", "DY", "DZ") for d in dofs)]
        return list(dict.fromkeys(self.support_nodes() + extra))

    # ------------------------------------------------------------------ YAML I/O
    def to_yaml(self) -> str:
        """YAML text; nodes, sections and members are written one per line."""
        data = self.model_dump(mode="json", exclude_none=True)
        for m in data["members"]:  # drop defaults so each member stays a flat mapping
            if not m.get("releases"):
                m.pop("releases", None)
            if m.get("K") == 1.0:
                m.pop("K")
        return yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=None, width=120
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml_str(cls, text: str) -> Bridge:
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def from_file(cls, path: str | Path) -> Bridge:
        return cls.from_yaml_str(Path(path).read_text(encoding="utf-8"))
