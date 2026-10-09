"""Through Warren truss generator.

Geometry (X along the bridge, Y up, Z across, table at Y = 0):

* panel length p = span / n
* bottom-chord nodes B_i at x = i p (i = 0..n), top-chord nodes T_i at x = (i + 1/2) p
  (i = 0..n-1), in two truss planes at z = +/- s/2 with s = deck clear width + truss
  member width, so the inner faces of the trusses are exactly the deck clear width apart
* bottom-chord elevation = deck top - deck thickness - floor-beam depth / 2, so the deck
  sits on top of the floor beams
* diagonals B_i -> T_i and T_i -> B_{i+1}
* a floor beam at every bottom-chord station, split at a centre node C_i (z = 0) where the
  deck load is applied
* top struts join the two top nodes at each station
* vertical piers from the end bottom-chord nodes down to the table
* optional X-bracing: top plane (every panel except the one spanning mid-span, to keep
  the clear opening of rulebook §8.9 free), bottom plane, and each end pier plane
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridgesim.schema import (
    MIN_SIZE_MM,
    Bridge,
    Deck,
    Member,
    Node,
    PlateLoad,
    Section,
    Stick,
    Supports,
)


class SectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sticks: int = Field(ge=1, le=60)
    layout: Literal["flat", "on_edge"]


def _default_sections() -> dict[str, SectionSpec]:
    s = SectionSpec
    return {
        "top_chord": s(sticks=4, layout="on_edge"),
        "bottom_chord": s(sticks=3, layout="on_edge"),
        "diagonal": s(sticks=6, layout="on_edge"),
        "floor_beam": s(sticks=18, layout="flat"),
        "top_strut": s(sticks=3, layout="flat"),
        "top_brace": s(sticks=3, layout="flat"),
        "bottom_brace": s(sticks=2, layout="flat"),
        "pier": s(sticks=10, layout="flat"),
        "pier_brace": s(sticks=3, layout="flat"),
    }


class WarrenParams(BaseModel):
    """Inputs for :func:`generate_warren`. Bounds keep the UI inputs sensible."""

    model_config = ConfigDict(extra="forbid")

    name: str = "Warren truss"
    span_mm: float = Field(1150.0, ge=500, le=1500, description="Pier centre to centre")
    n_panels: int = Field(8, ge=2, le=24, description="Number of bottom-chord panels (even)")
    truss_height_mm: float = Field(260.0, ge=40, le=520, description="Chord centroid to centroid")
    deck_top_elevation_mm: float = Field(200.0, ge=20, le=500, description="Table to deck top")
    deck_clear_width_mm: float = Field(170.0, ge=50, le=340)
    deck_overhang_mm: float = Field(50.0, ge=0, le=250, description="Deck beyond each pier")
    deck_thickness_mm: float = Field(2.0, ge=MIN_SIZE_MM, le=20)
    top_bracing: Literal["x", "none"] = "x"
    bottom_bracing: Literal["x", "none"] = "x"
    pier_bracing: Literal["x", "none"] = "x"
    joint_fixity: Literal["rigid", "pinned"] = "rigid"
    sections: dict[str, SectionSpec] = Field(default_factory=_default_sections)

    @model_validator(mode="after")
    def _check(self) -> WarrenParams:
        if self.n_panels % 2:
            raise ValueError("n_panels must be even so a floor beam sits at mid-span")
        missing = set(_default_sections()) - set(self.sections)
        if missing:
            raise ValueError(f"Missing section specs for groups: {sorted(missing)}")
        return self


def generate_warren(params: WarrenParams | None = None, stick: Stick | None = None) -> Bridge:
    """Build a through Warren truss :class:`Bridge` from ``params``."""
    p_ = params or WarrenParams()
    stick = stick or Stick()

    sections = [
        Section(id=g, sticks=spec.sticks, layout=spec.layout) for g, spec in p_.sections.items()
    ]
    dims = {s.id: s.dims(stick) for s in sections}  # (b, d)

    n = p_.n_panels
    p = p_.span_mm / n
    b_truss = max(dims["top_chord"][0], dims["bottom_chord"][0], dims["diagonal"][0])
    s = p_.deck_clear_width_mm + b_truss
    zs = {"n": -s / 2.0, "f": s / 2.0}
    d_fb = dims["floor_beam"][1]
    y_b = p_.deck_top_elevation_mm - p_.deck_thickness_mm - d_fb / 2.0
    y_t = y_b + p_.truss_height_mm
    if y_b <= 1.0:
        raise ValueError(
            f"Deck too low: bottom chord would sit at y = {y_b:.1f} mm. Raise the deck top "
            f"to at least {math.floor(p_.deck_thickness_mm + d_fb / 2.0 + 1.0) + 1} mm (it sits on "
            f"{d_fb:g} mm deep floor beams)."
        )

    nodes: list[Node] = []
    members: list[Member] = []

    def node(nid: str, x: float, y: float, z: float) -> str:
        nodes.append(Node(id=nid, x_mm=round(x, 6), y_mm=round(y, 6), z_mm=round(z, 6)))
        return nid

    def member(mid: str, i: str, j: str, group: str) -> None:
        members.append(Member(id=mid, i=i, j=j, section=group, group=group))  # type: ignore[arg-type]

    for side, z in zs.items():
        for i in range(n + 1):
            node(f"B{i}{side}", i * p, y_b, z)
        for i in range(n):
            node(f"T{i}{side}", (i + 0.5) * p, y_t, z)
    for i in range(n + 1):
        node(f"C{i}", i * p, y_b, 0.0)
    for e in (0, n):
        for side, z in zs.items():
            node(f"P{e}{side}", e * p, 0.0, z)

    for side in zs:
        for i in range(n):
            member(f"bc{i}{side}", f"B{i}{side}", f"B{i + 1}{side}", "bottom_chord")
            member(f"d{i}u{side}", f"B{i}{side}", f"T{i}{side}", "diagonal")
            member(f"d{i}d{side}", f"T{i}{side}", f"B{i + 1}{side}", "diagonal")
        for i in range(n - 1):
            member(f"tc{i}{side}", f"T{i}{side}", f"T{i + 1}{side}", "top_chord")
        for i in range(n + 1):
            member(f"fb{i}{side}", f"B{i}{side}", f"C{i}", "floor_beam")
        for e in (0, n):
            member(f"pier{e}{side}", f"P{e}{side}", f"B{e}{side}", "pier")
    for i in range(n):
        member(f"ts{i}", f"T{i}n", f"T{i}f", "top_strut")

    if p_.top_bracing == "x":
        mid_panel = n // 2 - 1  # top panel T_k..T_k+1 that spans mid-span
        for i in range(n - 1):
            if i == mid_panel:
                continue
            member(f"tb{i}a", f"T{i}n", f"T{i + 1}f", "top_brace")
            member(f"tb{i}b", f"T{i}f", f"T{i + 1}n", "top_brace")
    if p_.bottom_bracing == "x":
        for i in range(n):
            member(f"bb{i}a", f"B{i}n", f"B{i + 1}f", "bottom_brace")
            member(f"bb{i}b", f"B{i}f", f"B{i + 1}n", "bottom_brace")
    if p_.pier_bracing == "x":
        for e in (0, n):
            member(f"pb{e}a", f"P{e}n", f"B{e}f", "pier_brace")
            member(f"pb{e}b", f"P{e}f", f"B{e}n", "pier_brace")

    deck = Deck(
        x_start_mm=-p_.deck_overhang_mm,
        x_end_mm=p_.span_mm + p_.deck_overhang_mm,
        top_elevation_mm=p_.deck_top_elevation_mm,
        clear_width_mm=p_.deck_clear_width_mm,
        thickness_mm=p_.deck_thickness_mm,
    )
    return Bridge(
        name=p_.name,
        joint_fixity=p_.joint_fixity,
        nodes=nodes,
        sections=sections,
        members=members,
        deck=deck,
        supports=Supports(pinned=[f"P0{s_}" for s_ in zs], roller=[f"P{n}{s_}" for s_ in zs]),
        load=PlateLoad(deck_support_nodes=[f"C{i}" for i in range(n + 1)]),
        metadata={"generator": "warren", "params": p_.model_dump(mode="json")},
    )
