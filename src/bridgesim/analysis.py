"""Linear static analysis at the reference load and the load-factor method.

1. Apply the crusher plate at P_ref (default 1000 N) and solve once (Pynite, linear).
2. For each member take envelope maxima along its length: axial force N (tension +),
   moments |M_y|, |M_z|, shears; and the resultant force at each end for the joint check.
3. Utilisation U_i at P_ref (see ``checks.py``). Because the analysis is linear, every
   demand scales with load, so the load at which member i reaches capacity is P_ref / U_i:

       F_u,strength = P_ref * min_i (1 / U_i)

4. Deflection limit (rulebook §12.5, 50 mm at mid-span): with delta_ref the largest
   downward displacement of the loaded deck-support nodes at P_ref,

       F_u,delta = P_ref * 50 / delta_ref

5. Global elastic buckling ("instability", §12.5): F_u,cr = lambda_cr * P_ref from the
   eigenvalue problem (K + lambda K_g) phi = 0 (see ``stability.py``).
6. Predicted ultimate load F_u,p = min(F_u,strength, F_u,delta, F_u,cr); efficiency
   eta_s = F_u,p [kgf] / m [kg] (rulebook §12.6).

This is a first-failure, linear-elastic estimate: no P-Delta amplification, imperfections,
joint slip, or load redistribution after the first member fails. The one exception is a
member marked ``tension_only`` (a slender brace or strap): if it would be in compression it
is left out and the model re-solved without it (Pynite's tension-only iteration), because
such a member bows elastically instead of failing. Its forces are then zero, and it is
reported as slack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bridgesim.checks import MODE_LABELS, Capacities, Utilisation, capacities, utilisation
from bridgesim.loads import PlatePlacement, plate_nodal_loads, plate_placement
from bridgesim.mass import MassReport, bridge_mass
from bridgesim.materials import bridge_material
from bridgesim.model import COMBO, REFINE_MAX_MEMBERS, build_model, splittable_count
from bridgesim.schema import Bridge, Material
from bridgesim.stability import (
    BucklingResult,
    braces_working_terms,
    diaphragm_terms,
    global_buckling,
)
from bridgesim.units import n_to_kgf

DEFLECTION_LIMIT_MM = 50.0


class AnalysisError(ValueError):
    """The model cannot give a meaningful prediction (e.g. no load path)."""

N_POINTS = 11
#: Assumed static friction coefficient between the bridge and the steel platform, used only
#: to warn about horizontal support reactions (the bridge is not anchored, §8.3).
FRICTION_COEFF = 0.4
#: Above this fraction of the global buckling load, second-order (P-delta) amplification of
#: imperfections, 1 / (1 - F/F_cr), exceeds 2 and is reported as a warning.
NEAR_BUCKLING_RATIO = 0.5
#: A member that governs by buckling while carrying less than this fraction of the largest
#: member force is most likely a brace squeezed by the chords shortening (compatibility),
#: not one carrying the load: the results suggest marking it tension_only. Bracing groups
#: get the same hint whatever their share of the force.
LOW_FORCE_RATIO = 0.02
BRACE_GROUPS = frozenset({"top_brace", "bottom_brace", "pier_brace"})
#: A tension-only member is left out when its compression exceeds this fraction of P_ref
#: (so that round-off in a member with no force does not switch it off).
SLACK_TOLERANCE = 1e-9


@dataclass
class MemberResult:
    id: str
    group: str
    section: str
    section_label: str
    L_mm: float
    b_mm: float
    d_mm: float
    N_N: float  # tension positive, at P_ref
    My_Nmm: float
    Mz_Nmm: float
    Vy_N: float
    Vz_N: float
    F_end_N: float
    cap: Capacities
    util: Utilisation

    @property
    def U(self) -> float:
        return self.util.U

    @property
    def mode(self) -> str:
        return self.util.mode


@dataclass
class Reaction:
    node: str
    FX_N: float
    FY_N: float
    FZ_N: float


@dataclass
class AnalysisResult:
    bridge: Bridge
    material: Material
    P_ref_N: float
    nodal_loads_N: dict[str, float]
    plate: PlatePlacement
    members: list[MemberResult]
    displacements_mm: dict[str, tuple[float, float, float]]
    reactions: list[Reaction]
    delta_ref_mm: float
    delta_node: str
    Fu_strength_N: float
    Fu_deflection_N: float
    Fu_buckling_N: float
    Fu_pred_N: float
    governing_member: str | None
    governing_mode: str
    mass: MassReport
    deflection_limit_mm: float = DEFLECTION_LIMIT_MM
    buckling: BucklingResult | None = None
    #: Members modelled as one element in the global buckling analysis (all of them above
    #: REFINE_MAX_MEMBERS; otherwise any member crossed at every candidate split point).
    buckling_unsplit_members: list[str] = field(default_factory=list)
    #: Tension-only members that would be in compression, so were left out of the analysis.
    slack_members: list[str] = field(default_factory=list)
    #: Global buckling with every slack member that crosses another slack one (an X) put
    #: back at half its axial stiffness: one strap of each X working. The upper end of the
    #: bracket whose lower end, ``buckling``, leaves them out (``braces_working_terms``).
    buckling_braces_working: BucklingResult | None = None
    braces_working_members: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fe_model: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------ derived values
    @property
    def Fu_pred_kgf(self) -> float:
        return n_to_kgf(self.Fu_pred_N)

    @property
    def load_factor(self) -> float:
        return self.Fu_pred_N / self.P_ref_N

    @property
    def efficiency(self) -> float:
        """eta_s = F_u [kgf] / m [kg] (rulebook §12.6)."""
        return self.Fu_pred_kgf / self.mass.total_kg if self.mass.total_kg > 0 else math.nan

    @property
    def delta_at_Fu_mm(self) -> float:
        return self.delta_ref_mm * self.load_factor

    @property
    def governing_label(self) -> str:
        if self.governing_mode == "deflection":
            return f"deflection > {self.deflection_limit_mm:g} mm"
        return MODE_LABELS.get(self.governing_mode, self.governing_mode)

    @property
    def governing_detail(self) -> str:
        """Where the governing limit is reached: " in <member>", or for global buckling the
        shape of the mode in brackets. The member with the most strain energy alone can
        mislead: a small rib may hold the most energy while the whole span bows sideways."""
        if self.governing_mode == "global_buckling" and self.buckling is not None \
                and self.buckling.description:
            return f" ({self.buckling.description})"
        return f" in {self.governing_member}" if self.governing_member else ""

    @property
    def has_buckling_bracket(self) -> bool:
        """True if both ends of the buckling bracket for slack X-braces are finite."""
        F = self.Fu_buckling_braces_working_N
        return F is not None and math.isfinite(F) and math.isfinite(self.Fu_buckling_N)

    @property
    def Fu_buckling_braces_working_N(self) -> float | None:
        """Global buckling load with one strap of each slack X working (None if no X)."""
        b = self.buckling_braces_working
        if b is None:
            return None
        if math.isinf(b.lambda_cr) or math.isnan(b.lambda_cr):
            return b.lambda_cr
        return b.lambda_cr * self.P_ref_N

    @property
    def slack_note(self) -> str:
        """Which tension-only members were left out, and what that does to global buckling
        ("" if none)."""
        if not self.slack_members:
            return ""
        ids = ", ".join(self.slack_members[:12])
        more = f", … ({len(self.slack_members)} in total)" if len(self.slack_members) > 12 else ""
        note = (f"{len(self.slack_members)} tension-only member(s) would be in compression, so "
                f"they were left out and the bridge re-analysed without them (a slender brace "
                f"bows instead of failing, and then carries no load): {ids}{more}. A bowed "
                f"brace works again once a buckling mode stretches it, so the global buckling "
                f"load without them is a lower bound.")
        if self.has_buckling_bracket:
            F, b = self.Fu_buckling_braces_working_N, self.buckling_braces_working
            note += (f" With one strap of each X working ({len(self.braces_working_members)} "
                     f"crossing straps, each at half stiffness) it would be "
                     f"{n_to_kgf(F):,.0f} kgf ({F:,.0f} N; mode 1: {b.description}), against "
                     f"{n_to_kgf(self.Fu_buckling_N):,.0f} kgf without them. The real value "
                     f"lies in between; F_u,p uses the lower one.")
        return note

    def member(self, mid: str) -> MemberResult:
        return next(m for m in self.members if m.id == mid)

    def critical_members(self, n: int = 10) -> list[MemberResult]:
        return sorted(self.members, key=lambda m: m.U, reverse=True)[:n]

    def buckling_load_table(self, braces_working: bool = False) -> list[dict[str, Any]]:
        """One row per global buckling mode found: load factor, F_cr and the mode's shape.

        ``braces_working``: the modes with one strap of each slack X working (empty if the
        bridge has no such X). The ``case`` column tells the two apart in a combined file.
        """
        b = self.buckling_braces_working if braces_working else self.buckling
        if b is None:
            return []
        case = "X braces working" if braces_working else "as analysed"
        return [{"case": case, "mode": k + 1, "load_factor": round(lam, 4),
                 "F_cr_N": round(lam * self.P_ref_N, 1),
                 "F_cr_kgf": round(n_to_kgf(lam * self.P_ref_N), 1), "shape": desc}
                for k, (lam, desc) in enumerate(zip(b.lambdas, b.descriptions, strict=False))]

    def buckling_mode_table(self, mode: int = 1) -> list[dict[str, Any]]:
        """Node movements in global buckling mode ``mode`` (1 = the first), relative: the
        largest movement anywhere in the model is 1. Empty if there is no such mode."""
        b = self.buckling
        if b is None or not 1 <= mode <= len(b.modes_mm):
            return []
        shape = b.modes_mm[mode - 1]
        return [{"node": n.id, "x_mm": n.x_mm, "y_mm": n.y_mm, "z_mm": n.z_mm,
                 "dx_rel": round(shape[n.id][0], 4), "dy_rel": round(shape[n.id][1], 4),
                 "dz_rel": round(shape[n.id][2], 4),
                 "d_rel": round(math.sqrt(sum(c * c for c in shape[n.id])), 4)}
                for n in self.bridge.nodes if n.id in shape]

    def member_table(self) -> list[dict[str, Any]]:
        """One row per member, values at P_ref and at the predicted failure load."""
        rows = []
        lf = self.load_factor
        for m in self.members:
            rows.append({
                "member": m.id,
                "group": m.group,
                "section": m.section_label,
                "L_mm": round(m.L_mm, 1),
                "N_ref_N": round(m.N_N, 2),
                "N_at_Fu_N": round(m.N_N * lf, 1),
                "My_ref_Nmm": round(m.My_Nmm, 1),
                "Mz_ref_Nmm": round(m.Mz_Nmm, 1),
                "V_ref_N": round(math.hypot(m.Vy_N, m.Vz_N), 2),
                "N_t_R_N": round(m.cap.N_t_R, 1),
                "N_c_R_N": round(m.cap.N_c_R, 1),
                "P_cr_N": round(m.cap.P_cr, 1),
                "U_ref": round(m.U, 4),
                "U_at_Fu": round(m.U * lf, 3),
                "governing_check": MODE_LABELS[m.mode],
                "member_fails_at_kgf": round(n_to_kgf(self.P_ref_N / m.U), 1) if m.U else None,
            })
        return rows


def _max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def analyze(
    bridge: Bridge,
    material: Material | None = None,
    deflection_limit_mm: float = DEFLECTION_LIMIT_MM,
    include_buckling: bool = True,
) -> AnalysisResult:
    """Run the linear analysis and load-factor method. See the module docstring."""
    material = material or bridge_material(bridge)
    P_ref = bridge.load.P_ref_N
    model = build_model(bridge, material, P_ref)
    tension_only = [m.id for m in bridge.members if m.tension_only]
    try:
        if tension_only:
            # Each round can only switch members off (Pynite never switches one back on),
            # so this many rounds always suffice.
            model.analyze(check_stability=True, max_iter=len(tension_only) + 2,
                          member_tolerance=SLACK_TOLERANCE * P_ref)
        else:
            model.analyze_linear(check_stability=True)
    except Exception as exc:  # Pynite raises a bare Exception for a mechanism
        slack = [mid for mid in tension_only if not model.members[mid].active.get(COMBO, True)]
        hint = (
            f" Without the tension-only members that went into compression ({', '.join(slack)})"
            " the bridge cannot stand: either they are needed to hold it in place (remove "
            "tension_only from them), or a node is joined only by such members, for example "
            "where the two halves of a split X-brace cross (make each diagonal one member)."
            if slack else "")
        raise AnalysisError(
            f"The structure is unstable (a mechanism): {exc} Check that the supports and "
            f"member releases hold every part of the bridge in place.{hint}"
        ) from exc
    slack = [mid for mid in tension_only if not model.members[mid].active[COMBO]]

    nodes = bridge.node_map()
    secs = bridge.section_map()
    exclude = set(material.glue.exclude_groups)

    results: list[MemberResult] = []
    for mem in bridge.members:
        pm = model.members[mem.id]
        L = bridge.member_length_mm(mem, nodes)
        sec = secs[mem.section]
        b, d = sec.dims(material.stick)
        # Pynite axial force is compression-positive; flip to tension-positive.
        ax = -pm.axial_array(N_POINTS, COMBO)[1]
        N = float(ax[np.argmax(np.abs(ax))])
        My = _max_abs(pm.moment_array("My", N_POINTS, COMBO)[1])
        Mz = _max_abs(pm.moment_array("Mz", N_POINTS, COMBO)[1])
        vy = pm.shear_array("Fy", N_POINTS, COMBO)[1]
        vz = pm.shear_array("Fz", N_POINTS, COMBO)[1]
        Vy, Vz = _max_abs(vy), _max_abs(vz)
        # Both shear stresses peak at the centroid of a rectangle: check their resultant.
        V = float(np.max(np.hypot(vy, vz))) if vy.size else 0.0
        F_end = 0.0
        for x in (0.0, L):
            f = math.sqrt(
                pm.axial(x, COMBO) ** 2 + pm.shear("Fy", x, COMBO) ** 2
                + pm.shear("Fz", x, COMBO) ** 2
            )
            F_end = max(F_end, f)
        cap = capacities(sec, material, L, mem.K, mem.group, mem.glue_area_mm2)
        joint = mem.glue_area_mm2 is not None or mem.group not in exclude
        if mem.id in slack:  # left out of the model: no forces, nothing to check
            N = My = Mz = Vy = Vz = F_end = 0.0
            util = Utilisation(0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, "slack")
        else:
            util = utilisation(cap, N, My, Mz, V, F_end if joint else None)
        results.append(MemberResult(
            id=mem.id, group=mem.group, section=mem.section, section_label=sec.label(),
            L_mm=L, b_mm=b, d_mm=d, N_N=N, My_Nmm=My, Mz_Nmm=Mz, Vy_N=Vy, Vz_N=Vz,
            F_end_N=F_end, cap=cap, util=util,
        ))

    disp = {
        nid: (float(n.DX[COMBO]), float(n.DY[COMBO]), float(n.DZ[COMBO]))
        for nid, n in model.nodes.items() if nid in nodes
    }
    loads = plate_nodal_loads(bridge, P_ref)
    loaded = [nid for nid, F in loads.items() if F > 0]
    if not loaded:
        raise AnalysisError("The crusher load does not reach any deck support node.")
    delta_node = min(loaded, key=lambda nid: disp[nid][1])
    delta_ref = max(0.0, -disp[delta_node][1])

    U_max_member = max(results, key=lambda r: r.U) if results else None
    U_max = U_max_member.U if U_max_member else 0.0
    Fu_strength = P_ref / U_max if U_max > 0 else math.inf
    Fu_delta = P_ref * deflection_limit_mm / delta_ref if delta_ref > 0 else math.inf
    deck = diaphragm_terms(bridge)
    buck = global_buckling(model, bridge, extra=deck) if include_buckling else None
    working, working_ids = braces_working_terms(model, bridge, slack, material.stick)
    buck_working = (global_buckling(model, bridge, extra=deck + working)
                    if include_buckling and working else None)
    if buck is None or math.isinf(buck.lambda_cr):
        Fu_buck = math.inf  # not requested, or no compression-driven mode exists
    elif math.isnan(buck.lambda_cr) or buck.lambda_cr <= 0:
        Fu_buck = math.nan  # solver failed: NOT evaluated (reported, never treated as inf)
    else:
        Fu_buck = buck.lambda_cr * P_ref
    Fu = min(f for f in (Fu_strength, Fu_delta, Fu_buck) if not math.isnan(f))
    if Fu == Fu_strength:
        gov_member = U_max_member.id if U_max_member else None
        gov_mode = U_max_member.mode if U_max_member else "deflection"
    elif Fu == Fu_buck:
        gov_member, gov_mode = buck.key_member, "global_buckling"  # type: ignore[union-attr]
    else:
        gov_member, gov_mode = None, "deflection"

    reactions = []
    warnings: list[str] = []
    for nid in bridge.reaction_nodes():
        n = model.nodes[nid]
        r = Reaction(nid, float(n.RxnFX[COMBO]), float(n.RxnFY[COMBO]), float(n.RxnFZ[COMBO]))
        reactions.append(r)
        if -1e-6 * P_ref > r.FY_N:
            warnings.append(
                f"Support {nid} needs a downward (hold-down) reaction of {-r.FY_N:.1f} N at "
                f"P_ref. The bridge is not anchored (§8.3), so this support would lift off "
                f"and the model is unconservative."
            )
        H = math.hypot(r.FX_N, r.FZ_N)
        friction = FRICTION_COEFF * max(r.FY_N, 0.0)
        if H > friction + 1e-6 * P_ref:
            warnings.append(
                f"Support {nid} needs a horizontal reaction of {H:.1f} N at P_ref, more than "
                f"friction (mu = {FRICTION_COEFF}) can supply ({friction:.1f} N)."
                f" Unanchored supports may slide (§8.3), e.g. arch thrust."
            )
    if not math.isfinite(Fu):
        raise AnalysisError(
            "No member is stressed and no loaded node deflects, so the bridge has no "
            "structural load path (are the deck support nodes restrained?)."
        )
    if buck is not None and buck.note and not math.isinf(buck.lambda_cr):
        warnings.append(f"Global buckling: {buck.note}")
    if buck_working is not None and math.isnan(buck_working.lambda_cr):
        warnings.append(f"Global buckling with one strap of each slack X working: "
                        f"{buck_working.note}")
    if gov_mode == "buckling" and U_max_member is not None:
        N_gov = abs(U_max_member.N_N)
        N_big = max(abs(r.N_N) for r in results)
        if U_max_member.group in BRACE_GROUPS or N_gov < LOW_FORCE_RATIO * N_big:
            lf = Fu / P_ref
            warnings.append(
                f"The governing member {gov_member} ({U_max_member.group}) buckles under "
                f"{N_gov * lf:,.0f} N at F_u,p, {N_gov / N_big:.1%} of the largest member force "
                f"({N_big * lf:,.0f} N). A brace is often squeezed only because the chords "
                f"shorten, not because it carries the load. If {gov_member} is a slender brace, "
                f"it bows without the bridge failing and still works in tension: mark it "
                f"tension_only: true in the bridge file to analyse it that way."
            )
    unsplit = list(getattr(model, "unsplit_members", []))
    if buck is not None and unsplit and math.isfinite(Fu_buck):
        which = ("every member (the model has more than "
                 f"{REFINE_MAX_MEMBERS} members)" if len(unsplit) == splittable_count(bridge)
                 else f"{len(unsplit)} member(s) that could not be split")
        warnings.append(f"Global buckling used one element for {which}, so F_cr may be a "
                        f"few percent too high.")
    if math.isfinite(Fu_buck) and Fu < Fu_buck and Fu / Fu_buck > NEAR_BUCKLING_RATIO:
        ratio = Fu / Fu_buck
        warnings.append(
            f"F_u,p is {ratio:.0%} of the global buckling load ({n_to_kgf(Fu_buck):,.0f} kgf; "
            f"mode 1: {buck.description}). Near that load, sway from crooked sticks or "
            f"uneven supports grows about {1 / (1 - ratio):.1f}x (P-delta, not modelled in "
            f"v0.1), so the real failure load may be lower. Stiffening that mode adds margin."
        )

    return AnalysisResult(
        bridge=bridge, material=material, P_ref_N=P_ref, nodal_loads_N=loads,
        plate=plate_placement(bridge), members=results, displacements_mm=disp,
        reactions=reactions, delta_ref_mm=delta_ref, delta_node=delta_node,
        Fu_strength_N=Fu_strength, Fu_deflection_N=Fu_delta, Fu_buckling_N=Fu_buck, Fu_pred_N=Fu,
        governing_member=gov_member, governing_mode=gov_mode,
        mass=bridge_mass(bridge, material), deflection_limit_mm=deflection_limit_mm,
        buckling=buck, buckling_unsplit_members=unsplit, slack_members=slack,
        buckling_braces_working=buck_working,
        braces_working_members=working_ids if buck_working is not None else [],
        warnings=warnings, fe_model=model,
    )
