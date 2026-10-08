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
joint slip, or load redistribution after the first member fails.
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
from bridgesim.model import COMBO, build_model
from bridgesim.schema import Bridge, Material
from bridgesim.stability import BucklingResult, global_buckling
from bridgesim.units import n_to_kgf

DEFLECTION_LIMIT_MM = 50.0


class AnalysisError(ValueError):
    """The model cannot give a meaningful prediction (e.g. no load path)."""

N_POINTS = 11
#: Assumed static friction coefficient between the bridge and the steel platform, used only
#: to warn about horizontal support reactions (the bridge is not anchored, §8.3).
FRICTION_COEFF = 0.4


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
        return MODE_LABELS.get(self.governing_mode, self.governing_mode)

    def member(self, mid: str) -> MemberResult:
        return next(m for m in self.members if m.id == mid)

    def critical_members(self, n: int = 10) -> list[MemberResult]:
        return sorted(self.members, key=lambda m: m.U, reverse=True)[:n]

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
    model.analyze_linear(check_stability=True)

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
        cap = capacities(sec, material, L, mem.K)
        util = utilisation(cap, N, My, Mz, V, None if mem.group in exclude else F_end)
        results.append(MemberResult(
            id=mem.id, group=mem.group, section=mem.section, section_label=sec.label(),
            L_mm=L, b_mm=b, d_mm=d, N_N=N, My_Nmm=My, Mz_Nmm=Mz, Vy_N=Vy, Vz_N=Vz,
            F_end_N=F_end, cap=cap, util=util,
        ))

    disp = {
        nid: (float(n.DX[COMBO]), float(n.DY[COMBO]), float(n.DZ[COMBO]))
        for nid, n in model.nodes.items()
    }
    loads = plate_nodal_loads(bridge, P_ref)
    loaded = [nid for nid, F in loads.items() if F > 0]
    delta_node = min(loaded, key=lambda nid: disp[nid][1])
    delta_ref = max(0.0, -disp[delta_node][1])

    U_max_member = max(results, key=lambda r: r.U) if results else None
    U_max = U_max_member.U if U_max_member else 0.0
    Fu_strength = P_ref / U_max if U_max > 0 else math.inf
    Fu_delta = P_ref * deflection_limit_mm / delta_ref if delta_ref > 0 else math.inf
    buck = global_buckling(model, bridge) if include_buckling else None
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
        if r.FY_N > 0 and H > FRICTION_COEFF * r.FY_N:
            warnings.append(
                f"Support {nid} needs a horizontal reaction of {H:.1f} N at P_ref, more than "
                f"friction (mu = {FRICTION_COEFF}) can supply ({FRICTION_COEFF * r.FY_N:.1f} N)."
                f" Unanchored supports may slide (§8.3), e.g. arch thrust."
            )
    if not math.isfinite(Fu):
        raise AnalysisError(
            "No member is stressed and no loaded node deflects, so the bridge has no "
            "structural load path (are the deck support nodes restrained?)."
        )
    if buck is not None and buck.note and not math.isinf(buck.lambda_cr):
        warnings.append(f"Global buckling: {buck.note}")

    return AnalysisResult(
        bridge=bridge, material=material, P_ref_N=P_ref, nodal_loads_N=loads,
        plate=plate_placement(bridge), members=results, displacements_mm=disp,
        reactions=reactions, delta_ref_mm=delta_ref, delta_node=delta_node,
        Fu_strength_N=Fu_strength, Fu_deflection_N=Fu_delta, Fu_buckling_N=Fu_buck, Fu_pred_N=Fu,
        governing_member=gov_member, governing_mode=gov_mode,
        mass=bridge_mass(bridge, material), deflection_limit_mm=deflection_limit_mm,
        buckling=buck, warnings=warnings, fe_model=model,
    )
