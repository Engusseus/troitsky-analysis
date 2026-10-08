"""Member and joint capacity checks (linear, at the reference load).

Capacities (wood parallel to grain, all strengths from the material file):

* tension            N_t,R = f_t A
* compression        N_c,R = min(f_c A, P_cr),  P_cr = pi^2 E I_min / (K L)^2
* bending            M_R,y = f_b S_y,  M_R,z = f_b S_z
* member shear       V_R = f_v A / 1.5          (max shear stress 1.5 V / A in a rectangle;
                                               V = resultant of the two local shears)
* glued joint        F_R = tau_g A_glue,  A_glue = overlap x w x faces,  w = max(b, d)

Utilisation of a member at the reference load (linear interaction, conservative):

    U = |N| / N_R + |M_y| / M_R,y + |M_z| / M_R,z

with N_R = N_t,R or N_c,R by the sign of N. Member shear and joint utilisations are
separate checks. Buckling uses the full member length with K (default 1) and the weaker
axis, which is conservative for braced chords. Torsion is ignored in v0.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bridgesim.schema import Material, Section
from bridgesim.sections import euler_critical_load

MODE_LABELS = {
    "tension": "tension",
    "crushing": "compression (crushing)",
    "buckling": "buckling",
    "bending_z": "bending (primary plane)",
    "bending_y": "bending (out of plane)",
    "shear": "member shear",
    "joint": "glued joint shear",
    "deflection": "deflection limit",  # AnalysisResult.governing_label adds the limit
    "global_buckling": "global elastic buckling (instability)",
}


@dataclass
class Capacities:
    N_t_R: float
    N_c_crush: float
    P_cr: float
    M_R_y: float
    M_R_z: float
    V_R: float
    F_joint_R: float

    @property
    def N_c_R(self) -> float:
        return min(self.N_c_crush, self.P_cr)


def capacities(section: Section, material: Material, L_mm: float, K: float) -> Capacities:
    pr = section.props(material.stick)
    glue = material.glue
    w = max(pr.b_mm, pr.d_mm)
    return Capacities(
        N_t_R=material.f_t_MPa.value * pr.A_mm2,
        N_c_crush=material.f_c_MPa.value * pr.A_mm2,
        P_cr=euler_critical_load(material.E_MPa.value, pr.I_min_mm4, L_mm, K),
        M_R_y=material.f_b_MPa.value * pr.Sy_mm3,
        M_R_z=material.f_b_MPa.value * pr.Sz_mm3,
        V_R=material.f_v_MPa.value * pr.A_mm2 / 1.5,
        F_joint_R=glue.tau_g_MPa.value * glue.overlap_mm.value * w * glue.faces.value,
    )


@dataclass
class Utilisation:
    U_axial: float
    U_My: float
    U_Mz: float
    U_interaction: float
    U_shear: float
    U_joint: float | None
    U: float
    mode: str  # key of MODE_LABELS

    @property
    def mode_label(self) -> str:
        return MODE_LABELS[self.mode]


def _ratio(demand: float, capacity: float) -> float:
    if demand == 0:
        return 0.0
    return math.inf if capacity <= 0 else abs(demand) / capacity


def utilisation(
    cap: Capacities,
    N: float,
    My: float,
    Mz: float,
    V: float,
    F_end: float | None,
) -> Utilisation:
    """Utilisations for one member. ``N`` tension-positive, all values at the same load.

    ``F_end`` is the largest resultant member-end force (None if no joint check applies).
    """
    if N >= 0:
        U_ax, axial_mode = _ratio(N, cap.N_t_R), "tension"
    else:
        U_ax = _ratio(N, cap.N_c_R)
        axial_mode = "buckling" if cap.P_cr < cap.N_c_crush else "crushing"
    U_My, U_Mz = _ratio(My, cap.M_R_y), _ratio(Mz, cap.M_R_z)
    U_int = U_ax + U_My + U_Mz
    U_v = _ratio(V, cap.V_R)
    U_j = None if F_end is None else _ratio(F_end, cap.F_joint_R)

    # Name the interaction by its dominant term.
    terms = {axial_mode: U_ax, "bending_z": U_Mz, "bending_y": U_My}
    int_mode = max(terms, key=terms.get)  # type: ignore[arg-type]
    candidates = {int_mode: U_int, "shear": U_v}
    if U_j is not None:
        candidates["joint"] = U_j
    mode = max(candidates, key=candidates.get)  # type: ignore[arg-type]
    return Utilisation(
        U_axial=U_ax, U_My=U_My, U_Mz=U_Mz, U_interaction=U_int, U_shear=U_v, U_joint=U_j,
        U=candidates[mode], mode=mode,
    )
