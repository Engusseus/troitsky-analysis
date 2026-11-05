"""The Troitsky crushing test (rulebook §12.5), simulated.

Protocol, as at the competition:

1. the bridge is set on the press's flat base platform and **settles under its own
   weight** (an unstable bridge fails here: "instability (failure under self-weight)");
2. the steel **loading plate** (200 x 90 mm, two 15 mm plates) is laid on the deck at
   mid-span and settles too;
3. the **jack** comes down onto the plate's hemispherical seat and keeps moving down at a
   constant speed; its force is the load-cell reading.

The run stops when the bridge has failed by one of the §12.5 criteria and the load has
clearly dropped (so the collapse is visible), or the deflection passes the 50 mm limit.

Failure criteria tracked (first time each happens):

* **Deflection**: the deck under the plate has gone down more than 50 mm;
* **Contact**: a part that was not resting on the platform at the start touches it (or
  a part touches the press frame);
* **Shear / joint failure**: a glued joint has fully separated, or a member has split in
  shear;
* **Instability**: the bridge collapses under self-weight before loading.

The reported ultimate load F_u is the highest jack force reached before the bridge is
judged failed: by a load drop of 30 % (collapse), by the 50 mm deflection limit or by
contact with the platform. A bridge that fails under its own weight has F_u = 0. A glued
joint letting go inside the bridge is also a §12.5 failure ("shear, including joint
failure") but the judges only see it when the bridge visibly gives way, which shows as a
load drop; so the load at the first complete joint separation is reported separately
(``Fu_first_joint_failure_kgf``) together with when each criterion was first met. Every
damage event (first crack of a member, every joint that lets go, floss breaking) is logged
with the time and the jack force, so the failure sequence can be read like a test
report.

The simulation is dynamic but slow compared with the bridge's vibrations: quasi-static.
The kinetic energy is recorded and reported against the work done, so it can be checked.
Time is compressed (the real jack moves at ~mm/min, the simulated one at ~mm/s); since
the material laws have no rate effect, this does not change the result, and the replay
maps simulated time back to real test time.
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from bridgesim.sim import solver as S
from bridgesim.sim.model import Model

JOINT_FAILED = 0.9  # share of a joint's glued area let go: the joint has separated
JOINT_CRACKED = 0.3  # ... the joint is cracking (an event, and the jack holds)
KGF = 9.80665


@dataclass
class Frame:
    t: float
    travel_mm: float
    force_N: float
    deflection_mm: float
    x: np.ndarray  # (n, 3) float32
    q: np.ndarray  # (n, 4) quaternion float32
    beam_state: np.ndarray  # uint8: 0 ok, 1 cracked, 2 broken
    shell_state: np.ndarray
    patch_damage: np.ndarray  # share of each joint's glued area let go (0..1)
    cable_ok: np.ndarray


@dataclass
class Event:
    t: float
    travel_mm: float
    force_N: float
    kind: str
    what: str
    detail: str = ""
    where: tuple[float, float, float] | None = None
    fraction: float | None = None  # share of a glued joint's area that has let go

    def as_dict(self) -> dict:
        return {"t_s": round(float(self.t), 5),
                "jack_travel_mm": round(float(self.travel_mm), 2),
                "load_N": round(float(self.force_N), 1),
                "load_kgf": round(float(self.force_N) / KGF, 2),
                "kind": self.kind, "what": self.what, "detail": self.detail,
                "where_mm": [round(float(v), 1) for v in self.where] if self.where is not None
                else None,
                "fraction_separated": None if self.fraction is None
                else round(float(self.fraction), 3)}


@dataclass
class CrushResult:
    Fu_N: float
    Fu_kgf: float
    failure_mode: str
    failure_detail: str
    t_failure: float
    deflection_at_Fu_mm: float
    curve: dict[str, list[float]]  # t, travel, force, deflection, kinetic, ...
    events: list[Event]
    frames: list[Frame]
    criteria: dict[str, dict | None]
    dt: float
    steps: int
    wall_s: float
    added_mass_kg: float
    added_mass_frac: float
    settle_sag_mm: float
    kinetic_ratio_max: float
    model: Model = field(repr=False)
    notes: list[str] = field(default_factory=list)
    speed_mm_s: float = 0.0
    holds: int = 0  # times the jack was held while something broke
    hold_time_s: float = 0.0
    governing: str = ""  # what broke when the load fell ("member shear: <part>", ...)
    competing: list[str] = field(default_factory=list)  # other breaks within 10 % of F_u


def _quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices -> quaternions (w, x, y, z)."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = np.sqrt(np.maximum(0, 1 + tr)) / 2
    x = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2
    x = np.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = np.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = np.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    return np.stack([w, x, y, z], axis=1)


def nodal_stiffness(m: Model, s: S.Settings) -> tuple[np.ndarray, np.ndarray]:
    """Upper-bound estimates (Gershgorin-style) of the translational and rotational
    stiffness at every node, for the stable time step and selective mass scaling."""
    n = m.n_nodes
    kt = np.zeros(n)
    kr = np.zeros(n)
    if len(m.b_i):
        A = m.b_fA.sum(1)
        Iy = (m.b_fA * m.b_fz**2).sum(1)
        Iz = (m.b_fA * m.b_fy**2).sum(1)
        Im = np.maximum(Iy, Iz)
        L = m.b_L0
        kax = m.b_E * A / L + 12 * m.b_E * Im / L**3
        krot = 6 * m.b_E * Im / L + m.b_GJ / L + 6 * m.b_E * Im / L**2
        for idx in (m.b_i, m.b_j):
            np.add.at(kt, idx, 2 * kax)
            np.add.at(kr, idx, 2 * krot)
    if len(m.s_nodes):
        absK = np.abs(m.s_K)
        row = absK.sum(2).reshape(len(m.s_nodes), 3, 6)
        np.add.at(kt, m.s_nodes.ravel(), row[:, :, :3].max(2).ravel())
        np.add.at(kr, m.s_nodes.ravel(), row[:, :, 3:].max(2).ravel())
    if len(m.c_a):
        k = np.maximum(m.c_Kn, m.c_Ks) * m.c_area
        for att in (m.c_a, m.c_b):
            nd = m.a_node[att]
            r2 = (m.a_r0[att] ** 2).sum(1)
            np.add.at(kt, nd, 2 * k)
            np.add.at(kr, nd, 2 * k * r2)
    if len(m.k_a):
        k = m.k_EA / m.k_L0
        for att in (m.k_a, m.k_b):
            np.add.at(kt, m.a_node[att], 2 * k)
    for att, kc, near in ((m.p_att, s.k_platform, True), (m.q_att, s.k_plate, False)):
        if not len(att):
            continue
        nd = m.a_node[att]
        if near:
            y = m.X[nd, 1] + m.a_r0[att, 1]
            sel = y - m.table_y < 60
            nd = nd[sel]
            r2 = (m.a_r0[att][sel] ** 2).sum(1)
        else:
            r2 = (m.a_r0[att] ** 2).sum(1)
        np.add.at(kt, nd, 2 * kc)
        np.add.at(kr, nd, 2 * kc * r2)
        if not near:
            kt[m.plate_node] += 2 * kc * len(att)
            kr[m.plate_node] += 2 * kc * len(att) * float(np.max(m.plate_half) ** 2)
    kt[m.plate_node] += 2 * (s.k_jack + s.k_jack_lat)
    return kt, kr


def _beam_local_K(m: Model) -> np.ndarray:
    """Linear-elastic 12 x 12 local stiffness of every beam (for frequency estimates)."""
    nb = len(m.b_i)
    A = m.b_fA.sum(1)
    Iz = (m.b_fA * m.b_fy**2).sum(1)
    Iy = (m.b_fA * m.b_fz**2).sum(1)
    L = m.b_L0
    E = m.b_E
    K = np.zeros((nb, 12, 12))

    def put(i, j, v):
        K[:, i, j] += v
        if i != j:
            K[:, j, i] += v

    ka = E * A / L
    put(0, 0, ka); put(6, 6, ka); put(0, 6, -ka)  # noqa: E702
    kt = m.b_GJ / L
    put(3, 3, kt); put(9, 9, kt); put(3, 9, -kt)  # noqa: E702
    for I_, (v, r, sg) in ((Iz, (1, 5, 1.0)), (Iy, (2, 4, -1.0))):
        a = 12 * E * I_ / L**3
        b = 6 * E * I_ / L**2 * sg
        c = 4 * E * I_ / L
        d = 2 * E * I_ / L
        put(v, v, a); put(v + 6, v + 6, a); put(v, v + 6, -a)  # noqa: E702
        put(v, r, b); put(v, r + 6, b); put(v + 6, r, -b); put(v + 6, r + 6, -b)  # noqa: E702
        put(r, r, c); put(r + 6, r + 6, c); put(r, r + 6, d)  # noqa: E702
    return K


def _elem_lambda(Ke: np.ndarray, md: np.ndarray) -> np.ndarray:
    """Largest eigenvalue of M^-1 K for a batch of elements with diagonal mass md."""
    s = 1.0 / np.sqrt(md)
    Ks = Ke * s[:, :, None] * s[:, None, :]
    return np.linalg.eigvalsh(Ks)[:, -1]


def _entity_lambdas(m: Model, s: S.Settings, mass: np.ndarray, rot: np.ndarray,
                    Kb: np.ndarray):
    """(lambda_max, node index array) for every element-like entity."""
    out = []
    if len(m.b_i):
        nodes = np.stack([m.b_i, m.b_j], 1)
        md = np.concatenate([np.repeat(mass[nodes][:, :, None], 3, 2),
                             np.repeat(rot[nodes][:, :, None], 3, 2)], axis=2)
        md = md.reshape(len(m.b_i), 12)
        lam = _elem_lambda(Kb, md)
        out.append((np.repeat(lam[:, None], 2, 1), nodes))
    if len(m.s_nodes):
        nodes = m.s_nodes
        md = np.concatenate([np.repeat(mass[nodes][:, :, None], 3, 2),
                             np.repeat(rot[nodes][:, :, None], 3, 2)], axis=2)
        lam = _elem_lambda(m.s_K, md.reshape(len(nodes), 18))
        out.append((np.repeat(lam[:, None], 3, 1), nodes))

    def pair(ka, att_a, att_b):
        """Gershgorin share of a two-point spring for each of its nodes:
        k (1/m_a* + 1/sqrt(m_a* m_b*)), m* the effective mass at the arm."""
        na, nb = m.a_node[att_a], m.a_node[att_b]
        ra = (m.a_r0[att_a] ** 2).sum(1)
        rb = (m.a_r0[att_b] ** 2).sum(1)
        ca = 1 / mass[na] + ra / rot[na]
        cb = 1 / mass[nb] + rb / rot[nb]
        cross = np.sqrt(ca * cb)
        return np.stack([ka * (ca + cross), ka * (cb + cross)], 1), np.stack([na, nb], 1)

    if len(m.c_a):
        out.append(pair(np.maximum(m.c_Kn, m.c_Ks) * m.c_area * 1.5, m.c_a, m.c_b))
    if len(m.k_a):
        out.append(pair(m.k_EA / m.k_L0, m.k_a, m.k_b))
    for att, kc, near in ((m.p_att, s.k_platform, True), (m.q_att, s.k_plate, False)):
        if not len(att):
            continue
        nd = m.a_node[att]
        r2 = (m.a_r0[att] ** 2).sum(1)
        lam = kc * 1.5 * (1 / mass[nd] + r2 / rot[nd])
        if near:
            y = m.X[nd, 1] + m.a_r0[att, 1]
            lam = np.where(y - m.table_y < 60, lam, 0.0)
        out.append((lam[:, None], nd[:, None]))
    P = m.plate_node
    if P >= 0:
        nq = max(1, len(m.q_att) // 4)
        out.append((np.array([[(s.k_jack + s.k_plate * nq) / mass[P]]]), np.array([[P]])))
    return out


def _node_lambda(m: Model, s: S.Settings, mass, rot, Kb, lam_t):
    """Estimate of the highest frequency squared seen by each node: the largest beam
    and the largest shell eigenvalue around it, plus every spring and contact on it
    (those add up). Returns (per-node lambda, worst ratio to the target)."""
    lam_node = np.zeros(len(mass))
    for kind, (lam, nodes) in zip(("beam", "shell", "glue", "cable", "platform", "plate",
                                   "jack"), _entity_lambdas(m, s, mass, rot, Kb),
                                  strict=False):
        acc = np.zeros(len(mass))
        for col in range(nodes.shape[1]):
            if kind in ("beam", "shell"):
                np.maximum.at(acc, nodes[:, col], lam[:, col])
            else:
                np.add.at(acc, nodes[:, col], lam[:, col])
        lam_node += acc
    return lam_node, float(lam_node.max() / lam_t)


def choose_dt(m: Model, s: S.Settings, max_added: float = 0.05,
              candidates=(3e-6, 4e-6, 5e-6, 6e-6, 8e-6, 1e-5, 1.25e-5, 1.5e-5, 2e-5)
              ) -> tuple[float, np.ndarray, np.ndarray]:
    """Largest stable time step that needs at most ``max_added`` (fraction of the bridge
    mass) of selective mass scaling, with the scaled masses and rotational inertias.

    Stability of central differences needs dt <= 2 / omega_max. omega_max is bounded by
    the largest element frequency (each element's own eigenproblem with its nodes'
    lumped masses), the standard estimate of explicit codes. Rotational inertia is
    scaled freely (it hardly changes a slow test); added translational mass is limited
    and reported."""
    Kb = _beam_local_K(m) if len(m.b_i) else None
    total = m.mass.sum()
    best = None
    for dt in candidates:
        lam_t = (2 * 0.8 / dt) ** 2
        mass = m.mass.copy()
        rot = m.rot_inertia.copy()
        for rnd in range(40):
            lam_node, worst = _node_lambda(m, s, mass, rot, Kb, lam_t)
            if worst <= 1.0:
                break
            fac = np.maximum(lam_node / lam_t * 1.02, 1.0)
            rot *= fac
            if rnd >= 4:
                mass *= fac ** 1.5  # partner terms only fall as 1/sqrt(f)
        add = (mass - m.mass).sum()
        if worst <= 1.0 + 1e-9 and add <= max_added * total:
            best = (dt, mass, rot)
    if best is None:
        dt = candidates[0]
        lam_t = (2 * 0.8 / dt) ** 2
        mass = m.mass.copy()
        rot = m.rot_inertia.copy()
        for _ in range(20):
            lam_node, worst = _node_lambda(m, s, mass, rot, Kb, lam_t)
            if worst <= 1.0:
                break
            fac = np.maximum(lam_node / lam_t * 1.02, 1.0)
            mass *= fac
            rot *= fac
        best = (dt, mass, rot)
    return best


FIDELITY = {  # mesh scale, simulated jack speed (mm/s)
    "quick": (1.6, 200.0),     # screening design changes (rank designs at one fidelity)
    "standard": (1.0, 100.0),
    "fine": (0.75, 50.0),      # convergence check
}


@dataclass
class CrushOptions:
    speed_mm_s: float = 40.0  # simulated jack speed
    settle_s: float = 0.03  # gravity ramp
    settle_max_s: float = 0.4
    max_travel_mm: float = 70.0
    deflection_limit_mm: float = 50.0
    # judged failed when the load falls below drop_ratio x its peak: a 30 % fall is a
    # fracture the judges see and the load cell shows (§12.5 takes the load "at the
    # moment of failure"), even if the bridge then picks up some load again
    drop_ratio: float = 0.7
    after_failure_mm: float = 15.0  # keep going this far past failure (collapse replay)
    frame_mm: float = 0.15  # replay frame every this much jack travel
    alpha: float = 25.0
    # after the bridge is judged failed (F_u is known) the collapse runs with light damping
    # so wreckage falls at its real speed in the replay (alpha caps it at about g / alpha)
    alpha_collapse: float = 2.0
    max_added_mass: float = 0.05
    dt_target: float = 3e-6
    max_steps: int = 400_000
    self_contact: bool = True  # contact between parts not touching at the start
    # quasi-static failures: when something breaks (a joint, a member, a plate, floss)
    # the jack slows to hold_speed_frac of its speed until nothing more has broken for
    # hold_settle_s and the bridge has settled (its kinetic energy back within
    # hold_ke_settle times that of steady loading, its load steady within hold_force_tol
    # over the last hold_settle_s / 2), at most hold_max_s, as at the competition, where
    # a failure runs its course long before the jack has moved
    hold_on_failure: bool = True
    hold_speed_frac: float = 0.02
    hold_ramp_s: float = 0.002
    hold_settle_s: float = 0.004
    hold_max_s: float = 0.03
    hold_ke_settle: float = 2.0
    hold_force_tol: float = 0.01
    progress: object = None  # callable(fraction, message)


class _Unstable(RuntimeError):
    pass


def _node_diag(m: Model, st: S.State, n: int) -> str:
    """What is attached to a node (to explain a numerical problem)."""
    sh = np.where((m.s_nodes == n).any(1))[0] if len(m.s_nodes) else []
    bm = np.where((m.b_i == n) | (m.b_j == n))[0] if len(m.b_i) else []
    gl = np.where((m.a_node[m.c_a] == n) | (m.a_node[m.c_b] == n))[0] if len(m.c_a) else []
    pp = np.where(m.a_node[m.p_att] == n)[0]
    return (f"node {n}: m {m.mass[n]:.3g} t, I {m.rot_inertia[n]:.3g}, |w| "
            f"{np.linalg.norm(st.w[n]):.3g} rad/s, shells {len(sh)} dmg "
            f"{[round(float(d), 2) for d in st.s_dmg[sh]]}, beams {len(bm)} alive "
            f"{[bool(a) for a in st.b_alive[bm]]}, glue {len(gl)} dmg max "
            f"{float(st.c_d[gl].max()) if len(gl) else 0:.2f}, platform points {len(pp)} "
            f"touching {int(st.p_touch[pp].sum())}")


LOAD_CELL_MM = 0.3  # time constant of the online load-cell filter, in jack travel


def reported_load(t: np.ndarray, raw: np.ndarray, speed_mm_s: float,
                  travel: np.ndarray | None = None) -> np.ndarray:
    """The load reported for a crush: the raw jack force smoothed by a centred Hann
    window 0.9 mm of jack travel wide (three online load-cell time constants at full
    speed), along the jack travel: the load against travel, as a test record plots it.
    Centred, so it has no lag (the rising branch, the stiffness and F_u are where they
    are), while it removes the stress waves ringing through the bridge. Along the
    travel, a hold (the jack nearly still while something breaks) is one point of the
    curve, weighted by the little travel it covers: the load can neither jump nor keep
    rising at constant travel there. Without ``travel``: along time at full speed."""
    raw = np.asarray(raw, float)
    if len(raw) < 3:
        return raw.copy()
    u = (np.asarray(travel, float) / max(speed_mm_s, 1e-9) if travel is not None
         else np.asarray(t, float))  # the time the jack needs at full speed
    u = np.maximum.accumulate(u)
    W = 3 * LOAD_CELL_MM / max(speed_mm_s, 1e-9)
    du = np.gradient(u)
    du = np.maximum(du, 1e-6 * (float(np.median(du[du > 0])) if (du > 0).any() else 1.0))
    # odd reflection at both ends: a load rising from the start (or still changing at
    # the end) is not biased by a one-sided window there
    m0 = int(np.searchsorted(u, u[0] + W / 2))
    m1 = len(u) - int(np.searchsorted(u, u[-1] - W / 2))
    m0, m1 = max(1, min(m0, len(u) - 1)), max(1, min(m1, len(u) - 1))
    U = np.r_[2 * u[0] - u[m0:0:-1], u, 2 * u[-1] - u[-2:-m1 - 2:-1]]
    F = np.r_[2 * raw[0] - raw[m0:0:-1], raw, 2 * raw[-1] - raw[-2:-m1 - 2:-1]]
    D = np.r_[du[m0:0:-1], du, du[-2:-m1 - 2:-1]]
    lo = np.searchsorted(U, u - W / 2)
    hi = np.searchsorted(U, u + W / 2, side="right")
    out = np.empty_like(raw)
    for i in range(len(u)):
        a, b = lo[i], hi[i]
        k = 0.5 * (1 + np.cos(2 * np.pi * (U[a:b] - u[i]) / W)) * D[a:b]
        out[i] = float(k @ F[a:b]) / max(float(k.sum()), 1e-300)
    return out


BREAKS = ("joint_failure", "joint_cracking", "member_fracture", "member_shear",
          "plate_crack", "floss_break", "member_crushing")


def attribute(before: list[Event], t_c: float, peak: float, peak_travel: float,
              speed_mm_s: float) -> tuple[Event | None, list[str]]:
    """The governing failure and the competing ones, from the events up to the failure
    call. Governing: the break that started the fall of the load, i.e. the first break
    from ``t_c``, where the reported load last stood within 5 % of F_u (``peak``); a
    break before it was survived (the load came back). A quarter of the load-cell time
    constant of slack before ``t_c``: the centred smoothing leads a sudden drop. Among
    breaks at the same moment a clean break comes before a joint that only cracked, and
    a member split within 2 ms of a member fracture is put down to the fracture (its
    stress wave). Without a break from ``t_c``: the last survived break within 10 % of
    F_u; else one within 3 mm of jack travel of the peak. Competing: the other breaks
    within 10 % of F_u that were survived, or that came at the same moment as the
    governing one (not the cascade after it), as '<kind>: <part>'."""
    tol = 0.25 * LOAD_CELL_MM / max(speed_mm_s, 1e-9)

    def clean(evs):  # a clean break before a joint that only cracked or a member crushing
        return [e for e in evs if e.kind not in ("joint_cracking", "member_crushing")] or \
            [e for e in evs if e.kind != "joint_cracking"] or evs

    falling = [e for e in before if e.kind in BREAKS and e.t >= t_c - tol]
    survived = [e for e in before if e.kind in BREAKS and e.t < t_c - tol
                and e.force_N >= 0.9 * peak]
    if falling:
        first = clean([e for e in falling if e.t <= falling[0].t + 2 * tol])[0]
    elif survived:
        first = clean(survived)[-1]
    else:
        near = [e for e in before if e.kind in BREAKS
                and abs(e.travel_mm - peak_travel) <= 3.0]
        first = near[0] if near else next(
            (e for e in before if e.kind in BREAKS and e.kind != "joint_cracking"), None)
    if first is not None and first.kind == "member_shear":
        # a member breaking sends a stress wave that can split its neighbours (or itself)
        # in the same instant: the break is the cause, not the splits
        root = [e for e in before if e.kind == "member_fracture"
                and abs(e.t - first.t) <= 0.002]
        if root:
            first = root[0]
    t_gov = first.t if first is not None else t_c
    gov_what = first.what if first is not None else ""
    competing = list(dict.fromkeys(
        f"{e.kind.replace('_', ' ')}: {e.what}"
        for e in clean(survived + [e for e in falling if e.t <= t_gov + tol])
        if e.what != gov_what))[:8]
    return first, competing


def run_crush(m: Model, opt: CrushOptions | None = None) -> CrushResult:
    """Simulate the crushing test. If the explicit integration ever starts to diverge
    (a node faster than 1 km/s or a non-finite value), the run restarts with a smaller
    time step, so a result is never produced from an unstable integration."""
    opt = opt or CrushOptions()
    X0, m0, r0 = m.X.copy(), m.mass.copy(), m.rot_inertia.copy()
    att0 = (m.a_node.copy(), m.a_r0.copy())
    notes = []
    dt_target = opt.dt_target
    for _attempt in range(4):
        try:
            res = _run_crush_once(m, opt, dt_target)
            res.notes = notes + res.notes
            return res
        except _Unstable as exc:
            if opt.progress:
                opt.progress(0.0, f"unstable: {exc}; restarting with a smaller time step")
            notes.append(f"integration diverged at dt = {dt_target * 1e6:.1f} us ({exc}); "
                         "restarted with a smaller step")
            m.X, m.mass, m.rot_inertia = X0.copy(), m0.copy(), r0.copy()
            m.a_node, m.a_r0 = att0[0].copy(), att0[1].copy()
            dt_target *= 0.5
    raise RuntimeError("the simulation is unstable even with a small time step: "
                       + "; ".join(notes))


def _run_crush_once(m: Model, opt: CrushOptions, dt_target: float) -> CrushResult:
    s = S.Settings(dt=1e-5, alpha=opt.alpha,
                   mu_platform=m.params.get("friction_platform", 0.35),
                   mu_plate=m.params.get("friction_plate", 0.3),
                   k_jack=m.params.get("crusher_k_N_per_mm", 20000.0),
                   k_jack_lat=m.params.get("crusher_k_lat_N_per_mm", 2000.0))
    if opt.progress:
        opt.progress(0.0, f"model: {m.n_nodes} nodes, {len(m.b_i)} beams, "
                          f"{len(m.s_nodes)} shells, {len(m.c_a)} glue points; estimating "
                          "the stable time step")
    dt, mass, rot, scale_log = stable_dt(m, s, dt_target, opt.max_added_mass)
    if opt.progress:
        opt.progress(0.0, f"time step {dt * 1e6:.2f} us; settling under self-weight")
    s.dt = dt
    S.contact_damping_coeffs(m, s, mass)
    phys_mass = m.mass.copy()
    added = float((mass - phys_mass).sum())
    m.mass = mass
    m.rot_inertia = rot
    m.added_mass_kg = added * 1000
    t0 = time.time()
    # gravity uses the physical mass, inertia the scaled one
    grav_ratio = phys_mass / mass
    P = m.plate_node
    seat_y0 = m.X[P, 1] + m.plate_half[1]
    st = S.initial_state(m, jack_y=seat_y0 + 50.0)
    if opt.self_contact:
        from bridgesim.sim import contact

        st.sc = contact.setup(m, m.X, mass, dt)
    F = np.zeros((m.n_nodes, 3))
    Mo = np.zeros((m.n_nodes, 3))
    events: list[Event] = []
    criteria: dict[str, dict | None] = {"deflection": None, "contact": None,
                                         "joint_or_shear": None, "instability": None}
    curve = {k: [] for k in ("t", "travel", "force", "deflection", "kinetic", "plate_load",
                             "glue_broken", "members_broken")}
    frames: list[Frame] = []
    steps = 0
    # deck reference: the bridge points the plate rests on when loading starts, followed
    # as material points (so the deflection does not jump when the contact set changes)
    under = m.q_att
    nd_under = m.a_node[under]
    deck_pts: list = []

    def deck_y() -> float:
        if not len(under):
            return float(st.x[P, 1])
        if not deck_pts:
            sel = np.where(st.q_touch)[0]
            if not len(sel):
                return float(st.x[P, 1] - m.plate_half[1])
            deck_pts.append(sel)
        sel = deck_pts[0]
        sel = sel[~st.loose[nd_under[sel]]] if st.loose[nd_under[sel]].any() else sel
        if not len(sel):
            return float(st.x[P, 1] - m.plate_half[1])
        nd = nd_under[sel]
        p = st.x[nd] + np.einsum("nij,nj->ni", st.R[nd], m.a_r0[under[sel]])
        return float(p[:, 1].mean())

    # nodes that carry elements: when all of a node's elements are broken it is debris
    has_el = np.zeros(m.n_nodes, bool)
    has_el[m.b_i] = True
    has_el[m.b_j] = True
    has_el[m.s_nodes.ravel()] = True
    spring_nodes = np.c_[m.a_node[m.c_a], m.a_node[m.c_b]] if len(m.c_a) else \
        np.zeros((0, 2), int)
    cable_nodes = np.c_[m.a_node[m.k_a], m.a_node[m.k_b]] if len(m.k_a) else \
        np.zeros((0, 2), int)
    # rigid blocks hang on their glue and floss only: with none left they are debris too
    blk = ~has_el
    if m.plate_node >= 0:
        blk[m.plate_node] = False
    tied = np.zeros(m.n_nodes, bool)
    tied[spring_nodes.ravel()] = True
    tied[cable_nodes.ravel()] = True
    blk &= tied

    # developer aid: BRIDGESIM_DUMP_UNSTABLE=path keeps the last state before any node
    # first moved faster than 20 m/s
    dump_path = os.environ.get("BRIDGESIM_DUMP_UNSTABLE")
    snaps: list = [None, False]

    def mark_debris() -> None:
        """Nodes whose elements have all broken become debris: they fall under gravity
        and no longer interact (like eroded nodes in explicit codes)."""
        live = np.zeros(m.n_nodes, bool)
        ok_b = st.b_alive
        live[m.b_i[ok_b]] = True
        live[m.b_j[ok_b]] = True
        live[m.s_nodes[~st.s_gone].ravel()] = True
        newly = has_el & ~live & ~st.loose
        if blk.any():
            con = np.zeros(m.n_nodes, bool)
            con[spring_nodes[st.c_alive].ravel()] = True
            con[cable_nodes[st.k_alive].ravel()] = True
            newly |= blk & ~con & ~st.loose
        if newly.any():
            st.loose |= newly
            if len(spring_nodes):
                st.c_alive &= ~(st.loose[spring_nodes[:, 0]] | st.loose[spring_nodes[:, 1]])

    def do_step(grav: float, alpha: float) -> dict:
        nonlocal steps
        mark_debris()
        info = S.internal_and_contact(m, st, s, F, Mo)
        if s.use_numba and S.K.HAVE_NUMBA:
            S.K.integrate(st.x, st.v, st.w, st.R, F, Mo, mass, phys_mass,
                          rot, -s.gravity * grav, alpha, dt, st.loose, m.table_y + 1.0)
        else:
            fext = np.zeros_like(F)
            fext[:, 1] = -mass * s.gravity * grav * grav_ratio
            acc = (fext - F) / mass[:, None] - alpha * st.v
            st.v += dt * acc
            st.x += dt * st.v
            st.w += dt * (-Mo / rot[:, None] - alpha * st.w)
            if st.loose.any():
                st.w[st.loose] = 0.0
                low = st.loose & (st.x[:, 1] < m.table_y + 1.0)  # debris lands
                st.x[low, 1] = m.table_y + 1.0
                st.v[low, 1] = np.maximum(st.v[low, 1], 0.0)
            S.rotate(st, dt, s.use_numba)
        st.t += dt
        steps += 1
        if steps % 2000 == 0:
            S.reorthonormalize(st)
        if dump_path and steps % 200 == 0:
            vm = float(np.abs(st.v).max())
            if vm < 2e4:
                snaps[0] = copy.deepcopy(st)
            elif not snaps[1]:
                snaps[1] = True
                import pickle
                with open(dump_path, "wb") as fh:
                    pickle.dump({"st": snaps[0], "mass": mass, "rot": rot, "dt": dt,
                                 "s": s, "steps": steps}, fh)
        if steps % 100 == 0:
            vmax = float(np.abs(st.v).max())
            if not np.isfinite(vmax) or vmax > 1e6:  # 1 km/s: integration diverged
                m.mass = phys_mass
                iv = int(np.nanargmax(np.abs(st.v).max(1)))
                bid = int(m.node_body[iv])
                where = m.bodies[bid].name if bid >= 0 else "loading plate"
                diag = _node_diag(m, st, iv)
                raise _Unstable(f"speed {vmax:.3g} mm/s at t = {st.t:.4f} s in {where} "
                                f"[{diag}]")
        return info

    # ------------------------------------------------------------ 1. settle (self weight)
    # gravity is ramped on over settle_s, then the bridge and the plate are left to come
    # to rest (until no node moves faster than 0.5 mm/s, at most settle_max_s)
    y_ref = st.x[:, 1].copy()
    n_ramp = max(1, int(opt.settle_s / dt))
    k = 0
    while True:
        g = min(1.0, (k + 1) / n_ramp)
        do_step(g, max(opt.alpha, 150.0))
        k += 1
        if k > n_ramp and k % 200 == 0:
            vmax_s = float(np.abs(st.v).max())
            # at rest: mass-weighted RMS speed below 0.5 mm/s (a light toggle still
            # swinging a little does not hold up the test)
            vrms = float(np.sqrt((mass * (st.v**2).sum(1)).sum() / mass.sum()))
            if opt.progress and k % 4000 == 0:
                iv = int(np.argmax(np.abs(st.v).max(1)))
                bid = int(m.node_body[iv])
                opt.progress(0.0, f"settling under self-weight: t {st.t:.3f} s, "
                                  f"fastest node {vmax_s:.1f} mm/s ("
                                  f"{m.bodies[bid].name if bid >= 0 else 'plate'})")
            if vrms < 0.5 or st.t > opt.settle_max_s:
                break
    sag = float(np.max(y_ref[:-1] - st.x[:-1, 1]))
    ke = S.kinetic_energy(m, st)
    if sag > 20 or not np.isfinite(sag):
        criteria["instability"] = {"t_s": st.t, "detail": f"sagged {sag:.1f} mm under "
                                                          "its own weight"}
        events.append(Event(st.t, 0.0, 0.0, "instability",
                            "bridge collapses under its own weight", f"sag {sag:.1f} mm"))
    unstable = criteria["instability"] is not None  # failed before any load: F_u = 0
    deck0 = deck_y()
    seat = st.x[P] + st.R[P] @ np.array([0, m.plate_half[1], 0])
    st.jack_y = float(seat[1])  # the jack is brought onto the seat, then starts moving
    jack0 = st.jack_y

    support_mask = m.p_support.copy()

    # ------------------------------------------------------------ 2. load
    prev_b = st.b_fractured.copy()
    prev_bs = st.b_shear.copy()
    prev_s = st.s_dmg > 0.5
    prev_k = st.k_alive.copy()
    # a member crushing: a fibre's plastic strain in compression past its onset strain
    crush_eps = -np.asarray(m.b_fc) / np.maximum(np.asarray(m.b_E), 1e-9) \
        if len(m.b_i) else np.zeros(0)
    prev_c = np.zeros(len(m.b_i), bool)
    patch_of = m.c_patch
    n_patch = len(m.patches)
    prev_patch_broken = np.zeros(n_patch, bool)
    prev_patch_started = np.zeros(n_patch, bool)
    prev_patch_cracked = np.zeros(n_patch, bool)
    # glued area of each joint, to tell a joint that has cracked from one that has let go
    patch_area = np.bincount(patch_of, weights=m.c_area, minlength=n_patch) \
        if len(patch_of) else np.zeros(n_patch)
    peak = 0.0
    peak_travel = 0.0
    peak_defl = 0.0
    fail_at: float | None = None  # travel at which the bridge is judged failed
    fail_mode = ""
    fail_detail = ""
    Fu_at_fail = None
    i_fail: int | None = None  # curve sample at which the bridge was judged failed
    est_steps = int((opt.max_travel_mm / opt.speed_mm_s) / dt)
    frame_every = max(1, int(opt.frame_mm / opt.speed_mm_s / dt))
    ramp = 0.02  # s, the jack starts smoothly
    t_load0 = st.t
    max_ke_ratio = 0.0
    work = 0.0
    last_force = 0.0
    force_f = 0.0
    raw_sum, raw_n = 0.0, 0  # the raw jack force, averaged between curve samples
    glue_hist: list[np.ndarray] = []
    k = 0
    throttle = 1.0
    hold_until = -1.0
    hold_start = 0.0
    holds = 0
    hold_time = 0.0
    n_seen = 0
    ke_q = 0.0  # kinetic energy of the bridge under steady loading (full jack speed)
    recent: list[tuple[float, float]] = []  # (t, load) over the last half settle window
    major = ("joint_cracking", "joint_failure", "plate_crack", "member_fracture",
             "member_shear", "floss_break")
    while not unstable:
        tl = st.t - t_load0
        target = opt.hold_speed_frac if st.t < hold_until else 1.0
        step_thr = dt / opt.hold_ramp_s
        throttle = min(throttle + step_thr, target) if throttle < target else \
            max(throttle - step_thr, target)
        if throttle < 0.999:
            hold_time += dt
        v = opt.speed_mm_s * min(1.0, tl / ramp) * throttle
        st.jack_y -= v * dt
        st.jack_v = -v
        do_step(1.0, opt.alpha if fail_at is None else opt.alpha_collapse)
        k += 1
        travel = jack0 - st.jack_y
        raw = st.jack_force
        work += 0.5 * (raw + last_force) * v * dt
        last_force = raw
        # the load cell reading: low-pass filtered over ~0.3 mm of jack travel, so a
        # stress wave ringing through the bridge is not taken for its strength
        a_f = min(1.0, dt * opt.speed_mm_s / LOAD_CELL_MM)  # 0.3 mm at full speed
        force_f += a_f * (raw - force_f)
        force = force_f
        raw_sum += raw
        raw_n += 1
        if k % 25 == 0 or k == 1:
            raw_mean = raw_sum / raw_n
            raw_sum, raw_n = 0.0, 0
            defl = deck0 - deck_y()
            # kinetic energy of the bridge, without the plate (it moves with the jack)
            # and without debris (broken-off pieces falling away)
            inb = ~st.loose
            if P >= 0:
                inb[P] = False
            kin = 0.5 * float((mass[inb] * (st.v[inb] ** 2).sum(1)).sum())
            curve.setdefault("work", []).append(work)
            curve["t"].append(st.t)
            curve["travel"].append(travel)
            curve["force"].append(force)
            curve.setdefault("force_raw", []).append(raw_mean)
            curve.setdefault("hold", []).append(1.0 if throttle < 0.5 else 0.0)
            curve["deflection"].append(defl)
            curve["kinetic"].append(kin)
            curve["plate_load"].append(0.0)
            pdmg = np.zeros(n_patch)
            pfrac = np.zeros(n_patch)
            if len(patch_of):
                np.maximum.at(pdmg, patch_of, st.c_d)
                pfrac = np.bincount(patch_of, weights=m.c_area * (st.c_d >= 0.999),
                                    minlength=n_patch) / np.maximum(patch_area, 1e-12)
            curve["glue_broken"].append(int((pfrac >= JOINT_FAILED).sum()))
            # each joint's glued area let go so far (damage-weighted), for what governs
            glue_hist.append((np.bincount(patch_of, weights=m.c_area * st.c_d,
                                          minlength=n_patch) if len(patch_of)
                              else np.zeros(n_patch)).astype(np.float32))
            curve["members_broken"].append(int((st.b_fractured | st.b_shear).sum()))
            if force > peak and fail_at is None:  # F_u: the peak before the failure call
                peak, peak_travel, peak_defl = force, travel, defl
            # ---- events
            def ev(kind, what, detail="", where=None, _tr=travel, _f=force, fraction=None):
                events.append(Event(st.t, _tr, _f, kind, what, detail,
                                    None if where is None else tuple(where), fraction))

            def mid(nodes):  # current position of a group of nodes
                return st.x[np.asarray(nodes)].mean(0)

            nb = st.b_fractured & ~prev_b
            for e in np.where(nb)[0]:
                ev("member_fracture", m.bodies[m.b_body[e]].name,
                   "member broke (fibres cracked through the section)",
                   mid([m.b_i[e], m.b_j[e]]))
            ns = st.b_shear & ~prev_bs
            for e in np.where(ns)[0]:
                ev("member_shear", m.bodies[m.b_body[e]].name,
                   "member split in shear along its glue lines or grain: it bends as "
                   "separate parts but still carries axial load",
                   mid([m.b_i[e], m.b_j[e]]))
            sd = (st.s_dmg > 0.5) & ~prev_s
            if sd.any():
                for bid in np.unique(m.s_body[sd]):
                    els = np.where(sd & (m.s_body == bid))[0]
                    ev("plate_crack", m.bodies[bid].name,
                       f"{len(els)} shell element(s) cracked", mid(m.s_nodes[els].ravel()))
            kb = ~st.k_alive & prev_k
            for c in np.where(kb)[0]:
                ev("floss_break", m.bodies[m.k_body[c]].name, "floss snapped",
                   mid([m.a_node[m.k_a[c]], m.a_node[m.k_b[c]]]))
            if len(m.b_i):
                crushed = (st.eps_p.reshape(len(m.b_i), -1).min(1) <= crush_eps) & st.b_alive
                for e in np.where(crushed & ~prev_c)[0]:
                    ev("member_crushing", m.bodies[m.b_body[e]].name,
                       "member crushing in compression (its fibres fold past f_c)",
                       mid([m.b_i[e], m.b_j[e]]))
                prev_c = prev_c | crushed
            started = pdmg > 0.0
            cracked = pfrac >= JOINT_CRACKED
            broken = pfrac >= JOINT_FAILED
            for pid in np.where(started & ~prev_patch_started)[0]:
                pa = m.patches[pid]
                ev("glue_damage", f"{m.bodies[pa.body_a].name} | {m.bodies[pa.body_b].name}",
                   f"glue line starts to fail ({pa.area_mm2:.0f} mm^2)", pa.centroid,
                   fraction=float(pfrac[pid]))
            for pid in np.where(cracked & ~prev_patch_cracked & ~broken)[0]:
                pa = m.patches[pid]
                ev("joint_cracking", f"{m.bodies[pa.body_a].name} | {m.bodies[pa.body_b].name}",
                   f"{pfrac[pid]:.0%} of the glued joint has let go ({pa.area_mm2:.0f} mm^2)",
                   pa.centroid, fraction=float(pfrac[pid]))
            for pid in np.where(broken & ~prev_patch_broken)[0]:
                pa = m.patches[pid]
                ev("joint_failure", f"{m.bodies[pa.body_a].name} | {m.bodies[pa.body_b].name}",
                   f"glued joint separated ({pfrac[pid]:.0%} of its {pa.area_mm2:.0f} mm^2 "
                   "let go)", pa.centroid, fraction=float(pfrac[pid]))
            prev_b, prev_bs, prev_s = st.b_fractured.copy(), st.b_shear.copy(), st.s_dmg > 0.5
            prev_k = st.k_alive.copy()
            prev_patch_started, prev_patch_broken = started, broken
            prev_patch_cracked = cracked | broken
            # ---- hold the jack while something is breaking (quasi-static failure)
            if opt.hold_on_failure and fail_at is None and len(events) > n_seen:
                if any(e.kind in major for e in events[n_seen:]):
                    if st.t >= hold_until:
                        holds += 1
                        hold_start = st.t
                    hold_until = min(st.t + opt.hold_settle_s, hold_start + opt.hold_max_s)
                n_seen = len(events)
            # ... and until the bridge has settled: no hold ends while it still rings
            recent.append((st.t, force))  # the online reading (the raw force rings)
            while recent and recent[0][0] < st.t - 0.5 * opt.hold_settle_s:
                recent.pop(0)
            if throttle >= 0.999 and st.t >= hold_until and tl > ramp:
                ke_q = kin if ke_q <= 0 else 0.95 * ke_q + 0.05 * kin
            if opt.hold_on_failure and fail_at is None and hold_until > 0 and \
                    hold_until - 0.0005 <= st.t < hold_start + opt.hold_max_s:
                fr = [f for _, f in recent]
                ringing = kin > opt.hold_ke_settle * max(ke_q, 1e-9) or \
                    max(fr) - min(fr) > opt.hold_force_tol * max(abs(force), 1.0)
                if ringing:
                    hold_until = min(st.t + 0.001, hold_start + opt.hold_max_s)
            # ---- §12.5 criteria
            if criteria["deflection"] is None and defl > opt.deflection_limit_mm:
                criteria["deflection"] = {"t_s": st.t, "travel_mm": travel, "load_N": force}
                ev("criterion", "deflection", f"mid-span deflection {defl:.1f} mm > "
                                              f"{opt.deflection_limit_mm:g} mm")
            if criteria["contact"] is None:
                pts = m.a_node[m.p_att]
                touch = st.p_touch & ~support_mask
                if touch.any():
                    bid = int(m.node_body[pts[np.where(touch)[0][0]]])
                    nm = m.bodies[bid].name if bid >= 0 else "?"
                    criteria["contact"] = {"t_s": st.t, "travel_mm": travel, "load_N": force,
                                           "part": nm}
                    ev("criterion", "contact", f"'{nm}' touches the base platform")
                # the bridge itself, not debris flying off it (loose fragments)
                xs = st.x[:-1, 0][~st.loose[:-1]]
                if (xs < m.frame_x[0]).any() or (xs > m.frame_x[1]).any():
                    criteria["contact"] = {"t_s": st.t, "travel_mm": travel,
                                           "load_N": force, "part": "press frame"}
                    ev("criterion", "contact", "the bridge touches the press uprights")
            if criteria["joint_or_shear"] is None:
                hit = [e for e in events if e.kind in ("joint_failure", "member_shear")]
                if hit:
                    criteria["joint_or_shear"] = {"t_s": hit[0].t, "load_N": hit[0].force_N,
                                                  "travel_mm": hit[0].travel_mm,
                                                  "what": hit[0].what}
            # judged failed: a clear load drop after the peak, or a criterion
            if fail_at is None:
                dropped = peak > 50 and force < opt.drop_ratio * peak and travel > peak_travel
                if dropped:
                    fail_at, fail_mode = travel, "collapse"
                    fail_detail = (f"load fell from {peak / KGF:.1f} to {force / KGF:.1f} "
                                   "kgf")
                    Fu_at_fail = peak
                    ev("criterion", "collapse", fail_detail)
                elif criteria["deflection"] is not None:
                    fail_at, fail_mode = travel, "deflection"
                    fail_detail = "mid-span deflection passed 50 mm"
                    Fu_at_fail = peak
                elif criteria["contact"] is not None:
                    fail_at, fail_mode = travel, "contact"
                    fail_detail = criteria["contact"].get("part", "")
                    Fu_at_fail = peak
                if fail_at is not None:
                    i_fail = len(curve["t"]) - 1
            if opt.progress and k % 2000 == 0:
                opt.progress(min(1.0, travel / opt.max_travel_mm),
                             f"jack {travel:.1f} mm, load {force / KGF:.1f} kgf, "
                             f"dt {dt * 1e6:.2f} us, vmax {np.abs(st.v).max():.0f} mm/s, "
                             f"KE/W {kin / max(work, 1e-9):.3f}")
        if k % frame_every == 0 or k == 1:
            frames.append(_frame(m, st, travel, force, deck0 - deck_y()))
        if not np.isfinite(st.x).all():
            events.append(Event(st.t, travel, force, "numerical",
                                "simulation became unstable", "time step too large"))
            break
        if fail_at is not None and travel > fail_at + opt.after_failure_mm:
            break
        if travel > opt.max_travel_mm or k > min(opt.max_steps, 3 * est_steps):
            break
    frames.append(_frame(m, st, jack0 - st.jack_y, st.jack_force, deck0 - deck_y()))
    if unstable:
        fail_at, fail_mode, Fu_at_fail = 0.0, "instability", 0.0
        fail_detail = criteria["instability"]["detail"]
    if fail_at is None:
        fail_mode = "none"
        fail_detail = (f"no failure within {opt.max_travel_mm:g} mm of jack travel")
        Fu_at_fail = peak
    # ---- the reported load: the raw force smoothed without lag (reported_load). The
    # online reading above only decided when to hold the jack and when the bridge was
    # judged failed; F_u, the curve and the load of each event come from this one.
    t_c = np.inf  # where the load last stood within 5 % of F_u before the failure call
    if curve["t"] and not unstable:
        tt = np.asarray(curve["t"])
        fz = reported_load(tt, np.asarray(curve["force_raw"]), opt.speed_mm_s,
                           np.asarray(curve["travel"]))
        curve["force_online"] = curve["force"]
        curve["force"] = [float(f) for f in fz]
        for e in events:
            e.force_N = float(np.interp(e.t, tt, fz))
        for c in criteria.values():
            if c and "t_s" in c and "load_N" in c and c["t_s"] >= tt[0]:
                c["load_N"] = float(np.interp(c["t_s"], tt, fz))
        n_ok = len(tt) if i_fail is None else i_fail + 1
        i_pk = int(np.argmax(fz[:n_ok]))
        peak, peak_travel = float(fz[i_pk]), float(curve["travel"][i_pk])
        peak_defl = float(curve["deflection"][i_pk])
        Fu_at_fail = peak
        high = np.where(fz[i_pk:n_ok] >= 0.95 * peak)[0]
        t_c = float(tt[i_pk + high[-1]]) if i_fail is not None else np.inf
        if fail_mode == "collapse":
            fail_detail = (f"load fell more than {1 - opt.drop_ratio:.0%} below its peak of "
                           f"{peak / KGF:.1f} kgf")
            for e in events:
                if e.kind == "criterion" and e.what == "collapse":
                    e.detail = fail_detail
    # what happened up to the failure call (the online reading lags the fall of the
    # load, so what broke before the call caused it; what breaks after is the collapse)
    t_call = float(curve["t"][i_fail]) if (i_fail is not None and curve["t"]) else np.inf
    before = [e for e in events if e.t <= t_call]
    first_major, competing = attribute(before, t_c, peak, peak_travel, opt.speed_mm_s)
    # no part broke before the call: the joint whose glue let go during the fall (it is
    # the cause even before 30 % of it has let go; one that lets go after the call is
    # the collapse)
    glue_fall = None
    if first_major is None and glue_hist and np.isfinite(t_c) and len(patch_area):
        tt = np.asarray(curve["t"])
        lag = LOAD_CELL_MM / max(opt.speed_mm_s, 1e-9)
        i0 = max(0, int(np.searchsorted(tt, t_c - lag)))
        i1 = min(len(glue_hist) - 1, int(np.searchsorted(tt, t_call + lag)))
        lost = (glue_hist[i1] - glue_hist[i0]) / np.maximum(patch_area, 1e-12)
        k = int(np.argmax(lost)) if len(lost) else -1
        if k >= 0 and lost[k] >= 0.2:
            pa = m.patches[k]
            glue_fall = (f"{m.bodies[pa.body_a].name} | {m.bodies[pa.body_b].name}",
                         float(lost[k]))
    governing = fail_mode
    if fail_mode == "collapse" and first_major:
        governing = f"{first_major.kind.replace('_', ' ')}: {first_major.what}"
        fail_detail = f"{governing}. " + fail_detail
    elif fail_mode == "collapse" and glue_fall:
        governing = f"joint failure: {glue_fall[0]}"
        fail_detail = (f"{governing}: {glue_fall[1]:.0%} of its glue let go as the load "
                       "fell. " + fail_detail)
    elif fail_mode == "collapse":
        soft = list(dict.fromkeys(e.what for e in before if e.kind == "glue_damage"
                                  and e.force_N >= 0.9 * peak))
        if soft:
            governing = f"glue giving way: {soft[0]}"
            fail_detail = ("no part broke before the load fell, but the glue of "
                           + "; ".join(soft[:3]) + (f" (and {len(soft) - 3} more)"
                                                    if len(soft) > 3 else "")
                           + " was giving way at the peak: " + fail_detail)
        else:
            governing = "buckling or sway"
            fail_detail = ("no part broke: a buckling or sway mechanism (see the replay), "
                           "the " + fail_detail)
    m.mass = phys_mass
    # quasi-static check: kinetic energy vs the work done by the jack, once that work is
    # at least a tenth of the work done up to the peak load (before that both are ~0)
    W = np.array(curve.get("work", [0.0]))
    KE = np.array(curve["kinetic"]) if curve["kinetic"] else np.zeros(1)
    Fc = np.array(curve["force"]) if curve["force"] else np.zeros(1)
    i_pk = int(np.argmax(Fc))
    # judged on the loading up to 95 % of the peak load: whether inertia props up the
    # rising branch. The collapse that follows the peak is dynamic, as in the real test.
    i_95 = int(np.argmax(Fc >= 0.95 * Fc[i_pk])) if len(Fc) else 0
    w_ref = 0.1 * (W[i_pk] if len(W) > i_pk else 0.0)
    sel = (W > max(w_ref, 1.0)) & (np.arange(len(W)) <= i_95)
    max_ke_ratio = float((KE[sel] / W[sel]).max()) if sel.any() else 0.0
    res = CrushResult(
        Fu_N=float(Fu_at_fail), Fu_kgf=float(Fu_at_fail) / KGF, failure_mode=fail_mode,
        failure_detail=fail_detail, governing=governing, competing=competing,
        t_failure=float(fail_at or 0.0),
        deflection_at_Fu_mm=float(peak_defl), curve=curve, events=events, frames=frames,
        criteria=criteria, dt=dt, steps=steps, wall_s=time.time() - t0,
        added_mass_kg=added * 1000, added_mass_frac=added / phys_mass.sum(),
        settle_sag_mm=sag, kinetic_ratio_max=max_ke_ratio, model=m,
        speed_mm_s=float(opt.speed_mm_s), holds=holds, hold_time_s=hold_time)
    del ke
    return res


def _frame(m: Model, st: S.State, travel: float, force: float, defl: float) -> Frame:
    bstate = np.zeros(len(m.b_i), np.uint8)
    bstate[st.b_fractured] = 1
    bstate[st.b_shear | ~st.b_alive] = 2
    sstate = (st.s_dmg > 0.5).astype(np.uint8)
    sstate[st.s_gone] = 2
    pd = np.zeros(len(m.patches))
    if len(m.c_patch):  # share of each joint's glued area that has let go
        area = np.bincount(m.c_patch, weights=m.c_area, minlength=len(m.patches))
        pd = np.bincount(m.c_patch, weights=m.c_area * (st.c_d >= 0.999),
                         minlength=len(m.patches)) / np.maximum(area, 1e-12)
    return Frame(st.t, travel, force, defl, st.x.astype(np.float32),
                 _quat(st.R).astype(np.float32), bstate, sstate, pd.astype(np.float32),
                 st.k_alive.copy())


__all__ = ["run_crush", "CrushOptions", "CrushResult", "Event", "Frame", "choose_dt",
           "KGF", "math"]


def max_frequency(m: Model, s: S.Settings, mass: np.ndarray, rot: np.ndarray,
                  iters: int = 120, seed: int = 0, k: int = 1
                  ) -> tuple[float, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """Largest eigenvalue omega^2 of M^-1 K of the whole model (power iteration with the
    tangent stiffness from finite differences of the internal forces) and the energy of
    the corresponding mode at each node. Contacts with the platform, the plate and the
    jack are switched on for the estimate, since they all act during the test."""
    st0 = S.initial_state(m, jack_y=0.0)
    P = m.plate_node
    st0.jack_y = float(m.X[P, 1] + m.plate_half[1]) - 1e-3 if P >= 0 else 1e12
    saved = (m.table_y, m.plate_half.copy())
    m.table_y = m.table_y + 1e-3
    m.plate_half = m.plate_half + np.array([0, 0.05, 0])
    n = m.n_nodes
    F0 = np.zeros((n, 3))
    M0 = np.zeros((n, 3))
    S.internal_and_contact(m, st0, s, F0, M0, update=False)
    # (st0 has zero velocity, so element damping does not enter)
    sq = np.sqrt(np.c_[np.repeat(mass[:, None], 3, 1), np.repeat(rot[:, None], 3, 1)])
    F1 = np.zeros((n, 3))
    M1 = np.zeros((n, 3))

    def matvec(vflat):
        v = vflat.reshape(n, 6)
        u = v / sq
        amp = max(np.abs(u).max(), 1e-30)
        eps = 1e-6 / amp
        st = S.State(**{k: (val.copy() if isinstance(val, np.ndarray) else val)
                        for k, val in st0.__dict__.items()})
        st.x = st0.x + eps * u[:, :3]
        st.R = S.expmap(eps * u[:, 3:]) @ st0.R
        S.internal_and_contact(m, st, s, F1, M1, update=False)
        return (np.c_[F1 - F0, M1 - M0] / eps / sq).ravel()

    from scipy.sparse.linalg import LinearOperator, eigsh
    op = LinearOperator((6 * n, 6 * n), matvec=matvec, dtype=float)
    rng = np.random.default_rng(seed)
    try:
        vals, vecs = eigsh(op, k=k, which="LA", tol=1e-3, maxiter=iters * 4,
                           v0=rng.normal(size=6 * n), ncv=min(max(40, 3 * k + 10), 6 * n - 1))
        if k > 1:
            m.table_y, m.plate_half = saved
            V = vecs.reshape(n, 6, k)
            en = np.stack([(V[:, :3] ** 2).sum(1), (V[:, 3:] ** 2).sum(1)], axis=1)
            return vals, en  # (k,), (n, 2, k)
        lam = float(vals[0])
        v = vecs[:, 0].reshape(n, 6)
    except Exception:  # noqa: BLE001 - fall back to plain power iteration
        v = rng.normal(size=(n, 6))
        lam = 0.0
        for _ in range(iters * 3):
            v /= np.linalg.norm(v)
            w = matvec(v.ravel()).reshape(n, 6)
            lam = max(lam, float((v * w).sum()))
            v = w
    finally:
        m.table_y, m.plate_half = saved
    v /= np.linalg.norm(v)
    energy = np.c_[(v[:, :3] ** 2).sum(1), (v[:, 3:] ** 2).sum(1)]
    return max(lam, 0.0), energy


def stable_dt(m: Model, s: S.Settings, dt_target: float = 5e-6, max_added: float = 0.08,
              safety: float = 0.85, rounds: int = 30) -> tuple[float, np.ndarray, np.ndarray,
                                                                list[str]]:
    """Time step and selectively scaled masses so that the explicit scheme is stable.

    Start from the element-based estimate for rotational inertia (scaled freely), then
    repeatedly find the model's true highest mode (power iteration) and add mass only at
    the nodes that carry it, until dt_target is stable or the added mass reaches
    ``max_added``; then dt is set to what is stable."""
    log = []
    mass = m.mass.copy()
    rot = m.rot_inertia.copy()
    rot_cap = 50.0 * m.rot_inertia  # rotations must not become sluggish
    lam_t = (2 * safety / dt_target) ** 2
    total = m.mass.sum()
    # first pass: rotational inertia from the element estimates, capped
    Kb = _beam_local_K(m) if len(m.b_i) else None
    for _ in range(8):
        lam_node, worst = _node_lambda(m, s, mass, rot, Kb, lam_t)
        if worst <= 1.0:
            break
        rot = np.minimum(rot * np.clip(lam_node / lam_t * 1.02, 1.0, 100.0), rot_cap)
    lam = 0.0
    for r in range(rounds):
        vals, en = max_frequency(m, s, mass, rot, k=8)
        lam = float(vals.max())
        if lam <= lam_t:
            break
        added = (mass - m.mass).sum() / total
        if added >= max_added:
            break
        f = np.ones(len(mass))
        rot_only = np.zeros(len(mass), bool)
        n_hot = 0
        for j in np.where(vals > lam_t)[0]:
            e = en[:, 0, j] + en[:, 1, j]
            share = e / e.max()
            hot = share > 0.02
            n_hot += int(hot.sum())
            fj = 1 + (vals[j] / lam_t - 1) * share + 0.05
            f = np.where(hot, np.maximum(f, np.minimum(fj, 4.0)), f)
            rot_only |= hot & (en[:, 1, j] > en[:, 0, j])
        rot_only &= rot * f < rot_cap
        hot_any = f > 1.0
        rot[hot_any] *= f[hot_any]
        mass[hot_any & ~rot_only] *= f[hot_any & ~rot_only]
        log.append(f"round {r}: omega_max {np.sqrt(lam):.0f} rad/s, scaled {hot_any.sum()} "
                   f"nodes ({(hot_any & rot_only).sum()} rotation only), "
                   f"added mass {added:.2%}")
    lam = float(max_frequency(m, s, mass, rot)[0]) if log else lam
    dt = min(dt_target, 2 * safety / np.sqrt(max(lam, 1e-30)))
    return dt, mass, rot, log
