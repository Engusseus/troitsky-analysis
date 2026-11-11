"""Virtual laboratory: the physical tests a team can do, simulated with the same solver.

Each test builds a small model with the elements the bridge simulation uses (fibre beams,
glue springs, shells) and drives it quasi-statically with displacement control, like a
testing machine:

* :func:`three_point_bending` - a single stick on two supports, loaded at mid-span (E and
  the modulus of rupture);
* :func:`lap_shear` - two sticks glued with a single overlap, pulled apart (the glue's
  average shear strength tau_g);
* :func:`euler_column` - a pin-ended column with a small initial bow (buckling);
* :func:`cantilever` - tip-loaded cantilever, small or large deflection (elastica).

They serve three purposes: verification against textbook answers (tests/), calibration of
the simulation to a team's own test data (:mod:`bridgesim.sim.calibrate`), and a direct
comparison "simulated test vs your test" that measures how close the model is to reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bridgesim.sim import shell as shellmod
from bridgesim.sim import solver as S
from bridgesim.sim.materials import fibre_spread as _exact
from bridgesim.sim.materials import rect_torsion
from bridgesim.sim.model import Model


def blank_model(X: np.ndarray, nf: int = 15) -> Model:
    n = len(X)
    z = np.zeros
    return Model(
        X=np.asarray(X, float), mass=z(n), rot_inertia=z(n), node_body=z(n, int),
        b_i=z(0, int), b_j=z(0, int), b_L0=z(0), b_E0=z((0, 3, 3)), b_fy=z((0, nf)),
        b_fz=z((0, nf)), b_fA=z((0, nf)), b_E=z(0), b_ft=z(0), b_fc=z(0), b_Gf=z(0),
        b_GJ=z(0), b_A=z(0), b_fv=z(0), b_fv3=z(0), b_nply=z(0), b_Wt=z((0, 2)), b_kGA=z(0),
        b_body=z(0, int),
        s_nodes=z((0, 3), int), s_E0=z((0, 3, 3)), s_P=z((0, 3, 3)), s_t=z(0),
        s_body=z(0, int), **shellmod.empty(), a_node=z(0, int), a_r0=z((0, 3)),
        c_a=z(0, int), c_b=z(0, int), c_area=z(0), c_n0=z((0, 3)), c_Kn=z(0), c_Ks=z(0),
        c_sig=z(0), c_tau=z(0), c_GIc=z(0), c_GIIc=z(0), c_mu=z(0), c_patch=z(0, int),
        c_breakable=z(0, bool), k_a=z(0, int), k_b=z(0, int), k_L0=z(0), k_EA=z(0),
        k_Fb=z(0), k_body=z(0, int), p_att=z(0, int), p_support=z(0, bool),
        q_att=z(0, int), plate_node=-1, plate_half=np.ones(3), jack_xz=z(2),
        jack_radius=0.0, frame_x=(-1e9, 1e9), bodies=[], patches=[], table_y=-1e9,
        params={"crush_softening": 0.3, "crush_band_mm": CRUSH_BAND_MM})


#: the crushing fall's gauge length of a coupon model (wood.crush_band_mm's shipped value)
CRUSH_BAND_MM = 12.5


@dataclass
class BeamSpec:
    b: float  # width along local e2 (mm)
    d: float  # depth along local e3 (mm)
    E: float
    ft: float = 1e9
    fc: float = 1e9
    Gf: float = 1e3
    G: float = 600.0
    fv: float = 1e9
    rho: float = 6.5e-10  # t/mm^3
    ny: int = 3
    nz: int = 5
    shear: bool = True  # Timoshenko shear deformation (k = 5/6); False: Euler-Bernoulli


def add_beam_line(m: Model, nodes: list[int], spec: BeamSpec, up=(0.0, 0.0, 1.0)) -> None:
    """Fibre beam elements along consecutive ``nodes``; local e3 is ``up`` (the depth d
    direction), e2 = e3 x e1."""
    up = np.asarray(up, float)
    gy = ((np.arange(spec.ny) + 0.5) / spec.ny - 0.5) * spec.b * _exact(spec.ny)
    gz = ((np.arange(spec.nz) + 0.5) / spec.nz - 0.5) * spec.d * _exact(spec.nz)
    FY, FZ = np.meshgrid(gy, gz, indexing="ij")
    A = spec.b * spec.d
    fA = np.full(spec.ny * spec.nz, A / (spec.ny * spec.nz))
    J, Wt2, Wt3 = rect_torsion(spec.b, spec.d)
    rows = {k: [] for k in ("i", "j", "L0", "E0")}
    for a, b in zip(nodes[:-1], nodes[1:], strict=True):
        d = m.X[b] - m.X[a]
        L = float(np.linalg.norm(d))
        e1 = d / L
        e2 = np.cross(up, e1)
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        rows["i"].append(a)
        rows["j"].append(b)
        rows["L0"].append(L)
        rows["E0"].append(np.stack([e1, e2, e3], 1))
        me = spec.rho * A * L
        for nn in (a, b):
            m.mass[nn] += me / 2
            m.rot_inertia[nn] += me / 2 * (L * L / 12 + (spec.b**2 + spec.d**2) / 12)
    k = len(rows["i"])

    def cat(name, val):
        setattr(m, name, np.concatenate([getattr(m, name), val]))

    cat("b_i", np.array(rows["i"], int))
    cat("b_j", np.array(rows["j"], int))
    cat("b_L0", np.array(rows["L0"]))
    cat("b_E0", np.array(rows["E0"]))
    cat("b_fy", np.repeat(FY.ravel()[None], k, 0))
    cat("b_fz", np.repeat(FZ.ravel()[None], k, 0))
    cat("b_fA", np.repeat(fA[None], k, 0))
    for name, v in (("b_E", spec.E), ("b_ft", spec.ft), ("b_fc", spec.fc),
                    ("b_Gf", spec.Gf), ("b_GJ", spec.G * J), ("b_A", A),
                    ("b_fv", spec.fv), ("b_fv3", spec.fv), ("b_nply", 2.0),
                    ("b_kGA", 5 / 6 * spec.G * A if spec.shear else 0.0)):
        cat(name, np.full(k, float(v)))
    cat("b_Wt", np.tile([Wt2, Wt3], (k, 1)))
    cat("b_body", np.zeros(k, int))


@dataclass
class TestResult:
    disp: np.ndarray
    force: np.ndarray
    peak: float
    state: S.State
    slow_disp: float = 0.0  # the test ran slowly (quasi-statically) up to this displacement

    @property
    def stiffness(self) -> float:
        """Initial slope (N/mm): from the slow first part of the test when there is one,
        else from the first 25 % of the peak load."""
        if self.slow_disp > 0:
            sel = (self.disp > 0.1 * self.slow_disp) & (self.disp <= self.slow_disp) & \
                (self.force < 0.5 * self.peak)
            if sel.sum() >= 3:
                return float(np.polyfit(self.disp[sel], self.force[sel], 1)[0])
        sel = (self.force > 0.02 * self.peak) & (self.force < 0.25 * self.peak)
        if sel.sum() < 3:
            sel = self.force < 0.3 * self.peak
        return float(np.polyfit(self.disp[sel], self.force[sel], 1)[0])


def drive(m: Model, fixed: dict[int, str], node, dof: int, rate: float,
          max_disp: float, dt: float | None = None, alpha: float = 200.0,
          stop_drop: float = 0.3, record_every: int = 20, max_steps: int = 2_000_000,
          gravity: bool = False, sign: float = 1.0, slow_disp: float = 0.0,
          slow_rate: float | None = None, test_time_s: float = 0.0) -> TestResult:
    """Move ``node`` (one node or a list: a grip) along global ``dof`` at ``rate`` mm/s
    (displacement control) with the ``fixed`` nodes held ("xyz", "xyzr" = also rotations,
    any subset of x, y, z, r) and record the reaction force. Stops at ``max_disp`` or
    when the force drops below ``stop_drop`` x peak after the peak. Up to ``slow_disp``
    the node moves at ``slow_rate`` instead (a quasi-static start, for the stiffness).
    ``test_time_s``: the physical time to failure the glue's strength is taken at
    (solver.glue_rate; 0: the coupon's own, tau_g)."""
    node = np.atleast_1d(np.asarray(node, int))
    m.mass = np.maximum(m.mass, 1e-12)
    m.rot_inertia = np.maximum(m.rot_inertia, 1e-12)
    s = S.Settings(dt=dt or 1e-6)
    if dt is None:
        from bridgesim.sim.crush import stable_dt
        dt, mass, rot, _ = stable_dt(m, s, dt_target=5e-6, max_added=0.0)
        m.mass, m.rot_inertia = mass, rot
        s.dt = dt
    st = S.initial_state(m, jack_y=1e12)
    st.t_test = float(test_time_s)
    st.c_t0 = np.zeros(len(m.c_a))  # every line loaded from the coupon's start
    F = np.zeros((m.n_nodes, 3))
    Mo = np.zeros((m.n_nodes, 3))
    fix_t = np.zeros((m.n_nodes, 3), bool)
    fix_r = np.zeros((m.n_nodes, 3), bool)
    for nd, spec in fixed.items():
        for k, c in enumerate("xyz"):
            if c in spec:
                fix_t[nd, k] = True
        if "r" in spec:
            fix_r[nd] = True
    x0 = m.X[node, dof]
    disp, force = [], []
    peak = 0.0
    k = 0
    u = 0.0
    vr = rate
    while u < max_disp and k < max_steps:
        vr = rate if u >= slow_disp or not slow_rate else slow_rate
        u = min(max_disp, u + vr * s.dt)
        st.x[node, dof] = x0 + sign * u
        st.v[node, dof] = sign * vr
        S.internal_and_contact(m, st, s, F, Mo)
        fext = np.zeros_like(F)
        if gravity:
            fext[:, 1] = -m.mass * s.gravity
        acc = (fext - F) / m.mass[:, None] - alpha * st.v
        acc[fix_t] = 0.0
        acc[node, dof] = 0.0
        st.v += s.dt * acc
        st.v[fix_t] = 0.0
        st.v[node, dof] = sign * vr
        st.x += s.dt * st.v
        st.x[fix_t] = m.X[fix_t]
        wacc = -Mo / m.rot_inertia[:, None] - alpha * st.w
        wacc[fix_r] = 0.0
        st.w += s.dt * wacc
        st.w[fix_r] = 0.0
        S.rotate(st, s.dt)
        st.t += s.dt
        if k % record_every == 0:
            f = sign * float(F[node, dof].sum())
            disp.append(u)
            force.append(f)
            peak = max(peak, f)
            if peak > 0 and f < stop_drop * peak and u > 0.05 * max_disp:
                break
        k += 1
    return TestResult(np.array(disp), np.array(force), peak, st,
                      slow_disp if slow_rate else 0.0)


def three_point_bending(E: float, ft: float, fc: float, span: float = 100.0,
                        b: float = 10.0, t: float = 2.0, Gf: float = 3.0, n_el: int = 16,
                        rate: float = 50.0, fc_prop: float = 1.0,
                        crush_softening: float = 0.3,
                        crush_band_mm: float = CRUSH_BAND_MM) -> TestResult:
    """A stick (b x t) simply supported over ``span``, loaded at mid-span (vertical Y).
    ``fc_prop``: the proportional limit in compression (wood.fc_proportional; 1 =
    elastic-perfectly plastic); ``crush_band_mm`` (wood.crush_band_mm) the gauge length
    of the crushing fall."""
    X = np.c_[np.linspace(0, span, n_el + 1), np.zeros(n_el + 1), np.zeros(n_el + 1)]
    m = blank_model(X)
    m.params.update(fc_proportional=fc_prop, crush_softening=crush_softening,
                    crush_band_mm=crush_band_mm)
    add_beam_line(m, list(range(n_el + 1)), BeamSpec(b=b, d=t, E=E, ft=ft, fc=fc, Gf=Gf),
                  up=(0, 1, 0))
    mid = n_el // 2
    return drive(m, {0: "xyz", n_el: "yz"}, mid, 1, rate, max_disp=span / 4, sign=-1.0,
                 slow_disp=0.15, slow_rate=5.0, stop_drop=0.8)


def euler_column(E: float = 10000.0, L: float = 115.0, b: float = 10.0, t: float = 2.0,
                 bow: float = None, n_el: int = 10, rate: float = 5.0) -> TestResult:
    """Pin-ended column along X with a half-sine bow of ``bow`` mm (default L/1000),
    shortened at one end."""
    bow = L / 1000 if bow is None else bow
    s = np.linspace(0, L, n_el + 1)
    X = np.c_[s, bow * np.sin(np.pi * s / L), np.zeros_like(s)]
    m = blank_model(X)
    add_beam_line(m, list(range(n_el + 1)), BeamSpec(b=b, d=t, E=E), up=(0, 1, 0))
    return drive(m, {0: "xyz", n_el: "yz"}, n_el, 0, rate, max_disp=L * 0.01, sign=-1.0,
                 stop_drop=0.0)


def cantilever(P_over: float = 1.0, E: float = 10000.0, L: float = 100.0, b: float = 10.0,
               t: float = 2.0, n_el: int = 20) -> tuple[float, float, S.State]:
    """Tip load P = P_over * E I / L^2 (vertical), returned as (tip v/L, tip u/L)."""
    X = np.c_[np.linspace(0, L, n_el + 1), np.zeros(n_el + 1), np.zeros(n_el + 1)]
    m = blank_model(X)
    add_beam_line(m, list(range(n_el + 1)), BeamSpec(b=b, d=t, E=E), up=(0, 1, 0))
    I = b * t**3 / 12
    P = P_over * E * I / L**2
    m.mass = np.maximum(m.mass, 1e-12)
    m.rot_inertia = np.maximum(m.rot_inertia, 1e-12)
    from bridgesim.sim.crush import stable_dt
    s = S.Settings(dt=1e-6)
    dt, m.mass, m.rot_inertia, _ = stable_dt(m, s, dt_target=5e-6, max_added=0.0)
    s.dt = dt
    st = S.initial_state(m, jack_y=1e12)
    F = np.zeros((m.n_nodes, 3))
    Mo = np.zeros_like(F)
    # dynamic relaxation: damp, ramp the load, then run until the structure is at rest
    w1 = 3.516 * math.sqrt(E * I / (float(m.mass.sum()) / L * L**4))
    alpha = 2 * w1
    n_ramp = int(4 * 2 * math.pi / w1 / dt)
    k = 0
    while True:
        S.internal_and_contact(m, st, s, F, Mo)
        fext = np.zeros_like(F)
        fext[-1, 1] = -P * min(1.0, k / n_ramp)
        acc = (fext - F) / m.mass[:, None] - alpha * st.v
        acc[0] = 0
        st.v += dt * acc
        st.v[0] = 0
        st.x += dt * st.v
        st.x[0] = m.X[0]
        wacc = -Mo / m.rot_inertia[:, None] - alpha * st.w
        wacc[0] = 0
        st.w += dt * wacc
        st.w[0] = 0
        S.rotate(st, dt)
        k += 1
        if k > n_ramp and k % 200 == 0:
            resid = np.abs(fext - F)[1:].max()
            if resid < 1e-4 * P or k > 200 * n_ramp:
                break
    tip = st.x[-1] - m.X[-1]
    return float(-tip[1] / L), float(-tip[0] / L), st


def lap_shear(tau_max: float, peel_to_shear: float = 0.5, Kn: float = 150.0,
              Ks: float = 60.0, GIc: float = 0.25, GIIc: float = 0.6,
              overlap: float = 20.0, width: float = 10.0, t: float = 2.0,
              E: float = 10000.0, free: float = 40.0, rate: float = 20.0,
              n_ov: int = 8, n_glue: int = 9, one_node: bool = False,
              onset: float = 1.0, peak_slip: float = 1.0, rate_exponent: float = 0.0,
              test_time_s: float = 0.0, coupon_time_s: float = 60.0) -> TestResult:
    """Single-lap joint of two sticks (overlap x width), pulled along X. Returns the
    pull force; tau_avg = peak / (overlap * width). The overlap has ``n_ov`` elements per
    stick and ``n_glue`` rows of glue points, each hung on the nearest node of each stick
    by a rigid arm (as in a bridge model, where a joint's glue points hang on fewer
    nodes than there are points). ``onset``, ``peak_slip``: the glue law's pre-peak
    shape (glue.prepeak_onset, glue.peak_slip_ratio; 1, 1 bilinear). ``test_time_s``:
    a lap failing at that time instead of the coupons' ``coupon_time_s`` (with
    ``rate_exponent``, glue.rate_exponent)."""
    n_free = 8
    # stick A from x=0 to free+overlap at y = 0; stick B overlaps, at y = t
    xa = np.r_[np.linspace(0, free, n_free + 1)[:-1], np.linspace(free, free + overlap,
                                                                    n_ov + 1)]
    xb = np.r_[np.linspace(free, free + overlap, n_ov + 1),
               np.linspace(free + overlap, 2 * free + overlap, n_free + 1)[1:]]
    XA = np.c_[xa, np.zeros_like(xa), np.zeros_like(xa)]
    XB = np.c_[xb, np.full_like(xb, t), np.zeros_like(xb)]
    m = blank_model(np.vstack([XA, XB]))
    na = list(range(len(xa)))
    nb = list(range(len(xa), len(xa) + len(xb)))
    spec = BeamSpec(b=width, d=t, E=E)
    add_beam_line(m, na, spec, up=(0, 1, 0))
    add_beam_line(m, nb, spec, up=(0, 1, 0))
    # glue springs: on the interface y = t/2, n_glue rows along the overlap, 2 across
    a_node, a_r0, ca, cb, area = [], [], [], [], []
    ov_a = np.array(na[n_free:])
    ov_b = np.array(nb[: n_ov + 1])
    xg = np.linspace(free, free + overlap, n_glue)
    trib = np.full(n_glue, overlap / (n_glue - 1))
    trib[[0, -1]] *= 0.5
    for k, xq in enumerate(xg):
        xn = free + overlap / 2 if one_node else xq  # one_node: all on the middle node
        ia = int(ov_a[np.argmin(abs(m.X[ov_a, 0] - xn))])
        ib = int(ov_b[np.argmin(abs(m.X[ov_b, 0] - xn))])
        for zq in (-width / 4, width / 4):
            p = np.array([xq, t / 2, zq])
            a_node += [ia, ib]
            a_r0 += [p - m.X[ia], p - m.X[ib]]
            ca.append(len(a_node) - 2)
            cb.append(len(a_node) - 1)
            area.append(trib[k] * width / 2)
    nc = len(ca)
    m.a_node = np.array(a_node)
    m.a_r0 = np.array(a_r0)
    m.c_a, m.c_b = np.array(ca), np.array(cb)
    m.c_area = np.array(area)
    m.c_n0 = np.repeat(np.array([[0.0, 1.0, 0.0]]), nc, 0)
    m.c_Kn, m.c_Ks = np.full(nc, Kn), np.full(nc, Ks)
    m.c_tau, m.c_sig = np.full(nc, tau_max), np.full(nc, tau_max * peel_to_shear)
    m.c_GIc, m.c_GIIc = np.full(nc, GIc), np.full(nc, GIIc)
    m.c_mu = np.full(nc, 0.4)
    m.c_patch = np.zeros(nc, int)
    m.c_breakable = np.ones(nc, bool)
    m.params["glue_prepeak_onset"], m.params["glue_peak_slip_ratio"] = onset, peak_slip
    m.params["glue_coupon_time_s"] = coupon_time_s
    m.c_rate = np.full(nc, float(rate_exponent))
    # grips: the outer ends, clamped (no rotation); the B grip is pulled along +X
    return drive(m, {na[0]: "xyzr", nb[-1]: "yzr"}, nb[-1], 0, rate, max_disp=2.0,
                 stop_drop=0.2, test_time_s=test_time_s)


def shell_strip_tension(h: float, L: float = 20.0, W: float = 10.0, t: float = 2.0,
                        angles: tuple = (0.0,), weak: float = 0.95, rate: float = 40.0,
                        material=None, sim_params=None, grip: float = 1.0,
                        max_disp: float = 1.0) -> TestResult:
    """A plate strip (L along X, W across, thickness t, ply ``angles`` from X) meshed with
    right triangles of size ``h``, clamped at x = 0 and pulled along X at x = L until it
    breaks. The column of elements at mid-length is ``weak`` x as strong, so the crack
    forms there. With the crack-band regularisation the work to break the strip is
    G_f W t whatever ``h`` is (the mesh-convergence check of the shell damage).
    ``grip`` < 1 pulls only the middle share of the end (a stress concentration at the
    grip's edges)."""
    from bridgesim.schema import Material
    from bridgesim.sim.materials import Ply, SimParams

    sp = sim_params or SimParams.load()
    mat = material or Material.from_yaml(
        __import__("pathlib").Path(__file__).resolve().parents[3] / "materials"
        / "popsicle_birch.yaml")
    ply = Ply.from_material(mat, sp)
    nx, ny = max(2, round(L / h)), max(1, round(W / h))
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1)
    X = np.array([[x, y, 0.0] for x in xs for y in ys])
    m = blank_model(X)
    rho = mat.density_kg_m3.value * 1e-12  # t/mm^3

    def nid(i, j):
        return i * (ny + 1) + j

    G = (sp["wood.G_f_N_per_mm"], sp["wood.G_f90_N_per_mm"], sp["wood.G_v_N_per_mm"])
    els, nodes, E0s, Ps = [], [], [], []
    for i in range(nx):
        for j in range(ny):
            for tri in ((nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)),
                        (nid(i, j), nid(i + 1, j + 1), nid(i, j + 1))):
                P3 = X[list(tri)]
                c = P3.mean(0)
                ex = (P3[1] - P3[0]) / np.linalg.norm(P3[1] - P3[0])
                ez = np.array([0.0, 0.0, 1.0])
                ey = np.cross(ez, ex)
                E0 = np.stack([ex, ey, ez], axis=1)
                Ploc = (P3 - c) @ E0
                gang = math.degrees(math.atan2(float(np.array([1.0, 0, 0]) @ ey),
                                               float(np.array([1.0, 0, 0]) @ ex)))
                Sf = weak if abs(c[0] - L / 2) < 0.5 * L / nx else 1.0
                ed = shellmod.element_data(Ploc[:, :2], [gang + a for a in angles], t, ply,
                                           1.0, Sf, G, ply.E_L * t)
                els.append(ed)
                nodes.append(tri)
                E0s.append(E0)
                Ps.append(Ploc)
                me = rho * ed["area"] * t
                for nd in tri:
                    m.mass[nd] += me / 3
                    m.rot_inertia[nd] += me / 3 * (h * h / 12 + t * t / 12)
    for k, v in shellmod.pack(els).items():
        setattr(m, k, v)
    m.s_nodes = np.array(nodes, int)
    m.s_E0, m.s_P = np.array(E0s), np.array(Ps)
    m.s_t = np.full(len(nodes), t)
    m.s_body = np.zeros(len(nodes), int)
    m.params["crush_softening"] = sp["wood.crush_softening"]
    m.params["fc_proportional"] = sp.values.get("wood.fc_proportional", 1.0)
    m.params["crush_band_mm"] = sp.values.get("wood.crush_band_mm", 0.0)
    left = [nid(0, j) for j in range(ny + 1)]
    right = [nid(nx, j) for j in range(ny + 1)
             if abs(ys[j] - W / 2) <= grip * W / 2 + 1e-9]
    fixed = {n: "xyzr" for n in left} | {n: "yzr" for n in right}
    return drive(m, fixed, right, 0, rate, max_disp=max_disp, stop_drop=0.01, alpha=2000.0,
                 record_every=5)
