"""Time effects in the wood (round 28): duration of load and creep, in physical test time.

The crush compresses time (the simulated jack moves at mm/s, the real one at mm/min) and
maps it back to the physical test time ``st.t_test`` (the jack's travel at the real
pace, :class:`bridgesim.sim.crush.JackProtocol`), on which the glue's creep and its
rate-dependent strength already run (solver.glue_rate, glue_creep). This module gives
the wood the same clock. Each beam element and each plate element has its own time
under load ``t``, counted from when it first carried ``RATE_LOAD`` (a tenth) of its
strength, as each glue line's (solver.glue_rate):

* **Duration of load** (G1): its strengths at that time,
  f(t) = f_ref (t / t_ref)^-n, n = ``wood.rate_exponent``, t_ref = ``wood.coupon_time_s``
  (the coupons the strengths come from), t at least t_ref / kernels.RATE_RANGE as the
  glue's. A beam's f_t, f_c, G_f (the fracture energy follows the strength, as the stick
  field's factor makes it: kernels.beam_kernel) and its split strengths f_v, f_v3; a
  plate's ply failure strains (all five modes), its softening ratios keeping the final
  strain of each mode (G_f follows the strength there too).
* **Creep** (G2): an isochronous (secant) modulus,
  E(t) = E_ref (1 + phi(t_ref)) / (1 + phi(t)), phi(t) = a (t / 60 s)^m (relative creep,
  a = ``wood.creep_coeff`` after one minute under load, m = ``wood.creep_exponent``):
  the modulus the coupons measured (E_ref, a static test of t_ref) holds at t_ref; an
  element loaded for less is stiffer, up to the instantaneous E_ref (1 + phi(t_ref)), one
  held longer softer. A beam's E, G J, k G A (both directions) and a plate's ply
  stiffness and drilling stiffness scale alike. Creep does not lower a strength: the
  stress at which a fibre cracks or crushes stays f(t) (G1), so a member that creeps
  fails at a strain (1 + phi) (t / t_ref)^-n / (1 + phi(t_ref)) times its coupon's,
  within a few % of it over a test: the usual assumption that a creeping member fails
  at about its short-term strain; counting creep toward its failure as well (a fixed
  failure strain) would count the duration-of-load loss twice.

The kernels read these arrays every step, so nothing in them changes: the factors are
set from Python every ``EVERY`` steps (:func:`update`). A strength that falls below the
stress a fibre carries puts it on its softening branch (tension) or its plastic plateau
(compression) by the kernels' own laws, a step of at most n x (its time step) / t in
strength per update (about 0.1 %). Moduli only fall during a run, so the stable time
step taken at the start (with the instantaneous modulus, :func:`setup`) holds.
The pristine arrays are restored at every (re)start and when the run ends
(:func:`release`). With n = 0 and a = 0 nothing is touched: the model of round 27 bit
for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EVERY = 100  # steps between updates (~0.3 ms of simulated time, ~0.2 s of test time)
CREEP_REF_S = 60.0  # the time the creep coefficient is the relative creep at (s)
RATE_LOAD = 0.1  # share of its strength at which an element's clock starts (the glue's)
SHELL_TOL = 1e-4  # a plate element's strain limits are redone past this relative change

_BEAM = ("b_E", "b_ft", "b_fc", "b_Gf", "b_fv", "b_fv3", "b_GJ", "b_kGA", "b_kGA3")
# (not s_K, the elements' stiffness for the time-step estimates: crush.stable_dt refines
# them by power iteration on the forces, which read s_Q)
# (s_H, s_fil: the shear-deformable plates' transverse shear stiffness, which creeps with
# the moduli as a member's k G A does, and the inverse strengths of their glue planes,
# which follow the time under load as a member's split strengths do)
_SHELL = ("s_lim", "s_rf", "s_Q", "s_kd", "s_H", "s_fil")


@dataclass
class WoodTime:
    """The pristine arrays of a model, the time laws' parameters and each element's
    clock (the physical test time at which it first carried RATE_LOAD of its strength,
    -1 not yet)."""

    n: float  # wood.rate_exponent
    t_ref: float  # wood.coupon_time_s
    a: float  # wood.creep_coeff
    m: float  # wood.creep_exponent
    t_min: float  # the shortest time under load the strength law is taken at
    pristine: dict
    b_t0: np.ndarray
    s_t0: np.ndarray
    s_limfac: np.ndarray  # the factor each plate element's strain limits hold now

    @property
    def active(self) -> bool:
        return self.n > 0.0 or self.a > 0.0

    def phi(self, t) -> np.ndarray:
        """Relative creep after ``t`` s under load: a (t / 60 s)^m."""
        t = np.maximum(np.asarray(t, float), 0.0)
        return self.a * (t / CREEP_REF_S) ** self.m

    def strength_factor(self, tl: np.ndarray) -> np.ndarray:
        """(t / t_ref)^-n at the time under load ``tl`` (at least t_min)."""
        if self.n <= 0.0:
            return np.ones(len(tl))
        return np.exp(-self.n * np.log(np.maximum(tl, self.t_min) / self.t_ref))

    def modulus_factor(self, tl: np.ndarray) -> np.ndarray:
        """E(t) / E_ref = (1 + phi(t_ref)) / (1 + phi(t))."""
        if self.a <= 0.0:
            return np.ones(len(tl))
        return (1.0 + float(self.phi(self.t_ref))) / (1.0 + self.phi(tl))


def params_of(m) -> tuple[float, float, float, float]:
    """(n, t_ref, a, m) from a model's params (0, 60, 0, 0.25 when it has none: a coupon
    model of :mod:`bridgesim.sim.lab`, round 27's model)."""
    p = m.params
    return (float(p.get("wood_rate_exponent", 0.0)), float(p.get("wood_coupon_time_s", 60.0)),
            float(p.get("wood_creep_coeff", 0.0)), float(p.get("wood_creep_exponent", 0.25)))


def _restore(m, wt: WoodTime) -> None:
    for k, v in wt.pristine.items():
        getattr(m, k)[...] = v


def setup(m) -> WoodTime | None:
    """The wood's time laws for a run of model ``m`` (cached on it: a restarted run starts
    again from the pristine arrays): the pristine arrays restored, the clocks reset, and
    the moduli set to their instantaneous value E_ref (1 + phi(t_ref)) (no element under
    load yet; set before the stable time step is taken, since they only fall from there).
    None when both laws are off (n = 0 and a = 0): the arrays are not touched."""
    from bridgesim.sim.kernels import RATE_RANGE

    n, tref, a, mm = params_of(m)
    wt = getattr(m, "_wood_time", None)
    if wt is not None and len(wt.b_t0) == len(m.b_i) and len(wt.s_t0) == len(m.s_nodes):
        _restore(m, wt)
        wt.n, wt.t_ref, wt.a, wt.m = n, tref, a, mm
    elif n <= 0.0 and a <= 0.0:
        return None  # both laws off: nothing to keep
    else:
        names = [k for k in _BEAM if getattr(m, k).shape[:1] == m.b_i.shape] + \
            [k for k in _SHELL if getattr(m, k).shape[:1] == (len(m.s_nodes),)]
        wt = WoodTime(n, tref, a, mm, tref / RATE_RANGE,
                      {k: np.array(getattr(m, k), float, copy=True) for k in names},
                      np.zeros(0), np.zeros(0), np.zeros(0))
        m._wood_time = wt
    wt.t_min = wt.t_ref / RATE_RANGE
    wt.b_t0 = np.full(len(m.b_i), -1.0)
    wt.s_t0 = np.full(len(m.s_nodes), -1.0)
    wt.s_limfac = np.ones(len(m.s_nodes))
    if not wt.active:
        return None
    _apply(m, None, wt, 0.0)
    return wt


def release(m) -> None:
    """The pristine arrays back on the model, the cache dropped (a later run snapshots
    the model as it then is)."""
    wt = getattr(m, "_wood_time", None)
    if wt is not None:
        _restore(m, wt)
        del m._wood_time


def beam_utilisation(m, st, idx: np.ndarray) -> np.ndarray:
    """The largest stress / coupon-time strength of the fibres of beam elements ``idx``
    (at both Gauss points, f_t or f_c by the sign of the stress, the stick field's factor
    included), or their shear / split strength (st.b_rat2, b_rat3), from the local
    deformation the kernel left in ``st.b_prev`` (kernels.beam_kernel's strains, damage
    left out: a fibre that has cracked carried its strength long before)."""
    from bridgesim.sim import solver as S

    if not len(idx):
        return np.zeros(0)
    wt = m._wood_time
    P = wt.pristine
    L0 = m.b_L0[idx]
    D = st.b_prev[idx]
    eps0 = D[:, 0] / L0
    thi, thj = D[:, 1:4], D[:, 4:7]
    E = m.b_E[idx]
    fA, fy, fz = m.b_fA[idx], m.b_fy[idx], m.b_fz[idx]
    s2, s3 = st.b_s2[idx], st.b_s3[idx]
    EIz = (E[:, None] * fA * fy**2).sum(1) * s2**2
    EIy = (E[:, None] * fA * fz**2).sum(1) * s3**2
    kGA, kGA3 = m.b_kGA[idx], S.beam_kGA3(m)[idx]
    phz = np.where(kGA > 0, 12 * EIz / (np.where(kGA > 0, kGA, 1.0) * L0**2), 0.0)
    phy = np.where(kGA3 > 0, 12 * EIy / (np.where(kGA3 > 0, kGA3, 1.0) * L0**2), 0.0)
    gz = 0.5 * phz / (1 + phz) * (thi[:, 2] + thj[:, 2]) + st.b_gam[idx, 0]
    gy = 0.5 * phy / (1 + phy) * (thi[:, 1] + thj[:, 1]) + st.b_gam[idx, 1]
    bzi, bzj = thi[:, 2] - gz, thj[:, 2] - gz
    byi, byj = thi[:, 1] - gy, thj[:, 1] - gy
    fs = S.beam_fs(m)[idx]
    real = fA > 0.0  # fibres that carry area
    ft = np.maximum(P["b_ft"][idx][:, None] * fs, 1e-12)
    fc = np.maximum(P["b_fc"][idx][:, None] * fs, 1e-12)
    u = np.zeros(len(idx))
    for g, xi in enumerate(S.GP):
        kz = ((-4 + 6 * xi) * bzi + (-2 + 6 * xi) * bzj) / L0
        ky = ((-4 + 6 * xi) * byi + (-2 + 6 * xi) * byj) / L0
        eps = eps0[:, None] - s2[:, None] * fy * kz[:, None] + s3[:, None] * fz * ky[:, None]
        sig = E[:, None] * (eps - st.eps_p[idx, g])
        u = np.maximum(u, np.where(real, np.abs(sig) / np.where(sig > 0, ft, fc), 0.0).max(1))
    if len(st.b_rat2) == len(m.b_i):
        u = np.maximum(u, np.maximum(st.b_rat2[idx], st.b_rat3[idx]))
    return u


def _apply(m, st, wt: WoodTime, tnow: float) -> None:
    """Set the arrays for each element's time under load at test time ``tnow``."""
    P = wt.pristine
    if len(m.b_i):
        tl = np.where(wt.b_t0 >= 0.0, tnow - wt.b_t0, 0.0)
        r = wt.strength_factor(tl) if tnow > 0.0 else np.ones(len(tl))
        c = wt.modulus_factor(tl)
        for k in ("b_ft", "b_fc", "b_Gf", "b_fv", "b_fv3"):
            if k in P:
                getattr(m, k)[...] = P[k] * r
        for k in ("b_E", "b_GJ", "b_kGA", "b_kGA3"):
            if k in P:
                getattr(m, k)[...] = P[k] * c
    if len(m.s_nodes) and "s_lim" in P:
        tl = np.where(wt.s_t0 >= 0.0, tnow - wt.s_t0, 0.0)
        r = wt.strength_factor(tl) if tnow > 0.0 else np.ones(len(tl))
        c = wt.modulus_factor(tl)
        lf = r / c  # the failure strains: the strength over the modulus
        m.s_Q[...] = P["s_Q"] * c[:, None]
        m.s_kd[...] = P["s_kd"] * c
        if "s_H" in P:
            m.s_H[...] = P["s_H"] * c[:, None]
        if "s_fil" in P:
            m.s_fil[...] = P["s_fil"] / r[:, None]
        # the final strain of each softening mode stays (G_f follows the strength), so
        # its ratio to the onset strain goes as 1 / lf (an element at the snap-back limit,
        # shell.element_data, keeps 1.05)
        m.s_rf[...] = np.maximum(P["s_rf"] / lf[:, None], 1.05)
        redo = np.abs(lf / wt.s_limfac - 1.0) > SHELL_TOL
        if st is None:
            redo[:] = True
        if redo.any():
            m.s_lim[redo] = P["s_lim"][redo] * lf[redo, None]
            if st is not None and len(st.s_hist) == len(m.s_nodes):
                # the history holds strain / limit: kept as a strain (the state of the
                # softening is the largest strain reached, kernels._shell_one)
                st.s_hist[redo] *= (wt.s_limfac[redo] / lf[redo])[:, None, None]
            wt.s_limfac[redo] = lf[redo]


def update(m, st, wt: WoodTime | None) -> None:
    """Start the clocks of the elements that now carry RATE_LOAD of their strength and
    set every element's strengths and moduli for its time under load (every EVERY steps
    of a crush; nothing before any physical test time has passed)."""
    if wt is None:
        return
    tnow = float(st.t_test)
    if tnow <= 0.0:
        return
    if len(m.b_i):
        idx = np.where((wt.b_t0 < 0.0) & st.b_alive)[0]
        if len(idx):
            on = beam_utilisation(m, st, idx) >= RATE_LOAD
            wt.b_t0[idx[on]] = tnow
    if len(m.s_nodes) and len(st.s_hist) == len(m.s_nodes):
        idx = np.where(wt.s_t0 < 0.0)[0]
        if len(idx):
            # the history holds each point's largest strain / limit, the limit the
            # pristine one x s_limfac: the stress over the coupon-time strength is that x
            # s_limfac x the modulus factor (of no time under load: not yet started)
            c0 = float(wt.modulus_factor(np.zeros(1))[0])
            h = st.s_hist[idx].reshape(len(idx), -1).max(1) * wt.s_limfac[idx] * c0
            wt.s_t0[idx[h >= RATE_LOAD]] = tnow
    _apply(m, st, wt, tnow)


def creep_time_for(wt: WoodTime, phi: float) -> float:
    """The time under load (s) at which the relative creep reaches ``phi``:
    60 (phi / a)^(1 / m)."""
    if wt.a <= 0.0 or phi <= 0.0:
        return math.inf if phi > 0.0 else 0.0
    return CREEP_REF_S * (phi / wt.a) ** (1.0 / wt.m)
