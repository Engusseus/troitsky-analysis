"""Compiled (numba) versions of the force kernels in :mod:`bridgesim.sim.solver`.

They compute exactly the same forces and history updates as the NumPy reference
kernels (``tests/test_sim_kernels.py`` checks that), element by element in plain loops,
which is 10-30x faster. If numba is not installed the solver falls back to NumPy.
"""

from __future__ import annotations

import math

import numpy as np

# the glue's rate law is taken no further than its data: a line failing at most this many
# times faster than its coupons (PVAc 16x the rate, Wernersson 1994; the adversarial
# review of round 25: a line picked up in a cascade 1 s before it failed read 1.51x)
RATE_RANGE = 16.0

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore[no-redef]
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

    prange = range  # type: ignore[assignment]

_GP0 = 0.5 - 0.5 / math.sqrt(3.0)
_GP1 = 0.5 + 0.5 / math.sqrt(3.0)


@njit(cache=True, fastmath=True, error_model="numpy")
def _log3(R, out):
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    ang = math.acos(tr)
    vx = 0.5 * (R[2, 1] - R[1, 2])
    vy = 0.5 * (R[0, 2] - R[2, 0])
    vz = 0.5 * (R[1, 0] - R[0, 1])
    s = math.sin(ang)
    if ang < 1e-4:
        fac = 1.0 + ang * ang / 6.0
    else:
        fac = ang / s if s > 1e-12 else ang
    out[0] = vx * fac
    out[1] = vy * fac
    out[2] = vz * fac


@njit(cache=True, fastmath=True, error_model="numpy")
def beam_kernel(x, R, bi, bj, L0, E0, fy, fz, fA, Em, ft, fc, Gf, GJ, Ar, fv, fv3, Wt, nply,
                kGA, gam, s2, s3, soft, eps_p, dmg_t, alive, fractured, shear, F, Mo, update,
                beta, dt, prev, rat2, rat3, kap):
    nb = bi.shape[0]
    nf = fA.shape[1]
    Ti = np.empty((3, 3))
    Tj = np.empty((3, 3))
    Rl = np.empty((3, 3))
    thi = np.empty(3)
    thj = np.empty(3)
    e1 = np.empty(3)
    e2 = np.empty(3)
    e3 = np.empty(3)
    for e in range(nb):
        a = bi[e]
        b = bj[e]
        l2 = 0.0
        for k in range(3):
            e1[k] = x[b, k] - x[a, k]
            l2 += e1[k] * e1[k]
        l = math.sqrt(l2)
        for k in range(3):
            e1[k] /= l
        for r in range(3):
            for c in range(3):
                si = 0.0
                sj = 0.0
                for k in range(3):
                    si += R[a, r, k] * E0[e, k, c]
                    sj += R[b, r, k] * E0[e, k, c]
                Ti[r, c] = si
                Tj[r, c] = sj
        q0 = 0.5 * (Ti[0, 1] + Tj[0, 1])
        q1 = 0.5 * (Ti[1, 1] + Tj[1, 1])
        q2 = 0.5 * (Ti[2, 1] + Tj[2, 1])
        e3[0] = e1[1] * q2 - e1[2] * q1
        e3[1] = e1[2] * q0 - e1[0] * q2
        e3[2] = e1[0] * q1 - e1[1] * q0
        n3 = math.sqrt(e3[0] ** 2 + e3[1] ** 2 + e3[2] ** 2)
        for k in range(3):
            e3[k] /= n3
        e2[0] = e3[1] * e1[2] - e3[2] * e1[1]
        e2[1] = e3[2] * e1[0] - e3[0] * e1[2]
        e2[2] = e3[0] * e1[1] - e3[1] * e1[0]
        # local rotations: E^T T
        for r in range(3):
            er = e1 if r == 0 else (e2 if r == 1 else e3)
            for c in range(3):
                Rl[r, c] = er[0] * Ti[0, c] + er[1] * Ti[1, c] + er[2] * Ti[2, c]
        _log3(Rl, thi)
        for r in range(3):
            er = e1 if r == 0 else (e2 if r == 1 else e3)
            for c in range(3):
                Rl[r, c] = er[0] * Tj[0, c] + er[1] * Tj[1, c] + er[2] * Tj[2, c]
        _log3(Rl, thj)
        L = L0[e]
        eps0 = (l - L) / L
        E_ = Em[e]
        et0 = ft[e] / E_
        lch = 0.5 * L
        eu = et0 + 2.0 * Gf[e] / (ft[e] * lch)
        if eu < 1.5 * et0:
            eu = 1.5 * et0
        ec0 = fc[e] / E_
        # Timoshenko shear flexibility: a constant shear strain gz (gy) takes part of the
        # end rotations, so the fibres bend by th - g. phi = 12 EI / (k G A L^2); the
        # elastic split phi/(2(1+phi)) (th_i + th_j) is exact (interdependent
        # interpolation), gam[e] corrects it when the fibres are no longer elastic so the
        # shear force k G A g stays equal to the bending shear (M_i + M_j) / L.
        EIz = 0.0
        EIy = 0.0
        for f in range(nf):
            EIz += E_ * fA[e, f] * fy[e, f] * fy[e, f] * s2[e] * s2[e]
            EIy += E_ * fA[e, f] * fz[e, f] * fz[e, f] * s3[e] * s3[e]
        phz = 0.0
        phy = 0.0
        if kGA[e] > 0.0:
            phz = 12.0 * EIz / (kGA[e] * L * L)
            phy = 12.0 * EIy / (kGA[e] * L * L)
        gz = 0.5 * phz / (1.0 + phz) * (thi[2] + thj[2]) + gam[e, 0]
        gy = 0.5 * phy / (1.0 + phy) * (thi[1] + thj[1]) + gam[e, 1]
        bzi = thi[2] - gz
        bzj = thj[2] - gz
        byi = thi[1] - gy
        byj = thj[1] - gy
        N = 0.0
        Mzi = 0.0
        Mzj = 0.0
        Myi = 0.0
        Myj = 0.0
        Atot = 0.0
        cracked_max = 0.0
        EIzt = 0.0  # tangent (secant in tension) bending stiffness of the fibres
        EIyt = 0.0
        for g in range(2):
            xi = _GP0 if g == 0 else _GP1
            kz = ((-4.0 + 6.0 * xi) * bzi + (-2.0 + 6.0 * xi) * bzj) / L
            ky = ((-4.0 + 6.0 * xi) * byi + (-2.0 + 6.0 * xi) * byj) / L
            Ng = 0.0
            Mz = 0.0
            My = 0.0
            crk = 0.0
            for f in range(nf):
                # s2, s3 < 1: the member has split (delaminated) and its parts bend
                # each about its own axis: n parts carry 1/n^2 of the bending stiffness
                eps = eps0 - s2[e] * fy[e, f] * kz + s3[e] * fz[e, f] * ky
                ep = eps_p[e, g, f]
                dm = dmg_t[e, g, f]
                ee = eps - ep
                if ee > et0:
                    dn = eu * (ee - et0) / (ee * (eu - et0))
                    if dn > 1.0:
                        dn = 1.0
                    if dn > dm:
                        dm = dn
                if ee > 0.0:
                    sig = (1.0 - dm) * E_ * ee
                else:
                    sig = E_ * ee
                # compression: linear to kap f_c, then rising smoothly to f_c (wood's
                # proportional limit; the plastic strain at f_c is (1 - kap) f_c / E),
                # then crushing down to (1 - soft) f_c over 5 more f_c / E
                q = -ep
                qp = (1.0 - kap) * ec0
                hard = 0.0
                if q < qp:
                    u = math.sqrt(q / qp) if q > 0.0 else 0.0
                    fce = fc[e] * (1.0 - (1.0 - kap) * (1.0 - u) * (1.0 - u))
                    hard = 1.0 - u  # tangent on the rising curve: (1 - u) E
                else:
                    r = (q - qp) / (5.0 * ec0)
                    if r > 1.0:
                        r = 1.0
                    fce = fc[e] * (1.0 - soft * r)
                Et = E_ if ee <= 0.0 else (1.0 - dm) * E_
                if sig < -fce:
                    ep = eps + fce / E_
                    sig = -fce
                    Et = hard * E_
                if update:
                    dmg_t[e, g, f] = dm
                    eps_p[e, g, f] = ep
                A = fA[e, f]
                EIzt += 0.5 * Et * A * fy[e, f] * fy[e, f] * s2[e] * s2[e]
                EIyt += 0.5 * Et * A * fz[e, f] * fz[e, f] * s3[e] * s3[e]
                if dm > 0.99:
                    crk += A
                if g == 0:
                    Atot += A
                Ng += sig * A
                Mz -= sig * A * fy[e, f] * s2[e]
                My += sig * A * fz[e, f] * s3[e]
            if crk > cracked_max:
                cracked_max = crk
            N += 0.5 * Ng
            Mzi += 0.5 * Mz * (-4.0 + 6.0 * xi)
            Mzj += 0.5 * Mz * (-2.0 + 6.0 * xi)
            Myi += 0.5 * My * (-4.0 + 6.0 * xi)
            Myj += 0.5 * My * (-2.0 + 6.0 * xi)
        if update and kGA[e] > 0.0:
            # one Newton step on the shear equilibrium, with the fibres' current tangent
            # (k G A (1 + phi) while elastic)
            gam[e, 0] += ((Mzi + Mzj) / L - kGA[e] * gz) / (kGA[e] + 12.0 * EIzt / (L * L))
            gam[e, 1] += ((Myi + Myj) / L - kGA[e] * gy) / (kGA[e] + 12.0 * EIyt / (L * L))
        sj = (s2[e] * s3[e]) ** 2
        T = sj * GJ[e] / L * (thj[0] - thi[0])
        if beta > 0.0:
            # material (stiffness-proportional) damping on the local deformation rates,
            # with the Timoshenko element's elastic stiffness
            EA = 0.0
            for f in range(nf):
                EA += E_ * fA[e, f]
            r0 = (l - L - prev[e, 0]) / dt
            rxi = (thi[0] - prev[e, 1]) / dt
            ryi = (thi[1] - prev[e, 2]) / dt
            rzi = (thi[2] - prev[e, 3]) / dt
            rxj = (thj[0] - prev[e, 4]) / dt
            ryj = (thj[1] - prev[e, 5]) / dt
            rzj = (thj[2] - prev[e, 6]) / dt
            N += beta * EA / L * r0
            T += beta * sj * GJ[e] / L * (rxj - rxi)
            cz = beta * EIz / (L * (1.0 + phz))
            cy = beta * EIy / (L * (1.0 + phy))
            Mzi += cz * ((4.0 + phz) * rzi + (2.0 - phz) * rzj)
            Mzj += cz * ((2.0 - phz) * rzi + (4.0 + phz) * rzj)
            Myi += cy * ((4.0 + phy) * ryi + (2.0 - phy) * ryj)
            Myj += cy * ((2.0 - phy) * ryi + (4.0 + phy) * ryj)
        if update:
            prev[e, 0] = l - L
            prev[e, 1] = thi[0]
            prev[e, 2] = thi[1]
            prev[e, 3] = thi[2]
            prev[e, 4] = thj[0]
            prev[e, 5] = thj[1]
            prev[e, 6] = thj[2]
        Vz = (Mzi + Mzj) / l
        Vy = (Myi + Myj) / l
        # erosion: an element bent or twisted > 1 rad at an end is removed (reported
        # as member_erosion, not as a fracture of its fibres)
        if update and alive[e] and (thi[0] ** 2 + thi[1] ** 2 + thi[2] ** 2 > 1.0
                                    or thj[0] ** 2 + thj[1] ** 2 + thj[2] ** 2 > 1.0):
            alive[e] = False
        if update:
            if alive[e]:
                # shear along e3 runs on the glue planes between the plies, along e2
                # across the width: the stress over strength of each, for the split check
                # (solver.split_members: averaged over a section depth along the member)
                # (torsion: the shear on each family of planes, Wt[e, 0] on those
                # normal to e2, Wt[e, 1] on the glue lines; materials.rect_torsion)
                rat3[e] = (1.5 * abs(Vy) / Ar[e] + abs(T) / Wt[e, 1]) / fv3[e]
                rat2[e] = (1.5 * abs(Vz) / Ar[e] + abs(T) / Wt[e, 0]) / fv[e]
            if cracked_max / Atot > 0.9:
                fractured[e] = True
        if not alive[e]:
            continue
        for k in range(3):
            fk = e1[k] * (-N) + e2[k] * Vz + e3[k] * (-Vy)
            F[a, k] += fk
            F[b, k] -= fk
            Mo[a, k] += e1[k] * (-T) + e2[k] * Myi + e3[k] * Mzi
            Mo[b, k] += e1[k] * T + e2[k] * Myj + e3[k] * Mzj


@njit(cache=True, fastmath=True, error_model="numpy")
def _soft(r, rf):
    """Damage of a linear-softening mode at history ratio r = strain / onset strain."""
    if r <= 1.0:
        return 0.0
    d = rf / (rf - 1.0) * (1.0 - 1.0 / r)
    return 1.0 if d > 1.0 else d


@njit(cache=True, fastmath=True, error_model="numpy")
def _plateau(r, rc):
    """Secant damage of a crushing mode: the stress falls from the strength to the plateau
    rc x strength over one more onset strain, then stays there."""
    if r <= 1.0:
        return 0.0
    u = r - 1.0
    if u > 1.0:
        u = 1.0
    return 1.0 - (1.0 - (1.0 - rc) * u) / r


@njit(cache=True, fastmath=True, error_model="numpy")
def _shell_one(e, x, R, sn, E0, P, Bm, Bb, area, kd, lay, nl, Q, lim, rf, rc, hist, dmg,
               gone, update, fe, me, beta, dt, prev, wd, wR, wth, ws):
    d = wd[e]
    Rl = wR[e]
    th = wth[e]
    w = ws[e]
    n0 = sn[e, 0]
    n1 = sn[e, 1]
    n2 = sn[e, 2]
    for a in range(3):
        for c in range(3):
            fe[e, a, c] = 0.0
            me[e, a, c] = 0.0
    if gone[e]:
        return
    c0 = (x[n0, 0] + x[n1, 0] + x[n2, 0]) / 3.0
    c1 = (x[n0, 1] + x[n1, 1] + x[n2, 1]) / 3.0
    c2 = (x[n0, 2] + x[n1, 2] + x[n2, 2]) / 3.0
    ax0 = x[n1, 0] - x[n0, 0]
    ax1 = x[n1, 1] - x[n0, 1]
    ax2 = x[n1, 2] - x[n0, 2]
    b0 = x[n2, 0] - x[n0, 0]
    b1 = x[n2, 1] - x[n0, 1]
    b2 = x[n2, 2] - x[n0, 2]
    z0 = ax1 * b2 - ax2 * b1
    z1 = ax2 * b0 - ax0 * b2
    z2 = ax0 * b1 - ax1 * b0
    nx = math.sqrt(ax0 * ax0 + ax1 * ax1 + ax2 * ax2)
    nz = math.sqrt(z0 * z0 + z1 * z1 + z2 * z2)
    ax0 /= nx
    ax1 /= nx
    ax2 /= nx
    z0 /= nz
    z1 /= nz
    z2 /= nz
    y0 = z1 * ax2 - z2 * ax1
    y1 = z2 * ax0 - z0 * ax2
    y2 = z0 * ax1 - z1 * ax0
    for a in range(3):
        nd = sn[e, a]
        dx = x[nd, 0] - c0
        dy = x[nd, 1] - c1
        dz = x[nd, 2] - c2
        d[6 * a] = ax0 * dx + ax1 * dy + ax2 * dz - P[e, a, 0]
        d[6 * a + 1] = y0 * dx + y1 * dy + y2 * dz - P[e, a, 1]
        d[6 * a + 2] = z0 * dx + z1 * dy + z2 * dz - P[e, a, 2]
        for cc in range(3):
            g0 = E0[e, 0, cc]
            g1 = E0[e, 1, cc]
            g2 = E0[e, 2, cc]
            w0 = R[nd, 0, 0] * g0 + R[nd, 0, 1] * g1 + R[nd, 0, 2] * g2
            w1 = R[nd, 1, 0] * g0 + R[nd, 1, 1] * g1 + R[nd, 1, 2] * g2
            w2 = R[nd, 2, 0] * g0 + R[nd, 2, 1] * g1 + R[nd, 2, 2] * g2
            Rl[0, cc] = ax0 * w0 + ax1 * w1 + ax2 * w2
            Rl[1, cc] = y0 * w0 + y1 * w1 + y2 * w2
            Rl[2, cc] = z0 * w0 + z1 * w1 + z2 * w2
        _log3(Rl, th)
        d[6 * a + 3] = th[0]
        d[6 * a + 4] = th[1]
        d[6 * a + 5] = th[2]
    if update:
        # erosion: an element whose corners have turned > 0.5 rad relative to it is torn
        # apart; it is removed (the small-strain element means nothing there)
        for a in range(3):
            r2 = d[6 * a + 3] ** 2 + d[6 * a + 4] ** 2 + d[6 * a + 5] ** 2
            if r2 > 0.25:
                gone[e] = True
                dmg[e] = 1.0
                return
    # dofs for the stresses: displacement + beta x rate (stiffness-proportional damping)
    for j in range(18):
        dj = d[j]
        w[j] = dj + beta * (dj - prev[e, j]) / dt if beta > 0.0 else dj
        if update:
            prev[e, j] = dj
    # membrane strains (constant) and curvatures at the 3 in-plane points; [i] from the
    # displacement (damage history), [18 + i] from displacement + damping (stresses)
    for i in range(3):
        s0 = 0.0
        s1 = 0.0
        for a in range(3):
            s0 += Bm[e, i, 2 * a] * d[6 * a] + Bm[e, i, 2 * a + 1] * d[6 * a + 1]
            s1 += Bm[e, i, 2 * a] * w[6 * a] + Bm[e, i, 2 * a + 1] * w[6 * a + 1]
        w[18 + i] = s0
        w[21 + i] = s1
    for g in range(3):
        for i in range(3):
            s0 = 0.0
            s1 = 0.0
            for a in range(3):
                for k in range(3):
                    bb = Bb[e, g, i, 3 * a + k]
                    s0 += bb * d[6 * a + 2 + k]
                    s1 += bb * w[6 * a + 2 + k]
            w[24 + 3 * g + i] = s0
            w[33 + 3 * g + i] = s1
    for k in range(42, 60):
        w[k] = 0.0
    q11 = Q[e, 0]
    q22 = Q[e, 1]
    q12 = Q[e, 2]
    q66 = Q[e, 3]
    nlay = lay.shape[1]
    crack = 0.0
    all_open = True
    for g in range(3):
        mincr = 1.0
        for ll in range(nl[e]):
            zb = lay[e, ll, 0]
            zt = lay[e, ll, 1]
            c = lay[e, ll, 2]
            s = lay[e, ll, 3]
            hh = zt - zb
            cc2 = c * c
            ss2 = s * s
            cs = c * s
            for q in range(3):
                if q == 0:
                    z = zb
                    wq = hh / 6.0
                elif q == 1:
                    z = 0.5 * (zb + zt)
                    wq = 4.0 * hh / 6.0
                else:
                    z = zt
                    wq = hh / 6.0
                exx = w[18] + z * w[24 + 3 * g]
                eyy = w[19] + z * w[25 + 3 * g]
                gxy = w[20] + z * w[26 + 3 * g]
                e1 = exx * cc2 + eyy * ss2 + gxy * cs
                e2 = exx * ss2 + eyy * cc2 - gxy * cs
                g12 = 2.0 * (eyy - exx) * cs + gxy * (cc2 - ss2)
                exx = w[21] + z * w[33 + 3 * g]
                eyy = w[22] + z * w[34 + 3 * g]
                gxy = w[23] + z * w[35 + 3 * g]
                e1t = exx * cc2 + eyy * ss2 + gxy * cs
                e2t = exx * ss2 + eyy * cc2 - gxy * cs
                g12t = 2.0 * (eyy - exx) * cs + gxy * (cc2 - ss2)
                p = (g * nlay + ll) * 3 + q
                if update:
                    if e1 > 0.0:
                        r = e1 / lim[e, 0]
                        if r > hist[e, p, 0]:
                            hist[e, p, 0] = r
                    else:
                        r = -e1 / lim[e, 1]
                        if r > hist[e, p, 1]:
                            hist[e, p, 1] = r
                    if e2 > 0.0:
                        r = e2 / lim[e, 2]
                        if r > hist[e, p, 2]:
                            hist[e, p, 2] = r
                    else:
                        r = -e2 / lim[e, 3]
                        if r > hist[e, p, 3]:
                            hist[e, p, 3] = r
                    r = abs(g12) / lim[e, 4]
                    if r > hist[e, p, 4]:
                        hist[e, p, 4] = r
                hft = hist[e, p, 0]
                hmt = hist[e, p, 2]
                hs = hist[e, p, 4]
                dft = _soft(hft, rf[e, 0])
                dmt = _soft(hmt, rf[e, 1])
                ds = _soft(hs, rf[e, 2])
                d1 = dft if e1t >= 0.0 else _plateau(hist[e, p, 1], rc)
                d2 = dmt if e2t >= 0.0 else _plateau(hist[e, p, 3], 1.0)
                dsh = 1.0 - (1.0 - ds) * (1.0 - dft) * (1.0 - dmt)
                cr = dft
                if dmt > cr:
                    cr = dmt
                if ds > cr:
                    cr = ds
                if cr < mincr:
                    mincr = cr
                if all_open:
                    op = hft / rf[e, 0]
                    if hmt / rf[e, 1] > op:
                        op = hmt / rf[e, 1]
                    if hs / rf[e, 2] > op:
                        op = hs / rf[e, 2]
                    if op < 2.0:
                        all_open = False
                k1 = 1.0 - d1
                k2 = 1.0 - d2
                sg1 = k1 * q11 * e1t + k1 * k2 * q12 * e2t
                sg2 = k1 * k2 * q12 * e1t + k2 * q22 * e2t
                t12 = (1.0 - dsh) * q66 * g12t
                sxx = sg1 * cc2 + sg2 * ss2 - 2.0 * t12 * cs
                syy = sg1 * ss2 + sg2 * cc2 + 2.0 * t12 * cs
                txy = (sg1 - sg2) * cs + t12 * (cc2 - ss2)
                w[42 + 3 * g] += wq * sxx
                w[43 + 3 * g] += wq * syy
                w[44 + 3 * g] += wq * txy
                w[51 + 3 * g] += wq * z * sxx
                w[52 + 3 * g] += wq * z * syy
                w[53 + 3 * g] += wq * z * txy
        if mincr > crack:
            crack = mincr
    if update:
        if crack > dmg[e]:
            dmg[e] = crack
        if all_open and nl[e] > 0:
            gone[e] = True  # a wide-open crack through every ply: the element is cut
    ar = area[e]
    na0 = (w[42] + w[45] + w[48]) / 3.0
    na1 = (w[43] + w[46] + w[49]) / 3.0
    na2 = (w[44] + w[47] + w[50]) / 3.0
    # drilling (Hughes-Brezzi): theta_z of each node against the in-plane rigid rotation
    om = 0.0
    for a in range(3):
        om += -0.5 * Bm[e, 1, 2 * a + 1] * w[6 * a] + 0.5 * Bm[e, 0, 2 * a] * w[6 * a + 1]
    kde = kd[e] * max(0.0, 1.0 - 10.0 * dmg[e])  # a cracking element: no drilling stiffness
    sdr = (w[5] - om) + (w[11] - om) + (w[17] - om)
    for a in range(3):
        f0 = ar * (Bm[e, 0, 2 * a] * na0 + Bm[e, 1, 2 * a] * na1 + Bm[e, 2, 2 * a] * na2) \
            + kde * sdr * 0.5 * Bm[e, 1, 2 * a + 1]
        f1 = ar * (Bm[e, 0, 2 * a + 1] * na0 + Bm[e, 1, 2 * a + 1] * na1
                   + Bm[e, 2, 2 * a + 1] * na2) - kde * sdr * 0.5 * Bm[e, 0, 2 * a]
        f2 = 0.0
        m0 = 0.0
        m1 = 0.0
        for g in range(3):
            mx = w[51 + 3 * g]
            my = w[52 + 3 * g]
            mxy = w[53 + 3 * g]
            f2 += Bb[e, g, 0, 3 * a] * mx + Bb[e, g, 1, 3 * a] * my + Bb[e, g, 2, 3 * a] * mxy
            m0 += (Bb[e, g, 0, 3 * a + 1] * mx + Bb[e, g, 1, 3 * a + 1] * my
                   + Bb[e, g, 2, 3 * a + 1] * mxy)
            m1 += (Bb[e, g, 0, 3 * a + 2] * mx + Bb[e, g, 1, 3 * a + 2] * my
                   + Bb[e, g, 2, 3 * a + 2] * mxy)
        f2 *= ar / 3.0
        m0 *= ar / 3.0
        m1 *= ar / 3.0
        m2 = kde * (w[6 * a + 5] - om)
        fe[e, a, 0] = ax0 * f0 + y0 * f1 + z0 * f2
        fe[e, a, 1] = ax1 * f0 + y1 * f1 + z1 * f2
        fe[e, a, 2] = ax2 * f0 + y2 * f1 + z2 * f2
        me[e, a, 0] = ax0 * m0 + y0 * m1 + z0 * m2
        me[e, a, 1] = ax1 * m0 + y1 * m1 + z1 * m2
        me[e, a, 2] = ax2 * m0 + y2 * m1 + z2 * m2


@njit(cache=True, fastmath=True, parallel=True, error_model="numpy")
def shell_kernel(x, R, sn, E0, P, Bm, Bb, area, kd, lay, nl, Q, lim, rf, rc, hist, dmg,
                 gone, F, Mo, update, fe, me, beta, dt, prev, wd, wR, wth, ws):
    ns = sn.shape[0]
    for e in prange(ns):
        _shell_one(e, x, R, sn, E0, P, Bm, Bb, area, kd, lay, nl, Q, lim, rf, rc, hist, dmg,
                   gone, update, fe, me, beta, dt, prev, wd, wR, wth, ws)
    for e in range(ns):
        for a in range(3):
            nd = sn[e, a]
            F[nd, 0] += fe[e, a, 0]
            F[nd, 1] += fe[e, a, 1]
            F[nd, 2] += fe[e, a, 2]
            Mo[nd, 0] += me[e, a, 0]
            Mo[nd, 1] += me[e, a, 1]
            Mo[nd, 2] += me[e, a, 2]


@njit(cache=True, fastmath=True, error_model="numpy")
def _apos(x, R, a_node, a_r0, att, p, arm):
    nd = a_node[att]
    for k in range(3):
        v = 0.0
        for j in range(3):
            v += R[nd, k, j] * a_r0[att, j]
        arm[k] = v
        p[k] = x[nd, k] + v
    return nd


@njit(cache=True, fastmath=True, error_model="numpy")
def _add(F, Mo, nd, arm, f, sign):
    F[nd, 0] += sign * f[0]
    F[nd, 1] += sign * f[1]
    F[nd, 2] += sign * f[2]
    Mo[nd, 0] += sign * (arm[1] * f[2] - arm[2] * f[1])
    Mo[nd, 1] += sign * (arm[2] * f[0] - arm[0] * f[2])
    Mo[nd, 2] += sign * (arm[0] * f[1] - arm[1] * f[0])


@njit(cache=True, fastmath=True, error_model="numpy")
def _trilinear_damage(dmx, dm0, keff, Gc, ga, gp):
    """Damage of a glue point at the largest mixed-mode separation ``dmx`` reached: the
    traction rises with keff to ga x its strength (at ga dm0), bends over to the
    strength at gp dm0 (a PVA line's pre-peak work, Serrano 2000) and falls linearly to
    nothing at the separation where the work done is Gc. ga = gp = 1: bilinear."""
    e = ga * dm0
    if dmx <= e:
        return 0.0
    q = gp * dm0
    lf = gp + 2.0 * Gc / (keff * dm0 * dm0) - ga * ga - (ga + 1.0) * (gp - ga)
    if lf < 1.05 * gp:
        lf = 1.05 * gp
    dmf = lf * dm0
    if dmx <= q:
        T = dm0 * (ga + (1.0 - ga) * (dmx - e) / (q - e))
    elif dmx < dmf:
        T = dm0 * (dmf - dmx) / (dmf - q)
    else:
        return 1.0
    d = 1.0 - T / dmx
    if d < 0.0:
        return 0.0
    if d > 1.0:
        return 1.0
    return d


@njit(cache=True, fastmath=True, error_model="numpy")
def glue_kernel(x, R, a_node, a_r0, ca, cb, area, n0, Kn, Ks, sig, tau, GIc, GIIc, mu,
                breakable, dmax, dd, slip, F, Mo, update, beta, dt, prev, alive, crush,
                fc90, cdep, eps_d, E_d, fcap, creep, cfac, cdec, ga, gp, crate, tnow,
                tref, t0, gap0):
    nc = ca.shape[0]
    pa = np.empty(3)
    pb = np.empty(3)
    arma = np.empty(3)
    armb = np.empty(3)
    n = np.empty(3)
    ds = np.empty(3)
    es = np.empty(3)
    tf = np.empty(3)
    f = np.empty(3)
    for i in range(nc):
        if not alive[i]:
            continue
        na = _apos(x, R, a_node, a_r0, ca[i], pa, arma)
        nb = _apos(x, R, a_node, a_r0, cb[i], pb, armb)
        nn = 0.0
        for k in range(3):
            v = 0.0
            for j in range(3):
                v += (R[na, k, j] + R[nb, k, j]) * n0[i, j]
            n[k] = v
            nn += v * v
        nn = math.sqrt(nn)
        dn = 0.0
        for k in range(3):
            n[k] /= nn
            dn += (pb[k] - pa[k]) * n[k]
        for k in range(3):
            ds[k] = pb[k] - pa[k] - dn * n[k]
        # creep of the glue line in shear (a Kelvin element in series with the spring,
        # in physical test time): the elastic slip es = ds - creep carries the stress
        # and drives the damage; the creep slip moves toward cfac x ds by the factor
        # cdec per step (cfac = phi / (1 + phi), cdec = exp(-(1 + phi) dt_test / t_r))
        en = 0.0
        for k in range(3):
            es[k] = ds[k] - creep[i, k]
            en += es[k] * n[k]
        s2 = 0.0
        for k in range(3):
            es[k] -= en * n[k]
            s2 += es[k] * es[k]
        s = math.sqrt(s2)
        kn = Kn[i]
        # the wood at a glue line crushes (permanent set) at fc90[i]: across or along
        # its grain depending on the joint (Hankinson). It hardens as the crushed zone
        # (depth cdep[i]) compacts, and once its cells have collapsed (eps_d of the depth)
        # it is densified wood, about as stiff as wood across the grain (E_d)
        dn = dn - crush[i]
        q = -crush[i] / cdep[i]
        if q < eps_d:
            fy = fc90[i] * (1.0 + q)
            hh = fc90[i] / cdep[i]
        else:
            fy = fc90[i] * (1.0 + eps_d) + E_d * (q - eps_d)
            hh = E_d / cdep[i]
        if fy > fcap:  # no wood bears more than it crushes at along its grain
            fy = fcap
            hh = 0.0
        if kn * dn < -fy:
            dp = (-kn * dn - fy) / (kn + hh)
            if update:
                crush[i] -= dp
            dn = dn + dp
        dnp = dn if dn > 0.0 else 0.0
        ks = Ks[i]
        d = 0.0
        if breakable[i]:
            # the strength at the line's own time under load (glue_rate in solver.py): its
            # clock starts when it reaches RATE_LOAD of its mixed-mode onset
            dm = math.sqrt(dnp * dnp + s2)
            dn0 = sig[i] / kn
            ds0 = tau[i] / ks
            if dnp > 1e-12:
                mix = s / dnp
                dm0 = dn0 * ds0 * math.sqrt((1.0 + mix * mix)
                                            / (ds0 * ds0 + (mix * dn0) ** 2))
            else:
                dm0 = ds0
            if update and tnow > 0.0 and t0[i] < 0.0 and dm > 0.1 * dm0:  # RATE_LOAD
                t0[i] = tnow
            if crate[i] != 0.0 and tnow > 0.0 and tref > 0.0:
                tl = tnow - t0[i] if t0[i] >= 0.0 else 0.0
                tmin = tref / RATE_RANGE
                rs = math.exp(-crate[i] * math.log((tl if tl > tmin else tmin) / tref))
                dn0 *= rs
                ds0 *= rs
                dm0 *= rs
            Gs = ks * s2
            Gn = kn * dnp * dnp
            B = Gs / (Gs + Gn) if Gs + Gn > 1e-30 else 0.0
            Gc = GIc[i] + (GIIc[i] - GIc[i]) * B * B
            keff = (Gn + Gs) / (dm * dm) if dm > 1e-12 else ks
            dmx = dmax[i] if dmax[i] > dm else dm
            dnew = _trilinear_damage(dmx, dm0 if dm0 > 1e-12 else 1e-12, keff, Gc, ga, gp)
            d = dd[i] if dd[i] > dnew else dnew
            if update:
                dmax[i] = dmx
                dd[i] = d
                if cfac > 0.0 and d < 0.999:
                    for k in range(3):
                        creep[i, k] = cfac * ds[k] + (creep[i, k] - cfac * ds[k]) * cdec
            # a debonded pair that has turned, parted (3 mm from where it was drawn: a
            # dry face drawn apart by gap0 starts that far open), or crushed through its
            # whole depth: the two points no longer face each other, so the pair stops
            if d >= 0.999 and (nn < 1.6 or crush[i] < -cdep[i]
                               or (dn - gap0[i]) ** 2 + s2 > 9.0):
                if update:
                    alive[i] = False
                continue
        tn = (1.0 - d) * kn * dn if dn > 0.0 else kn * dn
        rn = 0.0
        for k in range(3):
            tf[k] = ds[k] - slip[i, k]
            rn += tf[k] * n[k]
        mag2 = 0.0
        for k in range(3):
            tf[k] = ks * (tf[k] - rn * n[k])
            mag2 += tf[k] * tf[k]
        mag = math.sqrt(mag2)
        cap = mu[i] * (-tn if tn < 0.0 else 0.0)
        if mag > cap:
            scl = cap / mag if mag > 1e-30 else 0.0
            for k in range(3):
                tf[k] *= scl
        if update:
            if d > 0.0:
                for k in range(3):
                    slip[i, k] = ds[k] - tf[k] / ks
            else:
                for k in range(3):
                    slip[i, k] = ds[k]
        for k in range(3):
            f[k] = (tn * n[k] + (1.0 - d) * ks * es[k] + d * tf[k]) * area[i]
        if beta > 0.0:
            dnr = 0.0
            for k in range(3):
                dnr += (pb[k] - pa[k] - prev[i, k]) * n[k]
            for k in range(3):
                rate = (pb[k] - pa[k] - prev[i, k]) / dt
                rn_ = dnr / dt * n[k]
                kk = kn if dn <= 0.0 else (1.0 - d) * kn
                f[k] += beta * (kk * rn_ + (1.0 - d) * ks * (rate - rn_)) * area[i]
        if update:
            for k in range(3):
                prev[i, k] = pb[k] - pa[k]
        # f is the pull of the spring ON node a (towards b); F holds internal forces
        _add(F, Mo, na, arma, f, -1.0)
        _add(F, Mo, nb, armb, f, 1.0)


@njit(cache=True, fastmath=True, error_model="numpy")
def cable_kernel(x, R, a_node, a_r0, ka, kb, L0, EA, Fb, alive, F, Mo, update):
    pa = np.empty(3)
    pb = np.empty(3)
    arma = np.empty(3)
    armb = np.empty(3)
    f = np.empty(3)
    for i in range(ka.shape[0]):
        na = _apos(x, R, a_node, a_r0, ka[i], pa, arma)
        nb = _apos(x, R, a_node, a_r0, kb[i], pb, armb)
        L = math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2)
        T = EA[i] * (L - L0[i]) / L0[i]
        if T < 0.0:
            T = 0.0
        if update and T >= Fb[i]:
            alive[i] = False
        if not alive[i]:
            continue
        for k in range(3):
            f[k] = (pb[k] - pa[k]) / max(L, 1e-9) * T
        # f is the pull of the spring ON node a (towards b); F holds internal forces
        _add(F, Mo, na, arma, f, -1.0)
        _add(F, Mo, nb, armb, f, 1.0)


@njit(cache=True, fastmath=True, error_model="numpy")
def _seated(k, pen, seat):
    """Normal force of a bearing face pressed ``pen`` into the steel: its stiffness grows
    from 0 to k over the first ``seat`` mm (roughness, glue beads, fibres crushing at the
    first touch), the soft start of a real load-deflection curve."""
    if seat <= 0.0:
        return k * pen
    if pen < seat:
        return 0.5 * k * pen * pen / seat
    return k * (pen - 0.5 * seat)


@njit(cache=True, fastmath=True, error_model="numpy")
def platform_kernel(x, R, a_node, a_r0, att, table_y, kk_, mu, anchor, touch, F, Mo, update,
                    vel, omg, cdamp, loose, fmax, lift, seat):
    p = np.empty(3)
    arm = np.empty(3)
    f = np.empty(3)
    for i in range(att.shape[0]):
        if loose[a_node[att[i]]]:
            touch[i] = False
            continue
        nd = _apos(x, R, a_node, a_r0, att[i], p, arm)
        pen = table_y - p[1] - lift[i]
        if pen <= 0.0:
            if update:
                touch[i] = False
                for kk in range(3):
                    anchor[i, kk] = p[kk]
            continue
        k = kk_[i]
        # contact damping on the approach speed (never pulls the point down)
        vy = vel[nd, 1] + omg[nd, 2] * arm[0] - omg[nd, 0] * arm[2]
        fn = _seated(k, pen, seat) - cdamp[i] * vy
        if fn > fmax[i]:
            fn = fmax[i]  # the wood crushes locally (bearing)
        if fn < 0.0:
            fn = 0.0
        f0 = -k * (p[0] - anchor[i, 0])
        f2 = -k * (p[2] - anchor[i, 2])
        mag = math.sqrt(f0 * f0 + f2 * f2)
        cap = mu * fn
        if mag > cap:
            sc = cap / mag
            f0 *= sc
            f2 *= sc
        if update:
            touch[i] = True
            anchor[i, 0] = p[0] + f0 / k
            anchor[i, 1] = p[1]
            anchor[i, 2] = p[2] + f2 / k
        f[0] = f0
        f[1] = fn
        f[2] = f2
        _add(F, Mo, nd, arm, f, 1.0)


@njit(cache=True, fastmath=True, error_model="numpy")
def plate_kernel(x, R, a_node, a_r0, att, P, h, kk_, mu, anchor, touch, F, Mo, update,
                 vel, omg, cdamp, loose, fmax, seat):
    p = np.empty(3)
    arm = np.empty(3)
    loc = np.empty(3)
    nl = np.empty(3)
    ftl = np.empty(3)
    f = np.empty(3)
    total_down = 0.0
    for i in range(att.shape[0]):
        if loose[a_node[att[i]]]:
            touch[i] = False
            continue
        nd = _apos(x, R, a_node, a_r0, att[i], p, arm)
        for j in range(3):
            v = 0.0
            for kk in range(3):
                v += (p[kk] - x[P, kk]) * R[P, kk, j]
            loc[j] = v
        inside = True
        best = 1e30
        ax = 0
        for j in range(3):
            dpt = h[j] - abs(loc[j])
            if dpt <= 0.0:
                inside = False
                break
            if dpt < best:
                best = dpt
                ax = j
        if not inside:
            if update:
                touch[i] = False
                for j in range(3):
                    anchor[i, j] = loc[j]
            continue
        for j in range(3):
            nl[j] = 0.0
        nl[ax] = 1.0 if loc[ax] > 0 else -1.0
        k = kk_[i]
        # relative velocity of the point with respect to the plate, along the normal
        vr = 0.0
        for j in range(3):
            r0 = p[(j + 1) % 3] - x[P, (j + 1) % 3]
            r1 = p[(j + 2) % 3] - x[P, (j + 2) % 3]
            vp_ = vel[P, j] + omg[P, (j + 1) % 3] * r1 - omg[P, (j + 2) % 3] * r0
            vb_ = vel[nd, j] + omg[nd, (j + 1) % 3] * arm[(j + 2) % 3] \
                - omg[nd, (j + 2) % 3] * arm[(j + 1) % 3]
            gn = R[P, j, 0] * nl[0] + R[P, j, 1] * nl[1] + R[P, j, 2] * nl[2]
            vr += (vb_ - vp_) * gn
        fn = _seated(k, best, seat) - cdamp[i] * vr
        if fn > fmax[i]:
            fn = fmax[i]  # the wood crushes locally under the plate (bearing)
        if fn < 0.0:
            fn = 0.0
        rn = 0.0
        for j in range(3):
            ftl[j] = loc[j] - anchor[i, j]
            rn += ftl[j] * nl[j]
        mag2 = 0.0
        for j in range(3):
            ftl[j] = -k * (ftl[j] - rn * nl[j])
            mag2 += ftl[j] * ftl[j]
        mag = math.sqrt(mag2)
        cap = mu * fn
        if mag > cap:
            sc = cap / mag
            for j in range(3):
                ftl[j] *= sc
        if update:
            touch[i] = True
            for j in range(3):
                anchor[i, j] = loc[j] + ftl[j] / k
        for j in range(3):
            v = 0.0
            for kk in range(3):
                v += R[P, j, kk] * (fn * nl[kk] + ftl[kk])
            f[j] = v
        _add(F, Mo, nd, arm, f, 1.0)
        total_down -= f[1]
        F[P, 0] -= f[0]
        F[P, 1] -= f[1]
        F[P, 2] -= f[2]
        r0 = p[0] - x[P, 0]
        r1 = p[1] - x[P, 1]
        r2 = p[2] - x[P, 2]
        Mo[P, 0] -= r1 * f[2] - r2 * f[1]
        Mo[P, 1] -= r2 * f[0] - r0 * f[2]
        Mo[P, 2] -= r0 * f[1] - r1 * f[0]
    return total_down


@njit(cache=True, fastmath=True, error_model="numpy")
def expmap_update(R, w, dt):
    for i in range(R.shape[0]):
        wx = w[i, 0] * dt
        wy = w[i, 1] * dt
        wz = w[i, 2] * dt
        th = math.sqrt(wx * wx + wy * wy + wz * wz)
        if th < 1e-12:
            continue
        kx, ky, kz = wx / th, wy / th, wz / th
        s = math.sin(th)
        c = 1.0 - math.cos(th)
        Q = np.empty((3, 3))
        Q[0, 0] = 1.0 + c * (-ky * ky - kz * kz)
        Q[0, 1] = -s * kz + c * kx * ky
        Q[0, 2] = s * ky + c * kx * kz
        Q[1, 0] = s * kz + c * kx * ky
        Q[1, 1] = 1.0 + c * (-kx * kx - kz * kz)
        Q[1, 2] = -s * kx + c * ky * kz
        Q[2, 0] = -s * ky + c * kx * kz
        Q[2, 1] = s * kx + c * ky * kz
        Q[2, 2] = 1.0 + c * (-kx * kx - ky * ky)
        M = np.empty((3, 3))
        for r in range(3):
            for cc in range(3):
                M[r, cc] = Q[r, 0] * R[i, 0, cc] + Q[r, 1] * R[i, 1, cc] + Q[r, 2] * R[i, 2, cc]
        for r in range(3):
            for cc in range(3):
                R[i, r, cc] = M[r, cc]


@njit(cache=True, fastmath=True, error_model="numpy")
def integrate(x, v, w, R, F, Mo, mass, gmass, rot, gy, alpha, dt, loose, floor):
    """One central-difference update of all nodes (velocities at half steps), with
    mass-proportional damping; gravity acts on ``gmass`` (the physical mass)."""
    for i in range(x.shape[0]):
        if loose[i]:  # debris: falls under gravity and lands on the platform
            v[i, 1] += dt * gmass[i] * gy / mass[i]
            x[i, 0] += dt * v[i, 0]
            x[i, 1] += dt * v[i, 1]
            x[i, 2] += dt * v[i, 2]
            if x[i, 1] < floor:
                x[i, 1] = floor
                if v[i, 1] < 0.0:
                    v[i, 1] = 0.0
                v[i, 0] *= 0.99
                v[i, 2] *= 0.99
            w[i, 0] = 0.0
            w[i, 1] = 0.0
            w[i, 2] = 0.0
            continue
        im = 1.0 / mass[i]
        v[i, 0] += dt * (-F[i, 0] * im - alpha * v[i, 0])
        v[i, 1] += dt * ((-F[i, 1] + gmass[i] * gy) * im - alpha * v[i, 1])
        v[i, 2] += dt * (-F[i, 2] * im - alpha * v[i, 2])
        x[i, 0] += dt * v[i, 0]
        x[i, 1] += dt * v[i, 1]
        x[i, 2] += dt * v[i, 2]
        ir = 1.0 / rot[i]
        w[i, 0] += dt * (-Mo[i, 0] * ir - alpha * w[i, 0])
        w[i, 1] += dt * (-Mo[i, 1] * ir - alpha * w[i, 1])
        w[i, 2] += dt * (-Mo[i, 2] * ir - alpha * w[i, 2])
    expmap_update(R, w, dt)
