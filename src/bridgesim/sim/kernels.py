"""Compiled (numba) versions of the force kernels in :mod:`bridgesim.sim.solver`.

They compute exactly the same forces and history updates as the NumPy reference
kernels (``tests/test_sim_kernels.py`` checks that), element by element in plain loops,
which is 10-30x faster. If numba is not installed the solver falls back to NumPy.
"""

from __future__ import annotations

import math

import numpy as np

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
def beam_kernel(x, R, bi, bj, L0, E0, fy, fz, fA, Em, ft, fc, Gf, GJ, Ar, fv, Wt, soft,
                eps_p, dmg_t, alive, fractured, shear, F, Mo, update, beta, dt, prev):
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
        N = 0.0
        Mzi = 0.0
        Mzj = 0.0
        Myi = 0.0
        Myj = 0.0
        Atot = 0.0
        cracked_max = 0.0
        for g in range(2):
            xi = _GP0 if g == 0 else _GP1
            kz = ((-4.0 + 6.0 * xi) * thi[2] + (-2.0 + 6.0 * xi) * thj[2]) / L
            ky = ((-4.0 + 6.0 * xi) * thi[1] + (-2.0 + 6.0 * xi) * thj[1]) / L
            Ng = 0.0
            Mz = 0.0
            My = 0.0
            crk = 0.0
            for f in range(nf):
                eps = eps0 - fy[e, f] * kz + fz[e, f] * ky
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
                r = -ep / (5.0 * ec0)
                if r > 1.0:
                    r = 1.0
                elif r < 0.0:
                    r = 0.0
                fce = fc[e] * (1.0 - soft * r)
                if sig < -fce:
                    ep = eps + fce / E_
                    sig = -fce
                if update:
                    dmg_t[e, g, f] = dm
                    eps_p[e, g, f] = ep
                A = fA[e, f]
                if dm > 0.99:
                    crk += A
                if g == 0:
                    Atot += A
                Ng += sig * A
                Mz -= sig * A * fy[e, f]
                My += sig * A * fz[e, f]
            if crk > cracked_max:
                cracked_max = crk
            N += 0.5 * Ng
            Mzi += 0.5 * Mz * (-4.0 + 6.0 * xi)
            Mzj += 0.5 * Mz * (-2.0 + 6.0 * xi)
            Myi += 0.5 * My * (-4.0 + 6.0 * xi)
            Myj += 0.5 * My * (-2.0 + 6.0 * xi)
        T = GJ[e] / L * (thj[0] - thi[0])
        if beta > 0.0:
            # material (stiffness-proportional) damping on the local deformation rates
            EA = 0.0
            EIz = 0.0
            EIy = 0.0
            for f in range(nf):
                EA += E_ * fA[e, f]
                EIz += E_ * fA[e, f] * fy[e, f] * fy[e, f]
                EIy += E_ * fA[e, f] * fz[e, f] * fz[e, f]
            r0 = (l - L - prev[e, 0]) / dt
            rxi = (thi[0] - prev[e, 1]) / dt
            ryi = (thi[1] - prev[e, 2]) / dt
            rzi = (thi[2] - prev[e, 3]) / dt
            rxj = (thj[0] - prev[e, 4]) / dt
            ryj = (thj[1] - prev[e, 5]) / dt
            rzj = (thj[2] - prev[e, 6]) / dt
            N += beta * EA / L * r0
            T += beta * GJ[e] / L * (rxj - rxi)
            Mzi += beta * EIz / L * (4.0 * rzi + 2.0 * rzj)
            Mzj += beta * EIz / L * (2.0 * rzi + 4.0 * rzj)
            Myi += beta * EIy / L * (4.0 * ryi + 2.0 * ryj)
            Myj += beta * EIy / L * (2.0 * ryi + 4.0 * ryj)
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
        # erosion: an element bent or twisted > 1 rad at an end is broken
        if update and alive[e] and (thi[0] ** 2 + thi[1] ** 2 + thi[2] ** 2 > 1.0
                                    or thj[0] ** 2 + thj[1] ** 2 + thj[2] ** 2 > 1.0):
            alive[e] = False
            fractured[e] = True
        if update:
            if alive[e]:
                tau = 1.5 * math.sqrt(Vz * Vz + Vy * Vy) / Ar[e] + abs(T) / Wt[e]
                if tau > fv[e]:
                    shear[e] = True
                    alive[e] = False
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
def _shell_one(e, x, R, sn, K, E0, P, Rr, t, ang, lim, matrix, dmg, update, fe, me, beta,
               dt, prev, wd, wR, wth):
    d = wd[e]
    Rl = wR[e]
    th = wth[e]
    n0 = sn[e, 0]
    n1 = sn[e, 1]
    n2 = sn[e, 2]
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
    if update and dmg[e] < 1.0:
        # erosion: an element whose corners have turned > 0.5 rad relative to it is
        # broken; it is removed (its local rotations would leave the range where the
        # small-strain element means anything)
        for a in range(3):
            r2 = d[6 * a + 3] ** 2 + d[6 * a + 4] ** 2 + d[6 * a + 5] ** 2
            if r2 > 0.25:
                dmg[e] = 1.0
    if update and dmg[e] < 0.97:
        e0 = 0.0
        e1_ = 0.0
        e2_ = 0.0
        k0 = 0.0
        k1 = 0.0
        k2 = 0.0
        for j in range(18):
            dj = d[j]
            e0 += Rr[e, 0, j] * dj
            e1_ += Rr[e, 1, j] * dj
            e2_ += Rr[e, 2, j] * dj
            k0 += Rr[e, 3, j] * dj
            k1 += Rr[e, 4, j] * dj
            k2 += Rr[e, 5, j] * dj
        fi = 0.0
        for zs in range(2):
            zf = -0.5 if zs == 0 else 0.5
            exx = e0 + zf * t[e] * k0
            eyy = e1_ + zf * t[e] * k1
            gxy = e2_ + zf * t[e] * k2
            for kk in range(2):
                an = ang[e, kk]
                if an != an:
                    continue
                ar = math.radians(an)
                cs = math.cos(ar)
                s_ = math.sin(ar)
                p1 = exx * cs * cs + eyy * s_ * s_ + gxy * cs * s_
                p2 = exx * s_ * s_ + eyy * cs * cs - gxy * cs * s_
                g12 = 2.0 * (eyy - exx) * cs * s_ + gxy * (cs * cs - s_ * s_)
                r1 = p1 / lim[e, 0] if p1 > 0 else -p1 / lim[e, 1]
                r3 = abs(g12) / lim[e, 4]
                r = r1 if r1 > r3 else r3
                if matrix[e]:
                    r2 = p2 / lim[e, 2] if p2 > 0 else -p2 / lim[e, 3]
                    if r2 > r:
                        r = r2
                if r > fi:
                    fi = r
        if fi >= 1.0:
            # cracking develops over a small strain range (energy released gradually)
            target = 2.0 * (fi - 1.0) + 0.1
            if target > 0.97:
                target = 0.97
            if target > dmg[e]:
                dmg[e] = target
    sc = 1.0 - dmg[e]
    if beta > 0.0:
        for j in range(18):
            dj = d[j]
            d[j] = dj + beta * (dj - prev[e, j]) / dt
            if update:
                prev[e, j] = dj
    elif update:
        for j in range(18):
            prev[e, j] = d[j]
    for a in range(3):
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        m0 = 0.0
        m1 = 0.0
        m2 = 0.0
        r0 = 6 * a
        for j in range(18):
            dj = d[j]
            f0 += K[e, r0, j] * dj
            f1 += K[e, r0 + 1, j] * dj
            f2 += K[e, r0 + 2, j] * dj
            m0 += K[e, r0 + 3, j] * dj
            m1 += K[e, r0 + 4, j] * dj
            m2 += K[e, r0 + 5, j] * dj
        f0 *= sc
        f1 *= sc
        f2 *= sc
        m0 *= sc
        m1 *= sc
        m2 *= sc
        fe[e, a, 0] = ax0 * f0 + y0 * f1 + z0 * f2
        fe[e, a, 1] = ax1 * f0 + y1 * f1 + z1 * f2
        fe[e, a, 2] = ax2 * f0 + y2 * f1 + z2 * f2
        me[e, a, 0] = ax0 * m0 + y0 * m1 + z0 * m2
        me[e, a, 1] = ax1 * m0 + y1 * m1 + z1 * m2
        me[e, a, 2] = ax2 * m0 + y2 * m1 + z2 * m2


@njit(cache=True, fastmath=True, parallel=True, error_model="numpy")
def shell_kernel(x, R, sn, K, E0, P, Rr, t, ang, lim, matrix, dmg, F, Mo, update, fe, me,
                 beta, dt, prev, wd, wR, wth):
    ns = sn.shape[0]
    for e in prange(ns):
        _shell_one(e, x, R, sn, K, E0, P, Rr, t, ang, lim, matrix, dmg, update, fe, me,
                   beta, dt, prev, wd, wR, wth)
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
def glue_kernel(x, R, a_node, a_r0, ca, cb, area, n0, Kn, Ks, sig, tau, GIc, GIIc, mu,
                breakable, dmax, dd, slip, F, Mo, update, beta, dt, prev, alive):
    nc = ca.shape[0]
    pa = np.empty(3)
    pb = np.empty(3)
    arma = np.empty(3)
    armb = np.empty(3)
    n = np.empty(3)
    ds = np.empty(3)
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
        s2 = 0.0
        for k in range(3):
            ds[k] = pb[k] - pa[k] - dn * n[k]
            s2 += ds[k] * ds[k]
        s = math.sqrt(s2)
        dnp = dn if dn > 0.0 else 0.0
        kn = Kn[i]
        ks = Ks[i]
        d = 0.0
        if breakable[i]:
            dn0 = sig[i] / kn
            ds0 = tau[i] / ks
            dm = math.sqrt(dnp * dnp + s2)
            if dnp > 1e-12:
                mix = s / dnp
                dm0 = dn0 * ds0 * math.sqrt((1.0 + mix * mix)
                                            / (ds0 * ds0 + (mix * dn0) ** 2))
            else:
                dm0 = ds0
            Gs = ks * s2
            Gn = kn * dnp * dnp
            B = Gs / (Gs + Gn) if Gs + Gn > 1e-30 else 0.0
            Gc = GIc[i] + (GIIc[i] - GIc[i]) * B * B
            keff = (Gn + Gs) / (dm * dm) if dm > 1e-12 else ks
            dmf = 2.0 * Gc / (keff * (dm0 if dm0 > 1e-12 else 1e-12))
            if dmf < 1.05 * dm0:
                dmf = 1.05 * dm0
            dmx = dmax[i] if dmax[i] > dm else dm
            dnew = dmf * (dmx - dm0) / ((dmx if dmx > 1e-12 else 1e-12) * (dmf - dm0))
            if dnew < 0.0:
                dnew = 0.0
            elif dnew > 1.0:
                dnew = 1.0
            d = dd[i] if dd[i] > dnew else dnew
            if update:
                dmax[i] = dmx
                dd[i] = d
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
            f[k] = (tn * n[k] + (1.0 - d) * ks * ds[k] + d * tf[k]) * area[i]
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
def platform_kernel(x, R, a_node, a_r0, att, table_y, k, mu, anchor, touch, F, Mo, update,
                    vel, omg, cdamp):
    p = np.empty(3)
    arm = np.empty(3)
    f = np.empty(3)
    for i in range(att.shape[0]):
        nd = _apos(x, R, a_node, a_r0, att[i], p, arm)
        pen = table_y - p[1]
        if pen <= 0.0:
            if update:
                touch[i] = False
                for kk in range(3):
                    anchor[i, kk] = p[kk]
            continue
        # contact damping on the approach speed (never pulls the point down)
        vy = vel[nd, 1] + omg[nd, 2] * arm[0] - omg[nd, 0] * arm[2]
        fn = k * pen - cdamp[i] * vy
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
def plate_kernel(x, R, a_node, a_r0, att, P, h, k, mu, anchor, touch, F, Mo, update,
                 vel, omg, cdamp):
    p = np.empty(3)
    arm = np.empty(3)
    loc = np.empty(3)
    nl = np.empty(3)
    ftl = np.empty(3)
    f = np.empty(3)
    total_down = 0.0
    for i in range(att.shape[0]):
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
        fn = k * best - cdamp[i] * vr
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
