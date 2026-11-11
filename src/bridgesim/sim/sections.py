"""A member crushed across its grain gets shallower (round 28).

Wood pressed across its grain at a joint crushes: the glue kernel keeps it as a permanent
set of the glue line (``st.c_crush``, the crushed depth at each glue point). The member
whose side was crushed is shallower there by that depth, and its centroid moves away
from the crushed face: it loses bending stiffness and the axial force gets a lever arm
(a kink). The fibres are compacted, not cut, so the axial area stays.

Geometric feedback only, no new constant: for each beam element with glue points on one
of its four side faces, the crushed depth c of its face is the crushed volume of those
points over the face's area in the element, c = sum a_i c_i s_i / (L x the face's width)
(a_i a point's area, s_i the share of its joint's crush on this member: the weaker side
crushes, both alike half each, only a member's side grain counts, never an end face
bearing on end grain or the other part: ``Model.params["patch_crush_share"]``). Its
section is squeezed on that side, the fibre coordinates across it mapped linearly from
[-d/2, d/2] to [-d/2 + c-, d/2 - c+] (fibre areas unchanged): the second moment falls by
((d - c) / d)^2 and the centroid moves by (c- - c+) / 2. The kernels read the fibre
coordinates each step, so the bending stiffness, the Timoshenko split and the axial lever
arm follow; the crush runs updates them every ``EVERY`` steps, holds included, only for
elements whose crushed depth changed by more than ``TOL_MM``. Sections only get smaller,
so the stable time step holds. A face crushed through more than ``MAX_SHARE`` of the
section is squeezed no further (crushed wood is densified, glue_kernel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVERY = 50  # steps between updates (~0.1-0.5 ms of simulated time)
TOL_MM = 0.002  # an element's crushed depth changed by more than this: its section is redone
MAX_SHARE = 0.9  # a section squeezed to at most this share of its depth


@dataclass
class FaceMap:
    """Glue points on members' side faces: for each, the beam element and its face
    (slot = element x 4 + face, faces +e2, -e2, +e3, -e3), the weight (area x share /
    the face's area in the element, 1/mm), and the pristine fibre coordinates."""

    spring: np.ndarray
    slot: np.ndarray
    weight: np.ndarray
    fy0: np.ndarray
    fz0: np.ndarray
    sec: np.ndarray  # (nb, 3): width along e2, depth along e3, centre along e2
    applied: np.ndarray  # (nb, 4): the crushed depths the sections hold now

    @property
    def n_points(self) -> int:
        return len(self.spring)


def face_map(m) -> FaceMap | None:
    """The glue-point -> (element, face) map of a model, built once (cached on the model;
    a rebuilt run starts again from the pristine sections). None when nothing can crush a
    member's side."""
    fm = getattr(m, "_face_map", None)
    if fm is not None and len(fm.sec) == len(m.b_i):
        m.b_fy[:] = fm.fy0
        m.b_fz[:] = fm.fz0
        fm.applied[:] = 0.0
        return fm if fm.n_points else None
    share = m.params.get("patch_crush_share")
    sec = m.params.get("b_section_mm")
    nb = len(m.b_i)
    if share is None or sec is None or not nb or not len(m.c_a) or len(sec) != nb:
        return None
    share = np.asarray(share, float)
    sec = np.asarray(sec, float)
    # each node's beam elements, by body
    adj: dict[int, list[int]] = {}
    for e in range(nb):
        adj.setdefault(int(m.b_i[e]), []).append(e)
        adj.setdefault(int(m.b_j[e]), []).append(e)
    pa = np.array([p.body_a for p in m.patches] or [0], int)
    pb = np.array([p.body_b for p in m.patches] or [0], int)
    rows, slots, wts = [], [], []
    for side, att in ((0, m.c_a), (1, m.c_b)):
        s_i = share[m.c_patch, side] if len(share) else np.zeros(len(m.c_a))
        body = (pa if side == 0 else pb)[m.c_patch]
        for i in np.where(s_i > 0)[0]:
            nd = int(m.a_node[att[i]])
            p = m.X[nd] + m.a_r0[att[i]]
            # the element whose length the point lies along (both, half each, for a point
            # on the node between two: a glue grid often has a row on a node)
            cand = []
            for e in adj.get(nd, ()):
                if int(m.b_body[e]) != int(body[i]):
                    continue
                t = float((p - m.X[m.b_i[e]]) @ m.b_E0[e][:, 0]) / m.b_L0[e]
                cand.append((max(-t, t - 1.0, 0.0), e))  # how far outside it, along it
            if not cand:
                continue
            d0 = min(d for d, _e in cand)
            els = [e for d, e in cand if d <= d0 + 1e-6]
            n_out = m.c_n0[i] if side == 0 else -m.c_n0[i]
            for e in els:
                nl = m.b_E0[e].T @ n_out
                if abs(nl[0]) > 0.5:  # an end face: bearing on end grain, no squeeze
                    continue
                k = 1 if abs(nl[1]) >= abs(nl[2]) else 2
                face = 2 * (k - 1) + (0 if nl[k] > 0 else 1)
                width = sec[e, 1] if k == 1 else sec[e, 0]  # the face spans the other side
                area = m.b_L0[e] * max(width, 1e-9)
                rows.append(i)
                slots.append(4 * e + face)
                wts.append(m.c_area[i] * s_i[i] / area / len(els))
    fm = FaceMap(np.array(rows, int), np.array(slots, int), np.array(wts, float),
                 m.b_fy.copy(), m.b_fz.copy(), sec, np.zeros((nb, 4)))
    m._face_map = fm
    return fm if fm.n_points else None


def crushed_depths(m, st, fm: FaceMap) -> np.ndarray:
    """(nb, 4): each element's crushed depth on each side face (mm)."""
    c = np.maximum(-st.c_crush[fm.spring], 0.0) * fm.weight
    return np.bincount(fm.slot, weights=c, minlength=4 * len(m.b_i)).reshape(-1, 4)


def squeeze(m, st, fm: FaceMap | None, tol: float = TOL_MM) -> int:
    """Squeeze the sections of the elements whose crushed depth changed by more than
    ``tol`` since they were last set; returns how many were redone (0: the fibre
    coordinates are untouched)."""
    if fm is None or not len(st.c_crush):
        return 0
    c = crushed_depths(m, st, fm)
    redo = np.where(np.abs(c - fm.applied).max(1) > tol)[0]
    if not len(redo):
        return 0
    cc = c[redo]
    W, D, Yc = fm.sec[redo, 0], fm.sec[redo, 1], fm.sec[redo, 2]
    for axis, (cp, cm, d, mid, arr0, arr) in enumerate((
            (cc[:, 0], cc[:, 1], W, Yc, fm.fy0, m.b_fy),
            (cc[:, 2], cc[:, 3], D, np.zeros(len(redo)), fm.fz0, m.b_fz))):
        del axis
        tot = cp + cm
        sc = np.where(tot > MAX_SHARE * d, MAX_SHARE * d / np.maximum(tot, 1e-12), 1.0)
        cp, cm = cp * sc, cm * sc
        k = (d - cp - cm) / np.maximum(d, 1e-12)
        y0 = arr0[redo]
        sq = (mid - 0.5 * d + cm)[:, None] + (y0 - mid[:, None] + 0.5 * d[:, None]) \
            * k[:, None]
        arr[redo] = np.where((tot > 0.0)[:, None], sq, y0)  # an axis not crushed: as built
    fm.applied[redo] = c[redo]
    return len(redo)
