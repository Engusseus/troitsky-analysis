"""A member's contact box ends where the member ends. Its inner elements have open ends
(no face between two elements of one member) that reach on past them, so a point at a
joint between elements is pushed out through the member's side; that reach must not pass
the member's own end. Found on Bridge_v12: its floor beam at the centre node is 200 mm
wide and 162 mm long, the reach was the section's radius (101 mm), and the node blocks,
gusset plates and stubs glued beyond its ends stood inside its box at the start (1232
points allowed up to 15 mm of penetration, 26 once fixed): pushed apart as soon as
anything moved, the standard crush diverged while the bridge settled."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bridgesim.cad.synthetic import box, simple_beam_bridge
from bridgesim.schema import Material
from bridgesim.sim import contact
from bridgesim.sim.idealize import Overrides, build_model
from bridgesim.sim.materials import SimParams

ROOT = Path(__file__).resolve().parents[1]
MAT = Material.from_yaml(ROOT / "materials" / "popsicle_birch.yaml")
Y0 = 173.1  # the synthetic bridge's deck top


@pytest.fixture(scope="module")
def wide_beam():
    """The synthetic bridge with a cross beam on its deck, 200 mm wide and 80 mm long
    (a member along z), and a lug on the deck 6 mm beyond the beam's end, within its
    section's outline."""
    bodies = simple_beam_bridge() + [
        box("Cross Beam (10-ply UD)", (-100, Y0, -40), (100, Y0 + 20, 40)),
        box("Lug (4-ply UD)", (-30, Y0, 46), (30, Y0 + 8, 70))]
    ov = Overrides.from_dict({"bodies": {"Cross Beam*": {"kind": "bar", "axis": "z"}}})
    m = build_model(bodies, MAT, SimParams.load(), ov, seed=None)
    sc = contact.setup(m, m.X, m.mass, 3e-6)
    return m, sc


def _body(m, name):
    return [b.name for b in m.bodies].index(name)


def test_an_open_end_reaches_no_farther_than_the_member_goes_on(wide_beam):
    m, sc = wide_beam
    cb = _body(m, "Cross Beam (10-ply UD)")
    els = np.where((sc.el_body == cb) & (sc.el_kind == 0))[0]
    assert len(els) >= 2
    z = m.X[:, 2]
    r = sc.el_r[els]
    assert r.min() > 95.0  # a 200 x 20 mm section: a radius longer than the member
    for e in els:
        for side in (0, 1):
            nd = sc.el_nodes[e, side]
            other = sc.el_nodes[e, 1 - side]
            reach = sc.el_cap[e, side]
            # going on from this end, away from the element: the member's end that side
            end = 40.0 if z[nd] > z[other] else -40.0
            left = abs(end - z[nd])
            if left < 1e-6:
                assert reach == 0.0  # the member's end: a face
            else:
                assert 0.0 < reach <= left + 1e-6
    # a slender member keeps its section's radius at its inner ends (as before)
    st_ = _body(m, "Stringer Near (3-ply UD)")
    se = np.where((sc.el_body == st_) & (sc.el_kind == 0))[0]
    inner = [e for e in se if sc.el_cap[e].min() > 0.0]
    assert len(inner) >= 2
    mid = inner[len(inner) // 2]
    assert sc.el_cap[mid] == pytest.approx(sc.el_r[mid])
    ends = [e for e in se if sc.el_cap[e].min() == 0.0]
    assert len(ends) == 2


def test_a_part_beyond_a_wide_members_end_is_outside_it(wide_beam):
    m, sc = wide_beam
    cb, lug = _body(m, "Cross Beam (10-ply UD)"), _body(m, "Lug (4-ply UD)")
    pts = np.where(sc.s_body == lug)[0]
    assert len(pts) >= 8
    # no point of the lug starts inside the cross beam (an allowance), and none is close
    # enough to be kept off it as glued (they share no patch)
    assert not (sc.sl_body[pts] == cb).any()
    # the signed distance of each to each of the beam's elements: 6 mm or more outside
    els = np.where((sc.el_body == cb) & (sc.el_kind == 0))[0]
    R0 = np.repeat(np.eye(3)[None], m.n_nodes, axis=0)
    pi = np.repeat(pts, len(els)).astype(np.int64)
    pe = np.tile(els, len(pts)).astype(np.int64)
    gaps = np.zeros(len(pi))
    contact.pair_gaps(m.X, R0, m.a_node, m.a_r0, sc.s_att, sc.el_nodes, sc.el_kind,
                      sc.el_r, sc.el_h, sc.el_c, sc.el_q0, sc.el_cap, pi, pe, len(pi), gaps)
    assert gaps.min() >= 6.0 - 1e-6
