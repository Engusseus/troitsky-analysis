"""Build a Pynite ``FEModel3D`` from a :class:`~bridgesim.schema.Bridge`.

* Every member is a 3D Euler-Bernoulli frame element (Pynite ``add_member``) with the
  rectangular section properties from ``sections.py``, the bridge material's E and G (or
  the section's own, if it sets them) and member rotation 0, so the section depth d lies
  along Pynite's local y axis and ``Iz = b d^3/12`` governs bending in the member's
  primary plane (verified in ``tests/test_orientation.py``).
* ``joint_fixity = rigid``: all joints rigid (glued joints are closer to rigid than
  pinned). ``pinned``: web members (diagonals, verticals) are released for in-plane
  bending (Rz) at both ends, and bracing members for both bending axes, while chords,
  floor beams, struts and piers stay continuous. Releasing everything would leave the
  end portals as a racking mechanism.
* Supports: ``pinned`` nodes restrain DX, DY, DZ, ``roller`` nodes DY, DZ. Rotations are
  free. The bridge is not anchored (rulebook §8.3): the analysis reports any support
  that would need to pull down (uplift) or resist large horizontal thrust.
* Every member also gets one interior node (its midpoint, or a nearby point that lies on
  no other member), which Pynite uses to split it into two elements. Static results are
  unchanged (cubic frame elements are exact for end loads), but the global buckling
  analysis gets two elements per member: with one, F_cr is about 2 % too high.
* ``tension_only`` members are passed to Pynite as tension-only, so its iterative analysis
  (``FEModel3D.analyze``) leaves out any that would be in compression, from the stiffness
  and from the geometric stiffness alike. They are not split: an interior node of a member
  that drops out would be left with no stiffness. Being in tension whenever they are
  active, they never need the finer buckling mesh.
* A deck ``diaphragm`` adds two pin-ended equivalent diagonals per deck panel, sized so
  each panel has the plate's in-plane shear stiffness (:func:`diaphragm_bars`). They
  are not bridge members: they carry no checks and appear in no results table.
"""

from __future__ import annotations

import numpy as np
from Pynite import FEModel3D

from bridgesim.loads import plate_nodal_loads
from bridgesim.schema import Bridge, Material

LOAD_CASE = "Plate"
COMBO = "Pref"

#: Releases applied by ``joint_fixity = pinned`` (by member group).
PINNED_RELEASES: dict[str, tuple[str, ...]] = {
    "diagonal": ("Rzi", "Rzj"),
    "vertical": ("Rzi", "Rzj"),
    "top_brace": ("Ryi", "Rzi", "Ryj", "Rzj"),
    "bottom_brace": ("Ryi", "Rzi", "Ryj", "Rzj"),
    "pier_brace": ("Ryi", "Rzi", "Ryj", "Rzj"),
}

_MATERIAL = "wood"

#: Prefix of the interior nodes added to split members (not part of the bridge). Extra "~"
#: are prepended if a bridge node already uses the prefix, so names never collide.
INTERIOR_PREFIX = "~mid "
#: Prefix of the deck diaphragm's equivalent diagonals (members and sections), lengthened
#: with "~" like INTERIOR_PREFIX if a bridge member or section id already uses it.
DIAPHRAGM_PREFIX = "~deck "
#: Above this many members the split is skipped: Pynite's member subdivision scans every
#: node for every member, and the extra nodes would make very large uploads slow.
REFINE_MAX_MEMBERS = 1500
_SPLIT_AT = tuple(sorted((0.5 + k * d for k in range(13) for d in (0.025, -0.025)),
                         key=lambda t: abs(t - 0.5)))  # 0.5, 0.475, 0.525, ... 0.2, 0.8


def _interior_points(bridge: Bridge) -> dict[str, tuple[float, float, float]]:
    """A point inside each member that lies on no other member (crossing X-braces meet at
    their midpoints, and a node there would glue them together) and on no deck diaphragm
    diagonal, unless the diagonal lies along the member (joining the two is then harmless:
    Pynite keeps the diagonal's end releases at its ends)."""
    nodes = bridge.node_map()
    segs = [(m.i, m.j) for m in bridge.members] + bridge.diaphragm_diagonals()
    A = np.array([nodes[i].xyz for i, _ in segs], dtype=float).reshape(-1, 3)
    B = np.array([nodes[j].xyz for _, j in segs], dtype=float).reshape(-1, 3)
    D = B - A
    L = np.linalg.norm(D, axis=1)
    U = D / L[:, None]
    n_mem = len(bridge.members)
    out = {}
    for k, mem in enumerate(bridge.members):
        others = np.arange(len(L)) != k
        along = np.linalg.norm(np.cross(U, U[k]), axis=1) < 1e-9
        along &= np.linalg.norm(np.cross(A - A[k], U[k]), axis=1) < 1e-6
        others[n_mem:] &= ~along[n_mem:]
        for t in _SPLIT_AT:
            pt = A[k] + t * D[k]
            d = pt - A[others]
            s = np.einsum("ij,ij->i", d, U[others])
            perp = np.linalg.norm(d - s[:, None] * U[others], axis=1)
            if not np.any((s > -1e-6) & (s < L[others] + 1e-6) & (perp < 1e-6)):
                out[mem.id] = (float(pt[0]), float(pt[1]), float(pt[2]))
                break
    return out


def diaphragm_bars(bridge: Bridge, E_MPa: float) -> list[tuple[str, str, float]]:
    """Equivalent diagonals (i node, j node, area A_d in mm^2) of the deck diaphragm.

    A panel of length a along the bridge, width b and thickness t, with in-plane shear
    modulus G, resists a sideways movement Delta of one end relative to the other with
    shear strain Delta / a over the b x t cross-section: k = G t b / a. Two crossing
    pin-ended diagonals of length L_d (modulus E) each stretch or shorten by Delta b / L_d,
    and the sideways share of their force is b / L_d, so k = 2 E A_d b^2 / L_d^3. Equal
    stiffness gives

        A_d = G t L_d^3 / (2 E a b)

    (the usual equivalent-truss model of a shear panel). a and b are the means of the
    panel's two edges along and across the bridge. As a side effect, the diagonals give the
    deck an axial stiffness of about G (a / b)^2 along the bridge, usually well below the
    real plate's, so the deck takes a little of the bottom chords' tension.
    """
    dia = bridge.deck.diaphragm
    if dia is None:
        return []
    t = dia.thickness_mm if dia.thickness_mm is not None else bridge.deck.thickness_mm
    P = {n.id: np.array(n.xyz, dtype=float) for n in bridge.nodes}
    out = []
    for n0, n1, f0, f1 in bridge.diaphragm_panels():
        a = 0.5 * (np.linalg.norm(P[n1] - P[n0]) + np.linalg.norm(P[f1] - P[f0]))
        b = 0.5 * (np.linalg.norm(P[f0] - P[n0]) + np.linalg.norm(P[f1] - P[n1]))
        for i, j in ((n0, f1), (f0, n1)):
            L_d = float(np.linalg.norm(P[j] - P[i]))
            out.append((i, j, dia.G_MPa * t * L_d**3 / (2.0 * E_MPa * a * b)))
    return out


def member_releases(bridge: Bridge, member) -> set[str]:
    rel = set(member.releases)
    if bridge.joint_fixity == "pinned":
        rel |= set(PINNED_RELEASES.get(member.group, ()))
    return rel


def build_model(bridge: Bridge, material: Material, P_N: float | None = None) -> FEModel3D:
    """Return an un-analysed Pynite model loaded with the crusher plate at ``P_N``."""
    m = FEModel3D()
    for n in bridge.nodes:
        m.add_node(n.id, n.x_mm, n.y_mm, n.z_mm)

    E = material.E_MPa.value
    m.add_material(_MATERIAL, E=E, G=material.G_MPa.value, nu=material.nu.value, rho=0.0)

    mat_of: dict[str, str] = {}  # section id -> Pynite material name
    for s in bridge.sections:
        pr = s.props(material.stick)
        m.add_section(s.id, A=pr.A_mm2, Iy=pr.Iy_mm4, Iz=pr.Iz_mm4, J=pr.J_mm4)
        mat_of[s.id] = _MATERIAL
        if s.E_MPa is not None or s.G_MPa is not None:  # the section's own stiffness
            mat_of[s.id] = f"{_MATERIAL} {s.id}"  # section ids are unique, so names are too
            m.add_material(mat_of[s.id], E=s.value("E_MPa", material),
                           G=s.value("G_MPa", material), nu=material.nu.value, rho=0.0)

    for mem in bridge.members:
        m.add_member(mem.id, mem.i, mem.j, mat_of[mem.section], mem.section,
                     tension_only=mem.tension_only)
        rel = member_releases(bridge, mem)
        if rel:
            m.def_releases(mem.id, **{r: True for r in rel})
    taken = {x.id for x in (*bridge.members, *bridge.sections)}
    prefix = DIAPHRAGM_PREFIX
    while any(x.startswith(prefix) for x in taken):
        prefix = "~" + prefix
    for k, (i, j, A) in enumerate(diaphragm_bars(bridge, E)):
        name = f"{prefix}{k}"
        # A square bar of area A: its bending stiffness is released at both ends anyway.
        m.add_section(name, A=A, Iy=A * A / 12.0, Iz=A * A / 12.0, J=0.141 * A * A)
        m.add_member(name, i, j, _MATERIAL, name)
        m.def_releases(name, Rxi=True, Ryi=True, Rzi=True, Ryj=True, Rzj=True)
    split: dict[str, tuple[float, float, float]] = {}
    if len(bridge.members) <= REFINE_MAX_MEMBERS:
        prefix = INTERIOR_PREFIX
        while any(n.id.startswith(prefix) for n in bridge.nodes):
            prefix = "~" + prefix
        tension_only = {mem.id for mem in bridge.members if mem.tension_only}
        split = {mid: pt for mid, pt in _interior_points(bridge).items()
                 if mid not in tension_only}
        for mid, (x, y, z) in split.items():
            m.add_node(prefix + mid, x, y, z)
    #: Members left as one element (model too large, or crossed at every split point).
    #: Tension-only members are never split (see the module docstring) and not listed.
    m.unsplit_members = [mem.id for mem in bridge.members
                         if mem.id not in split and not mem.tension_only]

    for nid in bridge.supports.pinned:
        m.def_support(nid, support_DX=True, support_DY=True, support_DZ=True)
    for nid in bridge.supports.roller:
        m.def_support(nid, support_DY=True, support_DZ=True)
    for nid, dofs in bridge.supports.extra_restraints.items():
        node = m.nodes[nid]
        for dof in dofs:
            setattr(node, f"support_{dof}", True)

    for nid, F in plate_nodal_loads(bridge, P_N).items():
        m.add_node_load(nid, "FY", -F, case=LOAD_CASE)
    m.add_load_combo(COMBO, {LOAD_CASE: 1.0})
    return m


def splittable_count(bridge: Bridge) -> int:
    """Members that the buckling mesh would split in two (all but tension-only ones)."""
    return sum(not mem.tension_only for mem in bridge.members)
