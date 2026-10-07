"""Build a Pynite ``FEModel3D`` from a :class:`~bridgesim.schema.Bridge`.

* Every member is a 3D Euler-Bernoulli frame element (Pynite ``add_member``) with the
  rectangular section properties from ``sections.py`` and member rotation 0, so the
  section depth d lies along Pynite's local y axis and ``Iz = b d^3/12`` governs bending
  in the member's primary plane (verified in ``tests/test_orientation.py``).
* ``joint_fixity = rigid``: all joints rigid (glued joints are closer to rigid than
  pinned). ``pinned``: web members (diagonals, verticals) are released for in-plane
  bending (Rz) at both ends, and bracing members for both bending axes, while chords,
  floor beams, struts and piers stay continuous. Releasing everything would leave the
  end portals as a racking mechanism.
* Supports: ``pinned`` nodes restrain DX, DY, DZ, ``roller`` nodes DY, DZ. Rotations are
  free. The bridge is not anchored (rulebook §8.3): the analysis reports any support
  that would need to pull down (uplift) or resist large horizontal thrust.
"""

from __future__ import annotations

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

    for s in bridge.sections:
        pr = s.props(material.stick)
        m.add_section(s.id, A=pr.A_mm2, Iy=pr.Iy_mm4, Iz=pr.Iz_mm4, J=pr.J_mm4)

    for mem in bridge.members:
        m.add_member(mem.id, mem.i, mem.j, _MATERIAL, mem.section)
        rel = member_releases(bridge, mem)
        if rel:
            m.def_releases(mem.id, **{r: True for r in rel})

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
