"""Arrays describing the structural model the explicit solver integrates.

Units are mm, N, s and tonnes (1 t = 1000 kg), so that F = m a holds with mm/s^2.
Every node has 3 translations and 3 rotations; nodal orientation is a rotation matrix
updated by the exponential map. Members of a body attach to other bodies, contact
surfaces and the crusher through *attachments*: a node plus a constant offset in the
node's own frame (a rigid arm), so eccentric joints are modelled at their real position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TONNE_PER_KG = 1e-3
G_MM_S2 = 9806.65


@dataclass
class BodyInfo:
    name: str
    kind: str  # "bar", "shell", "block", "cable"
    layup: str
    layup_source: str
    mass_kg: float
    volume_mm3: float
    nodes: list[int] = field(default_factory=list)
    beams: list[int] = field(default_factory=list)
    shells: list[int] = field(default_factory=list)
    cables: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    E_eff_MPa: float = 0.0
    f_t_eff_MPa: float = 0.0
    f_c_eff_MPa: float = 0.0
    thickness_mm: float = 0.0
    section_mm: tuple[float, float] = (0.0, 0.0)
    length_mm: float = 0.0
    centroid_mm: tuple = (0.0, 0.0, 0.0)
    bbox_mm: tuple | None = None
    axis: tuple | None = None


@dataclass
class Patch:
    """A glued interface between two bodies (overlapping coplanar faces)."""

    body_a: int
    body_b: int
    area_mm2: float
    centroid: np.ndarray
    normal: np.ndarray  # from A towards B
    springs: list[int] = field(default_factory=list)
    kind: str = "glue"  # "glue", "tie" (floss knot), "bonded" (bodies overlap in CAD)
    polygon: np.ndarray | None = None  # (k, 3) outline


@dataclass
class Model:
    # nodes
    X: np.ndarray  # (n, 3) initial positions
    mass: np.ndarray  # (n,)
    rot_inertia: np.ndarray  # (n,) scalar rotational inertia
    node_body: np.ndarray  # (n,)
    # beams (fibre-section corotational elements)
    b_i: np.ndarray
    b_j: np.ndarray
    b_L0: np.ndarray
    b_E0: np.ndarray  # (nb, 3, 3) initial frame, columns e1 e2 e3
    b_fy: np.ndarray  # (nb, nf) fibre coordinates along e2
    b_fz: np.ndarray  # (nb, nf) along e3
    b_fA: np.ndarray  # (nb, nf) fibre areas
    b_E: np.ndarray
    b_ft: np.ndarray
    b_fc: np.ndarray
    b_Gf: np.ndarray
    b_GJ: np.ndarray
    b_A: np.ndarray
    b_fv: np.ndarray
    b_Wt: np.ndarray  # torsional section modulus (mm^3)
    b_body: np.ndarray
    # shells
    s_nodes: np.ndarray  # (ns, 3)
    s_K: np.ndarray  # (ns, 18, 18) local stiffness
    s_E0: np.ndarray  # (ns, 3, 3)
    s_P: np.ndarray  # (ns, 3, 3) initial local coordinates of the 3 nodes
    s_R: np.ndarray  # (ns, 6, 18) strain recovery
    s_t: np.ndarray  # (ns,) thickness
    s_ang: np.ndarray  # (ns, 2) distinct ply angles in element axes (deg), nan if absent
    s_lim: np.ndarray  # (ns, 5) e1t, e1c, e2t, e2c, g12 failure strains
    s_matrix: np.ndarray  # (ns,) bool: across-grain limits count
    s_body: np.ndarray
    # attachments (rigid arms)
    a_node: np.ndarray
    a_r0: np.ndarray  # (na, 3)
    # cohesive glue springs
    c_a: np.ndarray
    c_b: np.ndarray
    c_area: np.ndarray
    c_n0: np.ndarray  # (nc, 3)
    c_Kn: np.ndarray
    c_Ks: np.ndarray
    c_sig: np.ndarray
    c_tau: np.ndarray
    c_GIc: np.ndarray
    c_GIIc: np.ndarray
    c_mu: np.ndarray
    c_patch: np.ndarray
    c_breakable: np.ndarray  # False for ties / bonded overlaps
    # cables (floss)
    k_a: np.ndarray
    k_b: np.ndarray
    k_L0: np.ndarray
    k_EA: np.ndarray
    k_Fb: np.ndarray
    k_body: np.ndarray
    # contact points with the base platform and with the loading plate
    p_att: np.ndarray
    p_support: np.ndarray  # bool: touching the platform at the start
    q_att: np.ndarray
    # crusher
    plate_node: int
    plate_half: np.ndarray  # half sizes of the plate stack (x, y, z) in its frame
    jack_xz: np.ndarray  # jack axis position (x, z)
    jack_radius: float
    frame_x: tuple[float, float]  # inner faces of the press uprights
    # bookkeeping
    bodies: list[BodyInfo]
    patches: list[Patch]
    table_y: float = 0.0
    deck_top_y: float = 0.0
    mid_x: float = 0.0
    deck_z: float = 0.0
    bridge_mass_kg: float = 0.0
    added_mass_kg: float = 0.0
    params: dict = field(default_factory=dict)
    questions: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # render skin: vertex -> up to 3 (attachment) weights
    skin: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.X)
