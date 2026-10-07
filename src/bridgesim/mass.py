"""Mass estimate.

    m = rho (sum_i A_i L_i + L_deck W_deck t_deck) (1 + g)

where L_i is the centre-line length of member i, the deck is a solid plate of the clear
width, and g is the glue mass fraction. Centre-line lengths double-count the wood inside
joints, which slightly over-estimates mass (conservative for the §8.8 limit).
Deviation from the original spec: the glue fraction is applied to the deck as well,
because the deck is also glued sticks.

Stick count = total wood volume / volume of one stick (115 x 10 x 2 = 2300 mm^3).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from bridgesim.schema import Bridge, Material
from bridgesim.units import volume_mm3_to_mass_kg


@dataclass
class MassReport:
    total_kg: float
    wood_kg: float
    glue_kg: float
    deck_kg: float
    members_kg: float
    wood_volume_mm3: float
    stick_count: int
    by_group_kg: dict[str, float] = field(default_factory=dict)


def bridge_mass(bridge: Bridge, material: Material) -> MassReport:
    rho = material.density_kg_m3.value
    g = material.glue.mass_fraction.value
    nodes = bridge.node_map()
    secs = bridge.section_map()
    by_group: dict[str, float] = defaultdict(float)
    vol_members = 0.0
    for m in bridge.members:
        v = secs[m.section].props(material.stick).A_mm2 * bridge.member_length_mm(m, nodes)
        vol_members += v
        by_group[m.group] += volume_mm3_to_mass_kg(v, rho)
    d = bridge.deck
    vol_deck = d.length_mm * d.clear_width_mm * d.thickness_mm
    vol = vol_members + vol_deck
    wood = volume_mm3_to_mass_kg(vol, rho)
    by_group["deck"] = volume_mm3_to_mass_kg(vol_deck, rho)
    return MassReport(
        total_kg=wood * (1.0 + g),
        wood_kg=wood,
        glue_kg=wood * g,
        deck_kg=volume_mm3_to_mass_kg(vol_deck, rho),
        members_kg=volume_mm3_to_mass_kg(vol_members, rho),
        wood_volume_mm3=vol,
        stick_count=math.ceil(vol / material.stick.volume_mm3),
        by_group_kg=dict(by_group),
    )
