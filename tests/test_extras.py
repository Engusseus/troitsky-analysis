"""Extra wood (gussets, plates) in the mass, and per-member glued areas in the joint check."""

import pytest
from pydantic import ValidationError

from bridgesim.analysis import analyze
from bridgesim.checks import capacities
from bridgesim.generators import generate_warren
from bridgesim.mass import bridge_mass
from bridgesim.report import model_assumptions, to_markdown
from bridgesim.rules import evaluate
from bridgesim.schema import Bridge
from bridgesim.units import volume_mm3_to_mass_kg


def _with(**changes) -> Bridge:
    data = generate_warren().model_dump(mode="json")
    data.update(changes)
    return Bridge.model_validate(data)


def test_extra_wood_adds_mass_with_glue_and_sticks(material) -> None:
    """V = 14 400 x 8 + 2 300 = 117 500 mm^3; m = rho V = 650e-9 x 117 500 = 0.0764 kg of wood,
    0.0840 kg with 10 % glue; 117 500 / 2 300 = 51 more sticks."""
    base = bridge_mass(generate_warren(), material)
    b = _with(extra_wood=[{"id": "gusset", "volume_mm3": 14_400, "count": 8},
                          {"id": "block", "volume_mm3": 2_300}])
    m = bridge_mass(b, material)
    wood = volume_mm3_to_mass_kg(14_400 * 8 + 2_300, material.density_kg_m3.value)
    assert m.extra_kg == pytest.approx(wood)
    assert m.by_group_kg["extra_wood"] == pytest.approx(wood)
    glue = material.glue.mass_fraction.value
    assert m.total_kg - base.total_kg == pytest.approx(wood * (1 + glue))
    assert m.stick_count - base.stick_count == pytest.approx((14_400 * 8 + 2_300) / 2_300, abs=1)
    assert m.members_kg == pytest.approx(base.members_kg)  # no stiffness or member volume


def test_mass_rule_counts_extra_wood(material) -> None:
    """10 x 1e6 mm^3 x 650 kg/m^3 x 1.1 = 7.15 kg on top of 2.11 kg: over the 6 kg limit."""
    b = _with(extra_wood=[{"id": "plates", "volume_mm3": 1e6, "count": 10}])  # ~7 kg
    r = next(r for r in evaluate(b, material).results if r.key == "mass")
    assert r.passed is False and r.measured == pytest.approx(bridge_mass(b, material).total_kg,
                                                             abs=0.005)


def test_extra_wood_round_trips_and_is_omitted_when_empty() -> None:
    assert "extra_wood" not in generate_warren().to_yaml()
    b = _with(extra_wood=[{"id": "gusset", "volume_mm3": 14_400, "count": 8}])
    assert Bridge.from_yaml_str(b.to_yaml()).extra_wood == b.extra_wood


@pytest.mark.parametrize("extra", [
    [{"id": "g", "volume_mm3": -1}],
    [{"id": "g", "volume_mm3": 1e12}],
    [{"id": "g", "volume_mm3": 10, "count": 0}],
    [{"id": "g", "volume_mm3": 10}, {"id": "g", "volume_mm3": 20}],
    [{"id": "  ", "volume_mm3": 10}],
])
def test_invalid_extra_wood_is_rejected(extra) -> None:
    with pytest.raises(ValidationError):
        _with(extra_wood=extra)


def test_reports_show_extra_wood(material) -> None:
    r = analyze(_with(extra_wood=[{"id": "gusset", "volume_mm3": 14_400, "count": 8}]), material,
                include_buckling=False)
    assert "of extra wood" in to_markdown(r)


# --------------------------------------------------------------------------- glued area


def _member(bridge: Bridge, group: str):
    return next(m for m in bridge.members if m.group == group)


def test_glue_area_replaces_the_placeholder_joint_area(material) -> None:
    """F_R = tau_g A = 2 MPa x 3600 mm^2 = 7200 N, instead of tau_g x overlap x d x faces."""
    b = generate_warren()
    diag = _member(b, "diagonal")
    sec = b.section_map()[diag.section]
    default = capacities(sec, material, 200.0, 1.0, "diagonal")
    own = capacities(sec, material, 200.0, 1.0, "diagonal", glue_area_mm2=3_600)
    assert own.F_joint_R == pytest.approx(material.glue.tau_g_MPa.value * 3_600)
    assert own.F_joint_R != pytest.approx(default.F_joint_R)


def test_larger_glue_area_raises_a_joint_governed_failure_load(material) -> None:
    """The default design fails in the diagonal joints (2 x 20 x 10 x 2 = 800 N each). With
    100x that area on every diagonal the joints no longer govern, so F_u rises to the next
    limit (floor-beam bending)."""
    b = generate_warren()
    r0 = analyze(b, material, include_buckling=False)
    assert r0.governing_mode == "joint" and r0.member(r0.governing_member).group == "diagonal"
    data = b.model_dump(mode="json")
    for m in data["members"]:
        if m["group"] == "diagonal":
            m["glue_area_mm2"] = 40_000.0
    r1 = analyze(Bridge.model_validate(data), material, include_buckling=False)
    assert r1.Fu_pred_N > 1.1 * r0.Fu_pred_N
    assert r1.governing_mode != "joint"


def test_member_in_a_continuous_group_with_its_own_area_is_checked(material) -> None:
    b = generate_warren()
    chord = _member(b, "top_chord")
    assert "top_chord" in material.glue.exclude_groups
    r0 = analyze(b, material, include_buckling=False)
    assert r0.member(chord.id).util.U_joint is None
    data = b.model_dump(mode="json")
    next(m for m in data["members"] if m["id"] == chord.id)["glue_area_mm2"] = 400.0
    r1 = analyze(Bridge.model_validate(data), material, include_buckling=False)
    assert r1.member(chord.id).util.U_joint is not None
    glue = next(a for a in model_assumptions(r1) if a.startswith("Glued joint"))
    assert f"{chord.id} (400 mm²)" in glue
    assert "own glued area" in to_markdown(r1)


@pytest.mark.parametrize("area", [0.0, 0.5, 1e7])
def test_glue_area_out_of_range_is_rejected(area: float) -> None:
    data = generate_warren().model_dump(mode="json")
    data["members"][0]["glue_area_mm2"] = area
    with pytest.raises(ValidationError):
        Bridge.model_validate(data)


def test_glue_area_round_trips_and_defaults_stay_out_of_the_yaml() -> None:
    b = generate_warren()
    assert "glue_area_mm2" not in b.to_yaml()
    data = b.model_dump(mode="json")
    data["members"][0]["glue_area_mm2"] = 1234.5
    b2 = Bridge.model_validate(data)
    assert Bridge.from_yaml_str(b2.to_yaml()).members[0].glue_area_mm2 == 1234.5
