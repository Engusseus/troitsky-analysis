"""YAML I/O: bridges and materials round-trip exactly; invalid input is rejected."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from bridgesim.materials import bridge_material, load_material, material_from_yaml_str
from bridgesim.schema import Bridge, Material

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIAL_PATH = REPO_ROOT / "materials" / "popsicle_birch.yaml"

MINIMAL: dict[str, Any] = {
    "name": "minimal",
    "nodes": [
        {"id": "A", "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0},
        {"id": "B", "x_mm": 100.0, "y_mm": 0.0, "z_mm": 0.0},
    ],
    "sections": [{"id": "s", "b_mm": 10.0, "d_mm": 10.0}],
    "members": [{"id": "m", "i": "A", "j": "B", "section": "s"}],
    "deck": {"x_start_mm": 0.0, "x_end_mm": 100.0, "top_elevation_mm": 10.0,
             "clear_width_mm": 150.0},
    "supports": {"pinned": ["A"], "roller": ["B"]},
    "load": {"deck_support_nodes": ["A"]},
}


def minimal(mutate: Callable[[dict[str, Any]], None] | None = None) -> str:
    data = copy.deepcopy(MINIMAL)
    if mutate:
        mutate(data)
    return yaml.safe_dump(data, sort_keys=False)


# --------------------------------------------------------------------------- bridges

@pytest.mark.parametrize("which", ["example", "default"])
def test_bridge_round_trips(request: pytest.FixtureRequest, which: str) -> None:
    """Bridge.from_yaml_str(b.to_yaml()) == b, and writing again gives identical text."""
    b: Bridge = request.getfixturevalue(f"{which}_bridge")
    text = b.to_yaml()
    again = Bridge.from_yaml_str(text)
    assert again == b
    assert again.to_yaml() == text


def test_save_and_load_file(default_bridge: Bridge, tmp_path: Path) -> None:
    """Bridge.save / Bridge.from_file round-trip through disk."""
    path = tmp_path / "bridge.yaml"
    default_bridge.save(path)
    assert Bridge.from_file(path) == default_bridge


def test_minimal_bridge_defaults() -> None:
    """Omitted fields take their documented defaults."""
    b = Bridge.from_yaml_str(minimal())
    assert b.schema_version == 1
    assert b.material == "popsicle_birch"
    assert b.joint_fixity == "rigid"
    assert b.load.P_ref_N == 1000.0
    assert b.load.plate_length_mm == 200.0
    assert b.load.plate_width_mm == 90.0
    assert b.load.x_center_mm is None
    assert b.deck.thickness_mm == 2.0
    assert b.members[0].K == 1.0
    assert b.members[0].releases == []
    assert b.members[0].group == "other"
    assert Bridge.from_yaml_str(b.to_yaml()) == b


def test_embedded_material_custom_k_and_releases_round_trip(default_bridge: Bridge,
                                                            material: Material) -> None:
    """A bridge carrying a full Material object (with a measured value), a non-default
    buckling factor K and member end releases survives to_yaml / from_yaml_str.
    """
    data = default_bridge.model_dump()
    mat = material.model_dump()
    mat["E_MPa"] = {"value": 9123.5, "source": "measured", "note": "3-point bending, n=12"}
    data["material"] = mat
    data["joint_fixity"] = "pinned"
    data["members"][0]["K"] = 0.7
    data["members"][1]["releases"] = ["Rzi", "Rzj"]
    data["members"][2]["releases"] = ["Rxi", "Ryj"]
    data["load"]["x_center_mm"] = 431.25
    b = Bridge.model_validate(data)

    rt = Bridge.from_yaml_str(b.to_yaml())
    assert rt == b
    assert isinstance(rt.material, Material)
    assert rt.material.E_MPa.value == 9123.5
    assert rt.material.E_MPa.source == "measured"
    assert rt.members[0].K == 0.7
    assert rt.members[1].releases == ["Rzi", "Rzj"]
    assert rt.members[2].releases == ["Rxi", "Ryj"]
    assert rt.load.x_center_mm == 431.25
    assert bridge_material(rt) is rt.material


# --------------------------------------------------------------------------- invalid bridges

def _set(path: list[Any], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(d: dict[str, Any]) -> None:
        target = d
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _append(key: str, item: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    return lambda d: d[key].append(item)


INVALID: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "unknown node in member": (_set(["members", 0, "j"], "Z"), "unknown node"),
    "duplicate node id": (_set(["nodes", 1, "id"], "A"), "Duplicate id"),
    "duplicate member id": (
        _append("members", {"id": "m", "i": "B", "j": "A", "section": "s"}), "Duplicate id"),
    "duplicate section id": (_append("sections", {"id": "s", "b_mm": 5.0, "d_mm": 5.0}),
                             "Duplicate id"),
    "zero-length member": (_set(["nodes", 1, "x_mm"], 0.0), "zero length"),
    "section with sticks and b_mm": (
        _set(["sections", 0], {"id": "s", "sticks": 2, "layout": "flat", "b_mm": 10.0,
                               "d_mm": 10.0}), "not both"),
    "section with neither": (_set(["sections", 0], {"id": "s"}), "not both"),
    "sticks without layout": (_set(["sections", 0], {"id": "s", "sticks": 2}),
                              "must both be set"),
    "b_mm without d_mm": (_set(["sections", 0], {"id": "s", "b_mm": 10.0}),
                          "must both be set"),
    "zero sticks": (_set(["sections", 0], {"id": "s", "sticks": 0, "layout": "flat"}),
                    "greater than or equal"),
    "unknown section": (_set(["members", 0, "section"], "nope"), "unknown section"),
    "unknown field": (_set(["nodes", 0, "colour"], "red"), "Extra inputs"),
    "deck reversed": (_set(["deck", "x_end_mm"], -1.0), "x_end_mm must be greater"),
    "unknown support node": (_set(["supports", "roller"], ["Q"]), "Unknown node"),
    "unknown load node": (_set(["load", "deck_support_nodes"], ["Q"]), "Unknown node"),
    "no supports": (_set(["supports", "pinned"], []), "At least one pinned"),
    "bad release": (_set(["members", 0, "releases"], ["Rzz"]), "releases"),
    "zero K": (_set(["members", 0, "K"], 0.0), "greater than 0"),
    "unknown group": (_set(["members", 0, "group"], "cable"), "group"),
    "negative P_ref": (_set(["load", "P_ref_N"], -1.0), "greater than 0"),
    "bad joint fixity": (_set(["joint_fixity"], "glued"), "joint_fixity"),
}


@pytest.mark.parametrize("case", list(INVALID))
def test_invalid_bridge_yaml_raises(case: str) -> None:
    """Invalid geometry, references or fields raise pydantic.ValidationError naming
    the problem.
    """
    mutate, message = INVALID[case]
    with pytest.raises(ValidationError, match=message):
        Bridge.from_yaml_str(minimal(mutate))


# --------------------------------------------------------------------------- materials

def test_material_yaml_loads(material: Material) -> None:
    """The bundled material resolves by name, by path, from text and via Material.from_yaml,
    with the placeholder values from the file (all marked 'assumed').
    """
    text = MATERIAL_PATH.read_text(encoding="utf-8")
    assert load_material(MATERIAL_PATH) == material
    assert load_material(str(MATERIAL_PATH)) == material
    assert material_from_yaml_str(text) == material
    assert Material.from_yaml(MATERIAL_PATH) == material
    assert load_material(material) is material
    assert material.E_MPa.value == 10000
    assert material.density_kg_m3.value == 650
    assert material.glue.exclude_groups == ["top_chord", "bottom_chord"]
    assert (material.stick.length_mm, material.stick.width_mm, material.stick.thickness_mm) \
        == (115, 10, 2)
    assert sorted(material.assumed_keys()) == sorted(material.props())


def test_material_round_trips(material: Material) -> None:
    """Material -> YAML -> Material is lossless."""
    text = yaml.safe_dump(material.model_dump(mode="json"), sort_keys=False)
    assert material_from_yaml_str(text) == material


def test_bridge_material_resolves_name(default_bridge: Bridge, material: Material) -> None:
    """A bridge naming 'popsicle_birch' resolves to the bundled material."""
    assert bridge_material(default_bridge) == material


def test_unknown_material_raises() -> None:
    """A missing material name is a FileNotFoundError, not a silent default."""
    with pytest.raises(FileNotFoundError):
        load_material("no_such_wood")


def test_invalid_material_raises() -> None:
    """A material missing a required property (E) or with an unknown source is invalid."""
    data = yaml.safe_load(MATERIAL_PATH.read_text(encoding="utf-8"))
    no_e = copy.deepcopy(data)
    del no_e["E_MPa"]
    with pytest.raises(ValidationError, match="E_MPa"):
        material_from_yaml_str(yaml.safe_dump(no_e))
    bad_source = copy.deepcopy(data)
    bad_source["E_MPa"]["source"] = "guessed"
    with pytest.raises(ValidationError):
        material_from_yaml_str(yaml.safe_dump(bad_source))
