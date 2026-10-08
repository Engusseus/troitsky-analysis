"""Load and validate material YAML files (see ``materials/popsicle_birch.yaml``)."""

from __future__ import annotations

from pathlib import Path

import yaml

from bridgesim.paths import data_dir
from bridgesim.schema import Bridge, Material


def load_material(ref: str | Path | Material) -> Material:
    """Resolve a material: a Material object, a YAML path, or a bundled name."""
    if isinstance(ref, Material):
        return ref
    path = Path(ref)
    if path.suffix not in (".yaml", ".yml"):
        path = data_dir("materials") / f"{ref}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Material {ref!r} not found (looked for {path})")
    return Material.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def material_from_yaml_str(text: str) -> Material:
    return Material.model_validate(yaml.safe_load(text))


def bridge_material(bridge: Bridge) -> Material:
    return load_material(bridge.material)
