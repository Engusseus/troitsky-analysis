"""Shared fixtures for the bridgesim test suite.

Session-scoped fixtures return shared objects: tests must not mutate them. To change a
bridge, copy it first (``bridge.model_copy(deep=True)``) or build a new one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bridgesim.analysis import AnalysisResult, analyze
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.materials import load_material
from bridgesim.rules import RuleSet
from bridgesim.schema import Bridge, Material

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "warren_2027.yaml"


@pytest.fixture(scope="session")
def material() -> Material:
    """The bundled placeholder birch material (E = 10000 MPa)."""
    return load_material("popsicle_birch")


@pytest.fixture(scope="session")
def ruleset() -> RuleSet:
    """The bundled Troitsky 2027 rule set."""
    return RuleSet.load()


@pytest.fixture(scope="session")
def default_params() -> WarrenParams:
    return WarrenParams()


@pytest.fixture(scope="session")
def default_bridge(default_params: WarrenParams) -> Bridge:
    """The default through Warren truss, freshly generated."""
    return generate_warren(default_params)


@pytest.fixture(scope="session")
def example_bridge() -> Bridge:
    """The committed example ``examples/warren_2027.yaml``."""
    return Bridge.from_file(EXAMPLE_PATH)


@pytest.fixture(scope="session")
def default_result(default_bridge: Bridge, material: Material) -> AnalysisResult:
    """Linear analysis of the default Warren truss at P_ref."""
    return analyze(default_bridge, material)
