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


# ------------------------------------------------------------------ fast and slow tests
#
# A handful of tests run small crush simulations or calibration loops and take two thirds
# of the suite's time. They are marked `slow`, so a quick check while iterating is
# `pytest -m "not slow"` (about 2 minutes instead of 6 on 4 cores); a plain `pytest` (and
# CI) runs everything.

SLOW_TESTS = {
    "test_bridge_calibration_recovers_known_factors",
    "test_three_point_bending",
    "test_calibration_closes_the_loop",
    "test_simply_supported_beam_matches_beam_theory",
    "test_agent_loop_through_mcp",
    "test_block_cell_mm_zero_is_the_rigid_block_of_round_27",
    "test_the_synthetic_bridges_stand_on_blocks_of_cells",
}


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "slow: runs crush simulations or calibration loops (deselect with "
                   "-m 'not slow')")


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        # the shared synthetic crush (a fixture) is the single most expensive setup
        if item.name in SLOW_TESTS or "synthetic_crush" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)
