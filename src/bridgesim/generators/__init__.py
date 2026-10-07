"""Parametric bridge generators."""

from bridgesim.generators.warren import WarrenParams, generate_warren

GENERATORS = {"warren": (WarrenParams, generate_warren)}

__all__ = ["GENERATORS", "WarrenParams", "generate_warren"]
