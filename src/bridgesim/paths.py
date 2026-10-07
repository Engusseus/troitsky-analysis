"""Locate bundled data (rules, materials, examples).

In a source checkout the data lives at the repository root so next year's team can edit
it directly. In an installed wheel it is shipped inside the package as ``bridgesim/data``.
"""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def data_dir(kind: str) -> Path:
    """Directory for ``kind`` in {"rules", "materials", "examples"}."""
    for candidate in (_PKG_DIR / "data" / kind, _PKG_DIR.parents[1] / kind):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the bundled {kind!r} directory")


def list_yaml(kind: str) -> list[Path]:
    return sorted(data_dir(kind).glob("*.yaml"))
