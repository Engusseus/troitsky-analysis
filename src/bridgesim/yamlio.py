"""YAML loading for bridge, material and rules files.

``yaml.safe_load`` keeps the last of two equal keys without a word, so a copy-pasted
``P_ref_N`` or section dimension would silently change the results while the file still
appears to state both. :func:`load_yaml` rejects duplicate keys instead.
"""

from __future__ import annotations

from typing import Any

import yaml
from yaml.constructor import ConstructorError

_MERGE_TAG = "tag:yaml.org,2002:merge"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that raises on a mapping key written twice (merge keys ``<<`` excepted)."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked: set[int] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # Flattening rewrites node.value in place (merged pairs replace ``<<``), and a merge
        # source may be flattened before its own mapping is built. So check each mapping
        # once, the first time it is flattened, while its keys are still as written.
        if id(node) not in self._checked:
            self._checked.add(id(node))
            seen: dict[Any, yaml.Mark] = {}
            for key_node, _ in node.value:
                if key_node.tag == _MERGE_TAG:
                    continue
                key = self.construct_object(key_node, deep=True)
                try:
                    first = seen.get(key)
                except TypeError:  # unhashable key: the base class reports it
                    continue
                if first is not None:
                    raise ConstructorError(
                        "while reading a mapping", node.start_mark,
                        f"found duplicate key {key!r} (first given on line {first.line + 1})",
                        key_node.start_mark)
                seen[key] = key_node.start_mark
        super().flatten_mapping(node)


def load_yaml(text: str) -> Any:
    """Parse YAML like ``yaml.safe_load``, but reject duplicate mapping keys."""
    return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (a SafeLoader subclass)
