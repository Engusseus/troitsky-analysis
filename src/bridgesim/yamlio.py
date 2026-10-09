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

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            seen: dict[Any, yaml.Mark] = {}
            for key_node, _ in node.value:
                if key_node.tag == _MERGE_TAG:
                    continue
                key = self.construct_object(key_node, deep=True)
                try:
                    first = seen.setdefault(key, key_node.start_mark)
                except TypeError:  # unhashable key: the base class reports it
                    continue
                if first is not key_node.start_mark:
                    raise ConstructorError(
                        "while reading a mapping", node.start_mark,
                        f"found duplicate key {key!r} (first given on line {first.line + 1})",
                        key_node.start_mark)
        return super().construct_mapping(node, deep=deep)


def load_yaml(text: str) -> Any:
    """Parse YAML like ``yaml.safe_load``, but reject duplicate mapping keys."""
    return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (a SafeLoader subclass)
