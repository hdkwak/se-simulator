"""Dot-path utilities for reading and writing nested dict/list structures."""
from __future__ import annotations

import re
from typing import Any

_INDEX_RE = re.compile(r"^(.*)\[(\d+)\]$")


def _parse_key(segment: str) -> tuple[str, int | None]:
    """Parse a path segment into (key, index_or_None).

    Examples
    --------
    "layers[0]"  -> ("layers", 0)
    "thickness"  -> ("thickness", None)
    "[2]"        -> ("", 2)   # bare index with no preceding key name
    """
    m = _INDEX_RE.match(segment)
    if m:
        return m.group(1), int(m.group(2))
    return segment, None


def resolve_get(obj: dict[str, Any], dotpath: str) -> Any:
    """Retrieve a value from a nested dict/list using a dot-path string.

    Supports:
      - Simple keys:   "forward_model.engine_override"
      - List indexing: "forward_model.sample.inline.layers[0].thickness_nm"

    Raises ``KeyError`` or ``IndexError`` if the path does not resolve.
    """
    current: Any = obj
    for segment in dotpath.split("."):
        key, idx = _parse_key(segment)
        if key:
            current = current[key]
        if idx is not None:
            current = current[idx]
    return current


def resolve_set(obj: dict[str, Any], dotpath: str, value: Any) -> None:
    """Set a value in a nested dict/list using a dot-path string.

    Modifies *obj* in place.  Supports the same path syntax as
    :func:`resolve_get`.
    """
    segments = dotpath.split(".")
    current: Any = obj

    # Navigate to the parent of the final target
    for segment in segments[:-1]:
        key, idx = _parse_key(segment)
        if key:
            current = current[key]
        if idx is not None:
            current = current[idx]

    # Set the final element
    last_key, last_idx = _parse_key(segments[-1])
    if last_idx is not None:
        # e.g. "layers[0]" as final segment — navigate into the key first
        if last_key:
            current = current[last_key]
        current[last_idx] = value
    else:
        current[last_key] = value
