"""SDK field access, lightweight paths and shared value expansion."""

from __future__ import annotations

import re
from typing import Any

from openai import Omit

from my_llm.types.exceptions import ConfigurationError, IncompleteStreamError

RESPONSE_BLOCKS = frozenset({"input_text", "output_text", "input_image", "input_file"})


def read(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value.copy() if isinstance(value, (dict, list)) else value


_PATH = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(?:\[(\d+)\])?")


def path_value(value: Any, path: str | Omit) -> Any:
    """Read dotted attributes/mapping keys and numeric indexes, without dumping SDK objects."""
    if isinstance(path, Omit):
        return None
    if path.startswith("$."):
        path = path[2:]
    for component in path.split("."):
        match = _PATH.fullmatch(component)
        if not match:
            raise ConfigurationError(f"unsupported path syntax: {path!r}")
        value = read(value, match[1])
        if value is None:
            return None
        if match[2] is not None:
            index = int(match[2])
            if not isinstance(value, (list, tuple)) or index >= len(value):
                return None
            value = value[index]
    return value


def _expand(value: Any, effort: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand(item, effort) for key, item in value.items() if not isinstance(item, Omit)
        }
    if isinstance(value, list):
        return [_expand(item, effort) for item in value]
    return effort if value == "{effort}" else value


def text_delta(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise IncompleteStreamError("configured text path did not return a string")
    return value
