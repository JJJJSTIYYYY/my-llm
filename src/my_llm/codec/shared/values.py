"""SDK field access, lightweight paths and shared value expansion."""

from __future__ import annotations

import re
from typing import Any

from openai import Omit

from my_llm.exceptions import ConfigurationError


def read(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value.copy() if isinstance(value, (dict, list)) else value


_PATH = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(?:\[(\d+)\])?")


def path_keys(path: str) -> list[str | int]:
    """Parse the dotted-key/index syntax shared by read and write paths."""
    keys: list[str | int] = []
    for component in path.removeprefix("$.").split("."):
        match = _PATH.fullmatch(component)
        if not match:
            raise ConfigurationError(f"unsupported path syntax: {path!r}")
        keys.append(match[1])
        if match[2] is not None:
            keys.append(int(match[2]))
    return keys


def set_path(value: dict, path: str, item: Any) -> None:
    """Write a path, creating containers and retaining existing sibling fields."""
    keys = path_keys(path)
    for index, key in enumerate(keys):
        if isinstance(key, int):
            if not isinstance(value, list):
                raise ConfigurationError(f"path requires a list: {path!r}")
            if key >= len(value):
                value.extend([None] * (key + 1 - len(value)))
        elif not isinstance(value, dict):
            raise ConfigurationError(f"path requires a dictionary: {path!r}")
        if index == len(keys) - 1:
            value[key] = item
            return
        child = value.get(key) if isinstance(value, dict) else value[key]
        child = (
            plain(child) if child is not None else ([] if isinstance(keys[index + 1], int) else {})
        )
        value[key] = child
        value = child


def path_value(value: Any, path: str | Omit) -> Any:
    """Read dotted attributes/mapping keys and numeric indexes, without dumping SDK objects."""
    if isinstance(path, Omit):
        return None
    for key in path_keys(path):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)) or key >= len(value):
                return None
            value = value[key]
        else:
            value = read(value, key)
        if value is None:
            return None
    return value


def _expand(value: Any, effort: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand(item, effort) for key, item in value.items() if not isinstance(item, Omit)
        }
    if isinstance(value, list):
        return [_expand(item, effort) for item in value]
    return effort if value == "{effort}" else value
