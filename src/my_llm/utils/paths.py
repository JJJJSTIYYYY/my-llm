"""Parse, read, write and delete addressed dictionary, list and SDK model values."""

import re
from functools import lru_cache
from typing import Any

from my_llm.basic.exceptions import ConfigurationError
from my_llm.utils.data import _MISSING, plain, read

_PATH = re.compile(r"([A-Za-z_][A-Za-z_0-9-]*)(?:\[(\d+)\])?")


def path_keys(path: str) -> tuple[str | int, ...]:
    """Parse an SDK-rooted path, reusing immutable keys across stream chunks."""
    if not isinstance(path, str):
        raise ConfigurationError(f"parameter path must start with '$.': {path!r}")
    return _parsed_path(path)


@lru_cache(maxsize=512)
def _parsed_path(path: str) -> tuple[str | int, ...]:
    """Cache validated SDK-rooted field names and nonnegative list indices."""
    if not path.startswith("$."):
        raise ConfigurationError(f"parameter path must start with '$.': {path!r}")
    keys: list[str | int] = []
    for component in path[2:].split("."):
        match = _PATH.fullmatch(component)
        if not match:
            raise ConfigurationError(f"unsupported path syntax: {path!r}")
        keys.append(match[1])
        if match[2] is not None:
            keys.append(int(match[2]))
    return tuple(keys)


def path_value(value: Any, path: str, default: Any = None) -> Any:
    """Read only the addressed keys; a present None is a value, not absence."""
    for key in path_keys(path):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)) or key >= len(value):
                return default
            value = value[key]
        else:
            value = read(value, key, _MISSING)
        if value is _MISSING:
            return default
    return value


def set_path(value: dict, path: str, item: Any) -> None:
    """Write one addressed value, creating only the necessary containers."""
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
        if child is None:
            child = [] if isinstance(keys[index + 1], int) else {}
        elif not isinstance(child, (dict, list)):
            child = plain(child)
        value[key] = child
        value = child


def delete_path(value: dict, path: str) -> None:
    """Delete a present parameter without creating a missing parent."""
    keys = path_keys(path)
    for key in keys[:-1]:
        value = (
            read(value, key, _MISSING)
            if isinstance(key, str)
            else (value[key] if isinstance(value, list) and key < len(value) else _MISSING)
        )
        if value is _MISSING or value is None:
            return
    key = keys[-1]
    if isinstance(value, dict) and isinstance(key, str):
        value.pop(key, None)
    elif isinstance(value, list) and isinstance(key, int) and key < len(value):
        del value[key]
