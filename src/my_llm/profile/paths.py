"""Path syntax shared by declarative configuration and protocol translation."""

import re
from functools import lru_cache

from my_llm.basic.exceptions import ConfigurationError

_PATH = re.compile(r"([A-Za-z_][A-Za-z_0-9-]*)(?:\[(\d+)\])?")


def path_keys(path: str) -> tuple[str | int, ...]:
    """Parse an SDK-rooted path, reusing immutable keys across stream chunks."""
    if not isinstance(path, str):
        raise ConfigurationError(f"parameter path must start with '$.': {path!r}")
    return _parsed_path(path)


@lru_cache(maxsize=512)
def _parsed_path(path: str) -> tuple[str | int, ...]:
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
