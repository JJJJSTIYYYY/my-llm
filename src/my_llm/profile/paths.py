"""Path syntax shared by declarative configuration and protocol translation."""

import re

from my_llm.exceptions import ConfigurationError

_PATH = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(?:\[(\d+)\])?")


def path_keys(path: str) -> list[str | int]:
    """Parse dotted keys and numeric indexes, with an optional $. prefix."""
    keys: list[str | int] = []
    for component in path.removeprefix("$.").split("."):
        match = _PATH.fullmatch(component)
        if not match:
            raise ConfigurationError(f"unsupported path syntax: {path!r}")
        keys.append(match[1])
        if match[2] is not None:
            keys.append(int(match[2]))
    return keys
