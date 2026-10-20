"""Independent configuration snapshots from registered protocol declarations."""

from copy import deepcopy
from dataclasses import fields
from typing import Any

from my_llm.profile.configuration import Protocol
from my_llm.registry import codec_for


def protocol_defaults(protocol: Protocol) -> dict[str, Any]:
    """Copy declared wrappers without flattening Parameter or Config objects."""
    defaults = codec_for(protocol).defaults
    return deepcopy({item.name: getattr(defaults, item.name) for item in fields(defaults)})
