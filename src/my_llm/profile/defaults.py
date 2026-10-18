"""Independent configuration snapshots from registered protocol declarations."""

from dataclasses import asdict
from typing import Any

from my_llm.profile.configuration import Protocol
from my_llm.registry import codec_for


def protocol_defaults(protocol: Protocol) -> dict[str, Any]:
    """Copy the registered protocol's declarative settings without constructing it."""
    return asdict(codec_for(protocol).defaults)
