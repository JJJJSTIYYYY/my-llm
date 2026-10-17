"""Stateless stream conversion helpers."""

from typing import NamedTuple

from my_llm.codec.shared.conversion import dictionaries, original, remember, residual
from my_llm.codec.shared.keys import SOURCE


class BlockRef(NamedTuple):
    """JSON-safe source address, independent of output order; append needs no ref."""

    scope: str
    index: int = 0
    part: int = 0


def batch(event: dict, operations: list[dict], protocol: str, consumed: dict | None = None) -> dict:
    # Opaque passthrough has interpreted no fields, even if their names happen
    # to match the consumption rules of a known event.
    if any(op.get("type") == "opaque" for op in operations):
        consumed = {}
        operations = [
            {**op, "protocol": protocol} if op.get("type") == "opaque" else op for op in operations
        ]
    value = remember(residual(event, consumed or {}), event, protocol)
    # Keep control data in our reserved namespace so native "type" / "events"
    # extensions can remain untouched at their original source paths.
    value[SOURCE]["events"] = operations
    return value


def is_batch(value: dict) -> bool:
    return "events" in value.get(SOURCE, {})


def operations(value: dict) -> list[dict]:
    return value[SOURCE]["events"] if is_batch(value) else [value]


def events(chunks: list, protocol: str):
    for chunk in dictionaries(chunks):
        native = original(chunk, protocol)
        if native is not None:
            yield {"type": "opaque", "value": native}
        elif is_batch(chunk):
            yield from events(operations(chunk), protocol)
        else:
            if chunk.get("type") != "opaque" or chunk.get("protocol", protocol) == protocol:
                yield chunk
