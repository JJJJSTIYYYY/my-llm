"""Stateless stream operations, source block identities and token accounting."""

from typing import NamedTuple

from my_llm.codec.shared.conversion import SOURCE, dictionaries, original, remember, residual


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


def compact(value):
    if isinstance(value, dict):
        return {key: compact(child) for key, child in value.items() if child is not None}
    return value


def record(
    value,
    protocol,
    *,
    input_base=None,
    input_add=None,
    output_base=None,
    output_add=None,
    total=None,
    cached=None,
    cache_write=None,
    reasoning=None,
):
    return remember(
        compact(
            {
                "input": {"base": input_base, "add": input_add},
                "output": {"base": output_base, "add": output_add},
                "total": total,
                "cached": cached,
                "cache_write": cache_write,
                "reasoning": reasoning,
            }
        ),
        value,
        protocol,
    )


def counts(value):
    def group(name):
        data = value.get(name, {})
        base = data.get("base")
        return base + sum(data.get("add", {}).values()) if base is not None else None

    incoming, outgoing = group("input"), group("output")
    total = value.get("total")
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return (
        incoming,
        outgoing,
        total,
        value.get("cached"),
        value.get("cache_write"),
        value.get("reasoning"),
    )
