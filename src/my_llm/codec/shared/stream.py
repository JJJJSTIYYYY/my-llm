"""Stateless stream operations, source block identities and token accounting."""

from collections.abc import Callable
from copy import deepcopy
from typing import NamedTuple

from my_llm.basic.data import dictionary_views


class BlockRef(NamedTuple):
    """JSON-safe source address, independent of output order; append needs no ref."""

    scope: str
    index: int = 0
    part: int = 0


def metadata(values: dict, *, path: tuple[str, ...] = ()) -> dict:
    """Update one metadata dictionary with fields projected by the native encoder."""
    return {"type": "metadata", "values": deepcopy(values), "path": path}


def batch(operations: list[dict], protocol: str) -> dict:
    """Group recognized operations with their source protocol, without native payloads."""
    return {"type": "batch", "protocol": protocol, "events": operations}


def encode_batch(
    operations: list[dict],
    protocol: str,
    *,
    usage: Callable[[dict | None], dict | None],
) -> dict:
    """Translate token counts before grouping recognized stream operations."""
    for operation in operations:
        if operation["type"] == "usage":
            operation["usage"] = usage(operation["usage"])
    return batch(operations, protocol)


def is_batch(value: dict) -> bool:
    return value.get("type") == "batch"


def operations(value: dict) -> list[dict]:
    return value["events"] if is_batch(value) else [value]


def events(chunks: list):
    for chunk in dictionary_views(chunks):
        if is_batch(chunk):
            yield from events(operations(chunk))
        else:
            yield chunk


def compact(value):
    if isinstance(value, dict):
        return {key: compact(child) for key, child in value.items() if child is not None}
    return value


def record(
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
    return compact(
        {
            "input": {"base": input_base, "add": input_add} if input_base is not None else None,
            "output": {"base": output_base, "add": output_add} if output_base is not None else None,
            "total": total,
            "cached": cached,
            "cache_write": cache_write,
            "reasoning": reasoning,
        }
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
