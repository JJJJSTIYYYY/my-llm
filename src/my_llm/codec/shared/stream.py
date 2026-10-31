"""Stateless stream operations, source block identities and token accounting."""

from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from math import isfinite
from typing import NamedTuple

from my_llm.basic.data import dictionary_views
from my_llm.middleware.usage import normalize_usage


def optional_usage(method):
    """Keep optional native accounting failures out of message conversion."""

    @wraps(method)
    def convert(self, value):
        try:
            return method(self, value)
        except Exception:
            return None

    return convert


def project_metadata(value: object, fields: tuple[tuple[str, str], ...]) -> dict:
    """Read optional native metadata; discard invalid identity/time values."""
    try:
        if not isinstance(value, dict):
            return {}
        result = {}
        for source, target in fields:
            item = value.get(source)
            valid = isinstance(item, str)
            if target == "created_at":
                valid = (
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and item >= 0
                    and isfinite(item)
                )
            if valid:
                result[target] = item
        return result
    except Exception:
        return {}


class BlockRef(NamedTuple):
    """JSON-safe source address, independent of output order; append needs no ref."""

    scope: str
    index: int = 0
    part: int = 0


def metadata(values: dict, *, path: tuple[str, ...] = ()) -> dict:
    """Update one metadata dictionary with fields projected by the native encoder."""
    try:
        if (
            not isinstance(values, dict)
            or not isinstance(path, (tuple, list))
            or not all(isinstance(key, str) for key in path)
        ):
            values, path = {}, ()
        return {"type": "metadata", "values": deepcopy(values), "path": path}
    except Exception:
        return {"type": "metadata", "values": {}, "path": ()}


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
            try:
                operation["usage"] = usage(operation.get("usage"))
            except Exception:
                operation["usage"] = None
    return batch(operations, protocol)


def is_batch(value: dict) -> bool:
    """Whether this value wraps encoded operations rather than one public event."""
    return value.get("type") == "batch"


def operations(value: dict) -> list[dict]:
    """Read a batch operation list, or wrap one standalone operation."""
    return value["events"] if is_batch(value) else [value]


def events(chunks: list):
    """Yield public events from dictionaries/SDK views, flattening nested batches."""
    for chunk in dictionary_views(chunks):
        if is_batch(chunk):
            yield from events(operations(chunk))
        else:
            yield chunk


def compact(value):
    """Recursively omit None-valued dictionary fields from a usage record."""
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
    """Build canonical usage groups while preserving unreported counters as absent."""
    return normalize_usage(
        compact(
            {
                "input": {"base": input_base, "add": input_add} if input_base is not None else None,
                "output": {"base": output_base, "add": output_add}
                if output_base is not None
                else None,
                "total": total,
                "cached": cached,
                "cache_write": cache_write,
                "reasoning": reasoning,
            }
        )
    )


def counts(value):
    """Return flattened usage totals, deriving total only when both token groups are known."""

    value = normalize_usage(value)
    if value is None:
        return (None,) * 6

    def group(name):
        """Sum one reported base count and its additive counters."""
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
