"""Stateless stream conversion helpers."""

from typing import Any, NamedTuple

from my_llm.codec.utils.conversion import dictionaries, extras, original, remember, residual
from my_llm.codec.utils.keys import SOURCE
from my_llm.exceptions import ResponseFailedError, ResponseIncompleteError


class BlockRef(NamedTuple):
    """JSON-safe source address, independent of output order; append needs no ref."""

    scope: str
    index: int = 0
    part: int = 0


def tool_operations(ref: BlockRef, call: dict) -> list[dict]:
    """Translate Chat-shaped call fragments into provider-independent operations."""
    fn = call.get("function", {})
    block = {
        **extras(call, "type", "index", "id", "function", "name", "input"),
        "type": "tool_use",
    }
    if call.get("id"):
        block["id"] = call["id"]
    if fn.get("name"):
        block["name"] = fn["name"]
    extensions = extras(fn, "name", "arguments")
    if extensions:
        block["function"] = extensions
    result = [{"type": "block_update", "ref": ref, "block": block}]
    if "arguments" in fn:
        result.append(
            {
                "type": "block_delta",
                "ref": ref,
                "kind": "tool_use",
                "field": "arguments",
                "value": fn["arguments"],
            }
        )
    return result


def batch(event: dict, operations: list[dict], protocol: str, consumed: dict | None = None) -> dict:
    # Opaque passthrough has interpreted no fields, even if their names happen
    # to match the consumption rules of a known event.
    if any(op.get("type") == "opaque" for op in operations):
        consumed = {}
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
            yield from operations(chunk)
        else:
            yield chunk


def check_error(event: dict) -> None:
    kind = event.get("type")
    if kind in ("response.failed", "response.incomplete", "error") or event.get("error"):
        error = ResponseIncompleteError if kind == "response.incomplete" else ResponseFailedError
        response = event.get("response", {})
        raise error(
            "provider generation failed",
            details=response.get(
                "incomplete_details", response.get("error", event.get("error", event))
            ),
        )


def metadata(event: dict) -> dict:
    return {
        key: event[key]
        for key in ("id", "model", "created", "created_at", "response_id")
        if key in event
    }


def envelope(**parameters: Any) -> dict:
    return {
        "id": parameters.get("response_id", "generated"),
        "model": parameters.get("model", ""),
        "created": parameters.get("created", 0),
    }
