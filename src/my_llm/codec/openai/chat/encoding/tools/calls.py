"""Translate native function calls into intermediate blocks."""

from my_llm.codec.shared.chunks import BlockRef
from my_llm.codec.shared.conversion import arguments, extras


def tool_block(call: dict, *, call_id: str | None = None) -> dict:
    fn = call.get("function", call)
    extensions = extras(fn, "name", "arguments")
    return {
        **extras(call, "id", "type", "function", "index"),
        "type": "tool_use",
        "id": call.get("id") or call_id,
        "name": fn.get("name"),
        "input": arguments(fn.get("arguments", {})),
        **({"function": extensions} if extensions else {}),
    }


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
    result = [{"type": "block_update", "ref": ref, "block": block, "merge": ["function"]}]
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
