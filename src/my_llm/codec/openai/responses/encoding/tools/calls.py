"""Responses function-call fragments use native item fields."""

from my_llm.codec.shared.chunks import BlockRef


def tool_operations(ref: BlockRef, *, call_id=None, name=None, arguments=None) -> list[dict]:
    block = {"type": "tool_use"}
    if call_id:
        block["id"] = call_id
    if name:
        block["name"] = name
    result = [{"type": "block_update", "ref": ref, "block": block}]
    if arguments is not None:
        result.append(
            {
                "type": "block_delta",
                "ref": ref,
                "kind": "tool_use",
                "field": "arguments",
                "value": arguments,
            }
        )
    return result
