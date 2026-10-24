"""Translate recognized Responses content and function fragments."""

from typing import Any

from my_llm.codec.shared.conversion import blocks, select, uri_source
from my_llm.codec.shared.stream import BlockRef
from my_llm.semantic_loss import originated


def encode_content(content: Any, *, path: str = "$.content") -> list[dict]:
    result = []
    for index, block in enumerate(blocks(content)):
        kind = block.get("type")
        if kind in ("input_text", "output_text", "text"):
            value = {"type": "text", **select(block, "text")}
        elif kind == "input_image" and isinstance(block.get("image_url"), str):
            value = {"type": "image", "source": uri_source(block["image_url"])}
        else:
            continue
        result.append(
            originated(value, path if isinstance(content, str) else f"{path}[{index}]", block)
        )
    return result


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
