"""openai/responses encoding: content, fields and tool helpers."""

from typing import Any

from my_llm.codec.shared.chunks import BlockRef
from my_llm.codec.shared.conversion import blocks, extras, residual


def encode_content(content: Any) -> list[dict]:
    result = []
    for block in blocks(content):
        kind = block.get("type")
        value = extras(block)
        if kind in ("input_text", "output_text"):
            value["type"] = "text"
        elif kind in ("input_image", "input_file"):
            file = block
            uri = (
                block.get("image_url")
                if kind == "input_image"
                else file.get("file_data", file.get("file_url"))
            )
            image_options = (
                {"detail": uri["detail"]} if isinstance(uri, dict) and "detail" in uri else {}
            )
            if isinstance(uri, dict):
                uri = uri.get("url")
            # Without a usable URI (for example, a file_id-only block), leave
            # the native block intact instead of consuming fields we cannot map.
            if isinstance(uri, str):
                source = {"type": "url", "url": uri}
                if uri.startswith("data:"):
                    header, _, data = uri.partition(",")
                    source = {
                        "type": "base64",
                        "media_type": header[5:].removesuffix(";base64"),
                        "data": data,
                    }
                value = {
                    **residual(block, content_fields(block)),
                    **image_options,
                    "type": "image" if kind == "input_image" else "document",
                    "source": source,
                }
        result.append(value)
    return result


def content_fields(part: dict) -> dict:
    kind = part.get("type")
    if kind in ("input_text", "output_text", "text"):
        return {
            "type": True,
            "text": True,
            "annotations": True,
            **({"logprobs": True} if kind == "output_text" else {}),
        }
    if kind == "input_image":
        return {"type": True, "image_url": True, "detail": True}
    if kind == "input_file":
        return {"type": True, "file_data": True, "file_url": True}
    return {}


def response_fields(item: dict) -> dict:
    kind = item.get("type", "message")
    common = {"type": True, "id": True, "status": True}
    if kind == "function_call":
        return {**common, "call_id": True, "name": True, "arguments": True}
    if kind == "function_call_output":
        return {**common, "call_id": True, "output": True}
    if kind == "reasoning":
        return {**common, "summary": [{"type": True, "text": True}]}
    if kind == "message":
        return {**common, "role": True, "content": [content_fields]}
    return {}


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
