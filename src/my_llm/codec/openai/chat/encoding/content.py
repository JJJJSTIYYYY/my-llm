"""openai/chat encoding: content, fields and tool helpers."""

from typing import Any

from my_llm.codec.shared.chunks import BlockRef
from my_llm.codec.shared.conversion import arguments, blocks, extras, residual

CALL = {"id": True, "type": True, "function": {"name": True, "arguments": True}}


def encode_content(content: Any) -> list[dict]:
    result = []
    for block in blocks(content):
        kind = block.get("type")
        value = extras(block)
        if kind in ("image_url", "file"):
            file = block.get("file", block)
            uri = (
                block.get("image_url")
                if kind == "image_url"
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
                    "type": "image" if kind == "image_url" else "document",
                    "source": source,
                }
        result.append(value)
    return result


def content_fields(part: dict) -> dict:
    kind = part.get("type")
    if kind not in (
        "text",
        "image_url",
        "file",
    ):
        return {}
    if kind == "text":
        return {"type": True, "text": True, "annotations": True}
    if kind == "image_url":
        return {
            "type": True,
            "image_url": {"url": True, "detail": True}
            if isinstance(part.get("image_url"), dict)
            else True,
        }
    if isinstance(part.get("file"), dict):
        return {"type": True, "file": {"file_data": True, "file_url": True}}
    return {"type": True, "file_data": True, "file_url": True}


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
