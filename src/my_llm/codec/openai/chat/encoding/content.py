"""Translate recognized Chat content and function calls."""

from typing import Any

from my_llm.codec.shared.conversion import arguments, blocks, select, uri_source
from my_llm.codec.shared.stream import BlockRef


def encode_content(content: Any) -> list[dict]:
    result = []
    for block in blocks(content):
        kind = block.get("type")
        if kind in ("text", "refusal"):
            result.append(select(block, "type", "text" if kind == "text" else "refusal"))
        elif kind == "input_audio":
            result.append(
                {
                    "type": kind,
                    "input_audio": select(block.get("input_audio", {}), "data", "format"),
                }
            )
        elif kind in ("image_url", "file"):
            file = block.get("file", {}) if kind == "file" else {}
            image = block.get("image_url", {}) if kind == "image_url" else {}
            uri = image.get("url") if isinstance(image, dict) else image
            if kind == "file":
                uri = file.get("file_data", file.get("file_url"))
            if isinstance(uri, str):
                source = uri_source(uri)
            elif file.get("file_id"):
                source = {"type": "file", "file_id": file["file_id"]}
            else:
                continue
            value = {"type": "image" if kind == "image_url" else "document", "source": source}
            if isinstance(image, dict) and "detail" in image:
                value["detail"] = image["detail"]
            if "filename" in file:
                value["filename"] = file["filename"]
            result.append(value)
    return result


def tool_block(call: dict, *, call_id: str | None = None) -> dict:
    fn = call.get("function", call)
    return {
        "type": "tool_use",
        "id": call.get("id") or call_id,
        "name": fn.get("name"),
        "input": arguments(fn.get("arguments", {})),
    }


def tool_operations(ref: BlockRef, call: dict) -> list[dict]:
    """Translate recognized fragments, preserving only open argument data."""
    fn = call.get("function", {})
    block = {"type": "tool_use"}
    if call.get("id"):
        block["id"] = call["id"]
    if fn.get("name"):
        block["name"] = fn["name"]
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
