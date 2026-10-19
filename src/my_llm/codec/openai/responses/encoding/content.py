"""Translate recognized Responses content and function fragments."""

from typing import Any

from my_llm.codec.shared.conversion import blocks, select, uri_source
from my_llm.codec.shared.stream import BlockRef


def encode_content(content: Any) -> list[dict]:
    result = []
    for block in blocks(content):
        kind = block.get("type")
        if kind in ("input_text", "output_text", "text"):
            value = {"type": "text", **select(block, "text")}
            if "annotations" in block:
                value["annotations"] = [
                    select(
                        item,
                        "type",
                        "file_id",
                        "index",
                        "filename",
                        "start_index",
                        "end_index",
                        "title",
                        "url",
                        "container_id",
                    )
                    for item in block["annotations"]
                ]
            if "logprobs" in block:
                value["logprobs"] = [
                    select(item, "token", "logprob", "bytes")
                    | (
                        {
                            "top_logprobs": [
                                select(child, "token", "logprob", "bytes")
                                for child in item["top_logprobs"]
                            ]
                        }
                        if "top_logprobs" in item
                        else {}
                    )
                    for item in block["logprobs"]
                ]
            result.append(value)
        elif kind == "refusal":
            result.append(select(block, "type", "refusal"))
        elif kind in ("input_image", "input_file"):
            uri = (
                block.get("image_url")
                if kind == "input_image"
                else block.get("file_data", block.get("file_url"))
            )
            if isinstance(uri, str):
                source = uri_source(uri)
            elif block.get("file_id"):
                source = {"type": "file", "file_id": block["file_id"]}
            else:
                continue
            result.append(
                {
                    "type": "image" if kind == "input_image" else "document",
                    "source": source,
                    **select(block, "detail", "filename"),
                }
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
