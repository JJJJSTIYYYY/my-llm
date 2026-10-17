"""gemini decoding: content, fields and tool helpers."""

from my_llm.codec.shared.conversion import (
    extras,
)
from my_llm.codec.shared.rendering import project_block

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
}


def decode_part(block: dict, *, foreign: bool = False) -> dict:
    kind = block.get("type")
    if kind == "tool_use":
        return {
            **extras(block, "type", "id", "name", "input", "function_call"),
            **({"thought_signature": "skip_thought_signature_validator"} if foreign else {}),
            "function_call": {
                **block.get("function_call", {}),
                "id": block["id"],
                "name": block["name"],
                "args": block["input"],
            },
        }
    if kind == "tool_result":
        response = block.get("content")
        return {
            **extras(
                block, "type", "tool_use_id", "name", "content", "is_error", "function_response"
            ),
            "function_response": {
                **block.get("function_response", {}),
                "id": block["tool_use_id"],
                "name": block.get("name"),
                "response": response if isinstance(response, dict) else {"result": response},
            },
        }
    if kind == "thinking":
        return {
            **extras(block, "type", "thinking", "signature"),
            "text": block.get("thinking", ""),
            "thought": True,
        }
    if kind in ("image", "document"):
        source = block.get("source", {})
        data = (
            {"data": source.get("data"), "mime_type": source.get("media_type")}
            if source.get("type") == "base64"
            else {
                "file_uri": source.get("url"),
                **({"mime_type": source["media_type"]} if source.get("media_type") else {}),
            }
        )
        return {
            **extras(block, "type", "source", "detail"),
            "inline_data" if source.get("type") == "base64" else "file_data": data,
        }
    return extras(block, "type", "annotations") if kind == "text" else extras(block)


def foreign_block(block: dict) -> dict | None:
    return project_block(
        block,
        FIELDS,
        text_fields={"thinking": "thinking", "refusal": "refusal"},
        omitted=("redacted_thinking",),
    )
