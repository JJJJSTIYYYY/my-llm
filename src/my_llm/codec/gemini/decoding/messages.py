"""gemini messages: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    coalesce_response_messages,
    extras,
    original,
)
from my_llm.codec.utils.keys import OPAQUE
from my_llm.profile import ResolvedProfile

PROTOCOL = "gemini"


def decode_part(block: dict) -> dict:
    kind = block.get("type")
    if kind == "tool_use":
        return {
            **extras(block, "type", "id", "name", "input", "function_call"),
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
            else {"file_uri": source.get("url"), "mime_type": source.get("media_type")}
        )
        return {
            **extras(block, "type", "source", "detail"),
            "inline_data" if source.get("type") == "base64" else "file_data": data,
        }
    return extras(block, "type", "annotations") if kind == "text" else extras(block)


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    names = {}
    for message in coalesce_response_messages(messages):
        if message.get(OPAQUE):
            result.append(extras(message, OPAQUE))
            continue
        native = original(message, PROTOCOL)
        parts = []
        for block in message.get("content", []):
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
            if block.get("type") == "tool_result":
                if not block.get("name"):
                    block["name"] = names.get(block["tool_use_id"])
            parts.append(decode_part(block))
        if native is not None:
            result.extend(native)
        else:
            result.append(
                {
                    **extras(message, "role", "content", "tool_calls"),
                    "role": "model" if message["role"] == "assistant" else message["role"],
                    "parts": parts,
                }
            )
    return result
