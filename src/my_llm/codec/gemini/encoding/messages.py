"""gemini messages: encoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    allocate_call_id,
    associate_tool_result,
    dictionaries,
    extras,
    remember,
    residual,
)
from my_llm.codec.utils.preservation import part_fields
from my_llm.exceptions import MessageValidationError
from my_llm.profile import ResolvedProfile

PROTOCOL = "gemini"


def encode_part(part: dict, call_id: str) -> dict:
    if "function_call" in part:
        call = part["function_call"]
        return {
            **residual(part, part_fields(part)),
            "type": "tool_use",
            "id": call.get("id") or call_id,
            "name": call.get("name"),
            "input": call.get("args", {}),
        }
    if "function_response" in part:
        call = part["function_response"]
        return {
            **residual(part, part_fields(part)),
            "type": "tool_result",
            "name": call.get("name"),
            "tool_use_id": call.get("id"),
            "content": call.get("response", {}),
        }
    if "text" in part:
        if part.get("thought"):
            return {**extras(part, "text", "thought"), "type": "thinking", "thinking": part["text"]}
        return {**part, "type": "text"}
    if "inline_data" in part or "file_data" in part:
        data = part.get("inline_data", part.get("file_data"))
        mime = data.get("mime_type", "application/octet-stream")
        source = (
            {"type": "base64", "media_type": mime, "data": data.get("data")}
            if "inline_data" in part
            else {"type": "url", "url": data.get("file_uri"), "media_type": mime}
        )
        return {
            **residual(part, part_fields(part)),
            "type": "image" if mime.startswith("image/") else "document",
            "source": source,
        }
    return part


def encode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    pending: list[dict] = []
    messages = dictionaries(messages)
    if any(not isinstance(message.get("parts"), list) for message in messages):
        raise MessageValidationError("message does not use the bound gemini protocol")
    # Reserve IDs from later messages too, before synthesizing any missing IDs.
    used_ids = {
        part["function_call"]["id"]
        for message in messages
        for part in message["parts"]
        if part.get("function_call", {}).get("id")
    }
    for index, message in enumerate(messages):
        content = []
        for part_index, part in enumerate(message["parts"]):
            call = part.get("function_call")
            call_id = (
                call.get("id") or allocate_call_id(used_ids, f"call_{index}_{part_index}")
                if call is not None
                else ""
            )
            block = encode_part(part, call_id)
            if block.get("type") == "tool_use":
                pending.append(block)
            elif block.get("type") == "tool_result":
                block["tool_use_id"] = associate_tool_result(
                    pending, block.get("tool_use_id"), block.get("name")
                )
            content.append(block)
        result.append(
            remember(
                {
                    **extras(message, "role", "parts"),
                    "role": "assistant"
                    if message.get("role") == "model"
                    else message.get("role", "user"),
                    "content": content,
                },
                [message],
                PROTOCOL,
                preserve=residual(message, {"role": True, "parts": [part_fields]}),
            )
        )
    return result
