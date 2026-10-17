"""ollama messages: encoding conversion."""

from typing import Any

from my_llm.codec.ollama.encoding.messages.fields import CALL
from my_llm.codec.ollama.encoding.messages.images import media_type
from my_llm.codec.ollama.encoding.tools.calls import tool_block
from my_llm.codec.shared.conversion import (
    allocate_call_id,
    associate_tool_result,
    blocks,
    dictionaries,
    extras,
    remember,
    residual,
)
from my_llm.exceptions import MessageValidationError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"


def encode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    pending = []
    result = []
    messages = dictionaries(messages)
    # Reserve explicit IDs first so ID-less calls cannot shadow a later native ID.
    used_ids = {
        call["id"]
        for message in messages
        for call in (message.get("tool_calls") or [])
        if call.get("id")
    }
    for index, message in enumerate(messages):
        if (
            message.get("role") not in ("system", "user", "assistant", "tool")
            or "parts" in message
            or "type" in message
        ):
            raise MessageValidationError("message does not use the bound ollama protocol")
        content = blocks(message.get("content"))
        content.extend(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type(image), "data": image},
            }
            for image in (message.get("images") or [])
        )
        # Appending preserves the original indices of content-block extensions.
        if message.get("thinking"):
            content.append({"type": "thinking", "thinking": message["thinking"]})
        for call_index, call in enumerate(message.get("tool_calls") or []):
            call_id = call.get("id") or allocate_call_id(used_ids, f"call_{index}_{call_index}")
            block = tool_block(call, call_id=call_id)
            content.append(block)
            pending.append(block)
        role = message.get("role")
        if role == "tool":
            name = message.get("tool_name")
            call_id = associate_tool_result(pending, message.get("tool_call_id"), name)
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "name": name,
                    "content": message.get("content"),
                }
            ]
            role = "user"
        result.append(
            remember(
                {
                    **extras(
                        message,
                        "role",
                        "content",
                        "images",
                        "thinking",
                        "tool_calls",
                        "tool_name",
                        "tool_call_id",
                    ),
                    "role": role,
                    "content": content,
                },
                [message],
                PROTOCOL,
                preserve=residual(
                    message,
                    {
                        "role": True,
                        "content": True,
                        "tool_calls": [CALL],
                        "tool_call_id": True,
                        "reasoning_content": True,
                        "thinking": True,
                        "images": True,
                        "tool_name": True,
                    },
                ),
            )
        )
    return result
