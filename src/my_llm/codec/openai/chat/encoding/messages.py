"""openai/chat messages: encoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    dictionaries,
    extras,
    from_chat_content,
    remember,
    residual,
    tool_block,
)
from my_llm.codec.utils.preservation import CALL, content_fields
from my_llm.exceptions import MessageValidationError
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_chat"


def encode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in dictionaries(messages):
        if (
            message.get("role") not in ("system", "developer", "user", "assistant", "tool")
            or "parts" in message
            or "type" in message
        ):
            raise MessageValidationError("message does not use the bound openai_chat protocol")
        content = from_chat_content(message.get("content"))
        # Appending preserves the original indices of content-block extensions.
        if message.get("reasoning_content"):
            content.append({"type": "thinking", "thinking": message["reasoning_content"]})
        content.extend(tool_block(c) for c in (message.get("tool_calls") or []))
        role = message.get("role")
        if role == "tool":
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id"),
                    "content": message.get("content"),
                    "name": message.get("name"),
                }
            ]
            role = "user"
        result.append(
            remember(
                {
                    **extras(message, "content", "reasoning_content", "tool_calls", "tool_call_id"),
                    "role": role,
                    "content": content,
                },
                [message],
                PROTOCOL,
                preserve=residual(
                    message,
                    {
                        "role": True,
                        "content": [content_fields]
                        if isinstance(message.get("content"), list)
                        else True,
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
