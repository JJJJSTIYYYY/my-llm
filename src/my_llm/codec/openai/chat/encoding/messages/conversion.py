"""openai/chat messages: encoding conversion."""

from typing import Any

from my_llm.codec.openai.chat.encoding.messages.content import encode_content
from my_llm.codec.openai.chat.encoding.messages.fields import CALL, content_fields
from my_llm.codec.openai.chat.encoding.tools.calls import tool_block
from my_llm.codec.shared.conversion import (
    dictionaries,
    extras,
    remember,
    residual,
)
from my_llm.exceptions import MessageValidationError
from my_llm.profile.resolution import ResolvedProfile

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
        content = encode_content(message.get("content"))
        # Appending preserves the original indices of content-block extensions.
        if message.get("reasoning_content"):
            content.append({"type": "thinking", "thinking": message["reasoning_content"]})
        if message.get("refusal") is not None:
            content.append({"type": "refusal", "refusal": message["refusal"]})
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
                    **extras(
                        message,
                        "content",
                        "reasoning_content",
                        "refusal",
                        "tool_calls",
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
                        "content": [content_fields]
                        if isinstance(message.get("content"), list)
                        else True,
                        "tool_calls": [CALL],
                        "tool_call_id": True,
                        "reasoning_content": True,
                        "refusal": True,
                        "thinking": True,
                        "images": True,
                        "tool_name": True,
                    },
                ),
            )
        )
    return result
