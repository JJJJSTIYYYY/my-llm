"""openai/responses messages: encoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    arguments,
    dictionaries,
    extras,
    from_chat_content,
    remember,
    residual,
)
from my_llm.codec.utils.keys import OPAQUE
from my_llm.codec.utils.preservation import response_fields
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_responses"


def encode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for item in dictionaries(messages):
        kind = item.get("type", "message")
        # A role extension on a non-message item cannot change its translated role.
        role = item.get("role", "assistant") if kind == "message" else "assistant"
        if kind == "function_call":
            content = [
                {
                    **extras(item, "type", "call_id", "name", "arguments", "id", "status"),
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": arguments(item.get("arguments", {})),
                }
            ]
        elif kind == "function_call_output":
            role = "user"
            content = [
                {
                    **extras(item, "type", "call_id", "output"),
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id"),
                    "content": item.get("output"),
                }
            ]
        elif kind == "reasoning":
            content = [
                {
                    **extras(item, "type"),
                    "type": "thinking",
                    "thinking": "".join(b.get("text", "") for b in item.get("summary", [])),
                }
            ]
        elif kind == "message":
            content = from_chat_content(item.get("content"))
        else:
            result.append(remember({**item, OPAQUE: True}, [item], PROTOCOL))
            continue
        value = remember(
            {
                **(
                    extras(item, "type", "content", "status", "id")
                    if kind == "message"
                    else residual(item, response_fields(item))
                ),
                "role": role,
                "content": content,
            },
            [item],
            PROTOCOL,
            preserve=residual(item, response_fields(item)),
        )
        # Keep per-item metadata here; target decoders decide whether to group items.
        result.append(value)
    return result
