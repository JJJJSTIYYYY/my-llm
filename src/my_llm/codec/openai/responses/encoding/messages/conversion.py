"""openai/responses messages: encoding conversion."""

from typing import Any

from my_llm.codec.openai.responses.encoding.messages.content import encode_content
from my_llm.codec.openai.responses.encoding.messages.fields import response_fields
from my_llm.codec.shared.conversion import (
    arguments,
    dictionaries,
    extras,
    remember,
    residual,
)
from my_llm.codec.shared.keys import OPAQUE, SOURCE
from my_llm.profile.resolution import ResolvedProfile

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
                    "content": encode_content(item["output"])
                    if isinstance(item.get("output"), list)
                    else item.get("output"),
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
            content = encode_content(item.get("content"))
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
        value[SOURCE]["coalesce_assistant"] = True
        # Keep per-item metadata here; target decoders decide whether to group items.
        result.append(value)
    return result
