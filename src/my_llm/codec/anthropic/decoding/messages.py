"""anthropic messages: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    blocks,
    coalesce_response_messages,
    extras,
    original,
    tool_output,
)
from my_llm.codec.utils.keys import OPAQUE
from my_llm.profile import ResolvedProfile

PROTOCOL = "anthropic"


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in coalesce_response_messages(messages):
        if message.get(OPAQUE):
            result.append(extras(message, OPAQUE))
            continue
        native = original(message, PROTOCOL)
        if native is not None:
            result.extend(native)
        else:
            content = blocks(message.get("content"))
            for block in content:
                block.pop("thought_signature", None)
                if block.get("type") == "text":
                    block.pop("annotations", None)
                if block.get("type") in ("image", "document"):
                    block.pop("detail", None)
                if block.get("type") == "tool_result":
                    block.pop("name", None)
                    block["content"] = tool_output(block.get("content"))
            result.append({**extras(message, "parts", "tool_calls"), "content": content})
    return result
