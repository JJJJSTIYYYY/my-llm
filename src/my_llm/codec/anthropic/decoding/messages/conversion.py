"""anthropic messages: decoding conversion."""

from typing import Any

from my_llm.codec.anthropic.decoding.messages.projection import foreign_block
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    extras,
    original,
    source_protocol,
)
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.rendering import project_message
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in coalesce_messages(messages):
        if message.get(OPAQUE):
            result.extend(original(message, PROTOCOL))
            continue
        native = original(message, PROTOCOL)
        if native is not None:
            result.extend(native)
        else:
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
            content = blocks(message.get("content"))
            result.append({**extras(message, "parts", "tool_calls"), "content": content})
    return result
