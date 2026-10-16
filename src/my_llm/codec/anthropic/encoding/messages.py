"""anthropic messages: encoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    blocks,
    dictionaries,
    remember,
)
from my_llm.exceptions import MessageValidationError
from my_llm.profile import ResolvedProfile

PROTOCOL = "anthropic"


def encode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in dictionaries(messages):
        if (
            message.get("role") not in ("system", "user", "assistant")
            or "content" not in message
            or "parts" in message
        ):
            raise MessageValidationError("message does not use the bound anthropic protocol")
        result.append(
            remember({**message, "content": blocks(message.get("content"))}, [message], PROTOCOL)
        )
    return result
