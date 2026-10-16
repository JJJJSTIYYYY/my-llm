"""openai/chat tools: decoding conversion."""

from typing import Any

from my_llm.codec.utils.tools import decode_functions
from my_llm.profile import ResolvedProfile


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "openai_chat",
        lambda functions: [{"type": "function", "function": f} for f in functions],
    )
