"""openai/chat tools: encoding conversion."""

from typing import Any

from my_llm.codec.shared.tools import encode_functions
from my_llm.profile.resolution import ResolvedProfile


def encode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return encode_functions(
        tools,
        "openai_chat",
        lambda tool: (
            [tool["function"]] if tool.get("type") == "function" and "function" in tool else None
        ),
        consumed={
            "type": True,
            "function": {"name": True, "description": True, "parameters": True, "strict": True},
        },
    )
