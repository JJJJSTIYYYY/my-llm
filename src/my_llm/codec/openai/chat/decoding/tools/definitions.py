"""openai/chat tools: decoding conversion."""

from typing import Any

from my_llm.codec.shared.rendering import select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.profile.resolution import ResolvedProfile


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "openai_chat",
        lambda functions: [
            {
                "type": "function",
                "function": select_present(f, "name", "description", "parameters", "strict"),
            }
            for f in functions
        ],
    )
