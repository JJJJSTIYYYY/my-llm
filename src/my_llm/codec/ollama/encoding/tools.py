"""ollama tools: encoding conversion."""

from typing import Any

from my_llm.codec.utils.tools import encode_functions
from my_llm.profile import ResolvedProfile


def encode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return encode_functions(
        tools,
        "ollama",
        lambda tool: (
            [tool["function"]] if tool.get("type") == "function" and "function" in tool else None
        ),
    )
