"""openai/responses tools: encoding conversion."""

from typing import Any

from my_llm.codec.shared.conversion import extras
from my_llm.codec.shared.tools import encode_functions
from my_llm.profile.resolution import ResolvedProfile


def encode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return encode_functions(
        tools,
        "openai_responses",
        lambda tool: [extras(tool, "type")] if tool.get("type") == "function" else None,
        consumed={
            "type": True,
            "name": True,
            "description": True,
            "parameters": True,
            "strict": True,
        },
    )
