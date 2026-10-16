"""openai/responses tools: encoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import extras
from my_llm.codec.utils.tools import encode_functions
from my_llm.profile import ResolvedProfile


def encode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return encode_functions(
        tools,
        "openai_responses",
        lambda tool: [extras(tool, "type")] if tool.get("type") == "function" else None,
    )
