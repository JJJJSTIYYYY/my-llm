"""openai/responses tools: decoding conversion."""

from typing import Any

from my_llm.codec.utils.tools import decode_functions
from my_llm.profile import ResolvedProfile


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "openai_responses",
        lambda functions: [
            {"type": "function", **f, "strict": f.get("strict", False)} for f in functions
        ],
    )
