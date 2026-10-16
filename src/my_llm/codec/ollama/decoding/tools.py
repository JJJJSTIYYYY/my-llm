"""ollama tools: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import extras
from my_llm.codec.utils.tools import decode_functions
from my_llm.profile import ResolvedProfile


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "ollama",
        lambda functions: [
            {"type": "function", "function": extras(f, "strict")} for f in functions
        ],
    )
