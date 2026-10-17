"""gemini tools: decoding conversion."""

from typing import Any

from my_llm.codec.shared.rendering import select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.profile.resolution import ResolvedProfile


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "gemini",
        lambda functions: [
            {
                "function_declarations": [
                    {
                        **select_present(f, "name"),
                        "description": f.get("description") or "",
                        "parameters_json_schema": f.get("parameters") or {"type": "object"},
                    }
                    for f in functions
                ]
            }
        ],
    )
