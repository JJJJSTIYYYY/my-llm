"""gemini tools: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import extras
from my_llm.codec.utils.tools import decode_functions
from my_llm.profile import ResolvedProfile


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "gemini",
        lambda functions: [
            {
                "function_declarations": [
                    {
                        **extras(f, "parameters", "strict"),
                        "description": f.get("description", ""),
                        "parameters_json_schema": f.get("parameters", {}),
                    }
                    for f in functions
                ]
            }
        ],
    )
