"""gemini tools: encoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import extras
from my_llm.codec.utils.tools import encode_functions
from my_llm.profile import ResolvedProfile


def encode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return encode_functions(
        tools,
        "gemini",
        lambda tool: (
            [
                {
                    **extras(f, "parameters_json_schema"),
                    "parameters": f.get("parameters_json_schema", f.get("parameters", {})),
                }
                for f in tool["function_declarations"]
            ]
            if "function_declarations" in tool
            else None
        ),
    )
