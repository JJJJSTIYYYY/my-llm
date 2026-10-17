"""gemini tools: encoding conversion."""

from typing import Any

from my_llm.codec.gemini.encoding.tools.schema import json_schema
from my_llm.codec.shared.conversion import extras
from my_llm.codec.shared.tools import encode_functions
from my_llm.profile.resolution import ResolvedProfile


def encode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return encode_functions(
        tools,
        "gemini",
        lambda tool: (
            [
                {
                    **extras(f, "parameters_json_schema"),
                    "parameters": f["parameters_json_schema"]
                    if "parameters_json_schema" in f
                    else json_schema(f.get("parameters", {})),
                }
                for f in tool["function_declarations"]
            ]
            if "function_declarations" in tool
            else None
        ),
        consumed={
            "function_declarations": [
                {
                    "name": True,
                    "description": True,
                    "parameters": True,
                    "parameters_json_schema": True,
                }
            ]
        },
    )
