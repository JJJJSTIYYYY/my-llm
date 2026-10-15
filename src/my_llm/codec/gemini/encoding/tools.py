"""Gemini local function declarations and model call parts."""

import json
from typing import Any

from google.genai import types
from openai import Omit, omit

from my_llm.codec.utils.native import function_definitions
from my_llm.profile.utils.resolution import ResolvedProfile


def encode_tools(tools: Any, profile: ResolvedProfile) -> list[types.Tool] | Omit:
    definitions = function_definitions(tools, profile)
    if isinstance(definitions, Omit) or not definitions:
        return omit
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=f["name"],
                    description=f.get("description"),
                    parameters_json_schema=f.get("parameters"),
                )
                for f in definitions
            ]
        )
    ]


def encode_function_call(call: dict[str, Any]) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=call["id"],
            name=call["function"]["name"],
            args=json.loads(call["function"]["arguments"]),
        )
    )
