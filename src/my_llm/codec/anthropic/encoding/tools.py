"""Anthropic local function declarations and assistant tool-use blocks."""

import json
from typing import Any

from anthropic import omit
from openai import Omit

from my_llm.codec.utils.native import function_definitions
from my_llm.profile.utils.resolution import ResolvedProfile


def encode_tools(tools: Any, profile: ResolvedProfile) -> Any:
    definitions = function_definitions(tools, profile)
    if isinstance(definitions, Omit):
        return omit
    return [
        {
            "name": f["name"],
            "input_schema": f.get("parameters", {"type": "object"}),
            **({"description": f["description"]} if "description" in f else {}),
        }
        for f in definitions
    ]


def encode_function_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": call["id"],
        "name": call["function"]["name"],
        "input": json.loads(call["function"]["arguments"]),
    }
