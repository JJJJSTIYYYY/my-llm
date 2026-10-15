"""Ollama local function declarations and native call objects."""

import json
from typing import Any

from openai import Omit, omit

from my_llm.codec.utils.native import function_definitions
from my_llm.profile.utils.resolution import ResolvedProfile


def encode_tools(tools: Any, profile: ResolvedProfile) -> list[dict[str, Any]] | Omit:
    definitions = function_definitions(tools, profile)
    if isinstance(definitions, Omit):
        return omit
    return [{"type": "function", "function": f} for f in definitions]


def encode_function_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "function": {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
        }
    }
