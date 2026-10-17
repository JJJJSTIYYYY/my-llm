"""ollama tools: decoding conversion."""

from typing import Any

from ollama import Tool
from pydantic import ValidationError

from my_llm.codec.shared.rendering import select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.exceptions import UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile


def validate_schema(parameters: dict) -> dict:
    """Do not let the official SDK silently discard JSON Schema constraints."""
    try:
        parsed = Tool(function={"name": "schema_check", "parameters": parameters}).model_dump(
            exclude_none=True
        )["function"]["parameters"]
    except (ValidationError, KeyError) as exc:
        raise UnsupportedCapabilityError(
            "installed Ollama SDK cannot represent the tool parameter schema"
        ) from exc

    def preserves(expected, actual):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and preserves(value, actual[key]) for key, value in expected.items()
            )
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(expected) == len(actual)
                and all(preserves(left, right) for left, right in zip(expected, actual))
            )
        return expected == actual

    if not preserves(parameters, parsed):
        raise UnsupportedCapabilityError(
            "installed Ollama SDK would discard tool parameter schema constraints"
        )
    return parameters


def function_tool(function: dict) -> dict:
    return {
        "type": "function",
        "function": {
            **select_present(function, "name", "description"),
            "parameters": validate_schema(function.get("parameters") or {"type": "object"}),
        },
    }


def decode(tools: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    return decode_functions(
        tools,
        "ollama",
        lambda functions: [function_tool(function) for function in functions],
    )
