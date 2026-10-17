"""ollama decoding: content, fields and tool helpers."""

from typing import Any

from ollama import Tool
from pydantic import ValidationError

from my_llm.codec.shared.conversion import extras
from my_llm.codec.shared.rendering import project_block, select_present, unsupported
from my_llm.exceptions import UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
    "thinking": ("type", "thinking"),
}


def foreign_block(block: dict) -> dict | None:
    if block.get("type") in ("image", "document"):
        source = block.get("source", {})
        if source.get("type") == "url":
            return {"type": "text", "text": source["url"]}
        if block["type"] == "document":
            return unsupported(block)
    return project_block(
        block, FIELDS, text_fields={"refusal": "refusal"}, omitted=("redacted_thinking",)
    )


def thinking_enabled(profile: Any, parameters: dict) -> bool:
    if parameters.get("reasoning") is True or parameters.get("reasoning_effort", "none") != "none":
        return True
    body = merge_dicts(
        profile.get("extra_body", {}) if profile else {}, parameters.get("extra_body")
    )
    return body.get("think") not in (None, False)


def chat_call(block: dict) -> dict:
    return {
        "function": {
            **extras(block.get("function", {}), "index"),
            "name": block["name"],
            "arguments": block["input"],
        }
    }


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
