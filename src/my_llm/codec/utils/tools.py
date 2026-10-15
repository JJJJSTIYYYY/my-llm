"""Read-only validation of local function definitions in either public format."""

from typing import Any

from my_llm.types.exceptions import MessageValidationError


def _function_definition(tool: Any) -> tuple[bool, dict[str, Any]]:
    function_fields = {"name", "description", "parameters", "strict"}
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise MessageValidationError("only local function tools are supported")
    chat_format = "function" in tool
    if chat_format and set(tool) & function_fields:
        raise MessageValidationError("function tool mixes Responses and Chat Completions formats")
    function = (
        tool["function"]
        if chat_format
        else {key: value for key, value in tool.items() if key != "type"}
    )
    if (
        not isinstance(function, dict)
        or not isinstance(function.get("name"), str)
        or not function["name"]
    ):
        raise MessageValidationError("function tools need a non-empty name")
    if function.get("parameters") is not None and not isinstance(function["parameters"], dict):
        raise MessageValidationError("function parameters must be a JSON schema object")
    if function.get("description") is not None and not isinstance(function["description"], str):
        raise MessageValidationError("function description must be a string")
    if function.get("strict") is not None and not isinstance(function["strict"], bool):
        raise MessageValidationError("function strict must be a bool")
    return chat_format, function
