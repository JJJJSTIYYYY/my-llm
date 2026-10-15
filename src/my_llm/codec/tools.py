"""Identify and translate function templates, isolating the tool schema once."""

from copy import deepcopy
from typing import Any

from openai import Omit, omit

from my_llm.exceptions import MessageValidationError, UnsupportedCapabilityError
from my_llm.profiles import APIType, ResolvedProfile


def encode_tools(tools: Any, profile: ResolvedProfile) -> list[dict[str, Any]] | Omit:
    """Identify each function template and encode it for the selected API.

    Native templates pass through unchanged; cross-API templates are flattened
    or wrapped. Copies protect the caller's schema from later request mutation.
    """
    if tools is None or isinstance(tools, Omit):
        return omit
    if not isinstance(tools, list):
        raise MessageValidationError(
            "tools must be a list of Responses or Chat Completions function tools"
        )
    if tools and not profile.get("supports_function_calling", False):
        raise UnsupportedCapabilityError("model does not support function calling")
    return deepcopy([_encode_tool(tool, profile.api_type) for tool in tools])


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


def _encode_tool(tool: Any, api_type: APIType) -> dict[str, Any]:
    chat_format, function = _function_definition(tool)
    if api_type == "responses" and chat_format:
        outer = {key: value for key, value in tool.items() if key not in ("type", "function")}
        if set(outer) & set(function) or "type" in function:
            raise MessageValidationError("function tool contains conflicting fields")
        return {"type": "function", **outer, **function}
    if api_type == "chat_completions" and not chat_format:
        return {"type": "function", "function": function}
    return tool
