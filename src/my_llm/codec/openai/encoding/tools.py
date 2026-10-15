"""Identify and translate function templates, isolating the tool schema once."""

from copy import deepcopy
from typing import Any

from openai import Omit, omit

from my_llm.codec.utils.tools import _function_definition
from my_llm.profile import APIType
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types.exceptions import MessageValidationError, UnsupportedCapabilityError


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
