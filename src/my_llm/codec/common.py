"""Small path reader and validation shared by OpenAI codecs."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from openai import Omit

from ..exceptions import ConfigurationError, IncompleteStreamError, MessageValidationError
from ..profiles import EFFORTS, ResolvedProfile, merge_dicts, validate_max_tokens
from ..types import Message, ToolCall


def read(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return deepcopy(value)


_PATH = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(?:\[(\d+)\])?")


def path_value(value: Any, path: str | Omit) -> Any:
    """Read dotted attributes/mapping keys and numeric indexes, without dumping SDK objects."""
    if isinstance(path, Omit):
        return None
    if path.startswith("$."):
        path = path[2:]
    for component in path.split("."):
        match = _PATH.fullmatch(component)
        if not match:
            raise ConfigurationError(f"unsupported path syntax: {path!r}")
        value = read(value, match[1])
        if value is None:
            return None
        if match[2] is not None:
            index = int(match[2])
            if not isinstance(value, (list, tuple)) or index >= len(value):
                return None
            value = value[index]
    return value


def validate_tools(tools: Any, profile: ResolvedProfile) -> list[dict[str, Any]] | Omit:
    """Identify each function template and encode it for the selected API.

    Native templates pass through unchanged; cross-API templates are flattened
    or wrapped. Copies protect the caller's schema from later request mutation.
    """
    from openai import omit

    if tools is None or isinstance(tools, Omit):
        return omit
    if not isinstance(tools, list):
        raise MessageValidationError(
            "tools must be a list of Responses or Chat Completions function tools"
        )
    if tools and not profile.get("supports_function_calling", False):
        from ..exceptions import UnsupportedCapabilityError

        raise UnsupportedCapabilityError("model does not support function calling")
    result = []
    function_fields = {"name", "description", "parameters", "strict"}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise MessageValidationError("only local function tools are supported")
        chat_format = "function" in tool
        if chat_format and set(tool) & function_fields:
            raise MessageValidationError(
                "function tool mixes Responses and Chat Completions formats"
            )
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
        if profile.api_type == "responses" and chat_format:
            outer = {key: value for key, value in tool.items() if key not in ("type", "function")}
            if set(outer) & set(function) or "type" in function:
                raise MessageValidationError("function tool contains conflicting fields")
            result.append({"type": "function", **deepcopy(outer), **deepcopy(function)})
        elif profile.api_type == "chat_completions" and not chat_format:
            result.append({"type": "function", "function": deepcopy(function)})
        else:
            result.append(deepcopy(tool))
    return result


def validate_tool_call(call: Any, *, streamed: bool = False) -> ToolCall:
    error = IncompleteStreamError if streamed else MessageValidationError
    if not isinstance(call, dict) or call.get("type") != "function":
        raise error("only complete function calls are supported")
    function = call.get("function")
    if not isinstance(call.get("id"), str) or not call["id"] or not isinstance(function, dict):
        raise error("function call is missing its ID or function")
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise error("function call is missing its name")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise error("function arguments must be a complete JSON string")
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError) as exc:
        raise error("function arguments are incomplete or invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise error("function arguments must encode a JSON object")
    return deepcopy(call)


def validate_messages(messages: Any, *, allow_external_calls: bool = False) -> list[Message]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise MessageValidationError("messages must be a non-empty list")
    result = []
    known_calls: set[str] = set()
    used_calls: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "system",
            "user",
            "assistant",
            "tool",
        ):
            raise MessageValidationError("message role must be system, user, assistant or tool")
        role = message["role"]
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise MessageValidationError("message content must be a string, content blocks or None")
        if isinstance(content, list) and any(not isinstance(block, dict) for block in content):
            raise MessageValidationError("content blocks must be dictionaries")
        if role in ("user", "system", "tool") and content is None:
            raise MessageValidationError(f"{role} messages require content")
        for key in ("name", "reasoning_content", "tool_call_id"):
            if key in message and not isinstance(message[key], str):
                raise MessageValidationError(f"{key} must be a string")
        if message.get("reasoning_content") is not None and role != "assistant":
            raise MessageValidationError("reasoning_content belongs to assistant messages")
        if message.get("tool_calls") is not None:
            if role != "assistant" or not isinstance(message["tool_calls"], list):
                raise MessageValidationError("tool_calls belong to assistant messages")
            for call in message["tool_calls"]:
                valid = validate_tool_call(call)
                if valid["id"] in known_calls:
                    raise MessageValidationError("duplicate function call ID in history")
                known_calls.add(valid["id"])
        output = message.get("response_output")
        if output is not None:
            if (
                role != "assistant"
                or not isinstance(output, list)
                or any(not isinstance(item, dict) for item in output)
            ):
                raise MessageValidationError("response_output must be assistant output items")
            native_calls: dict[str, ToolCall] = {}
            for item in output:
                if item.get("type") == "function_call":
                    native = validate_tool_call(
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments"),
                            },
                        }
                    )
                    call_id = native["id"]
                    if call_id in native_calls:
                        raise MessageValidationError("duplicate native function call ID")
                    native_calls[call_id] = native
            if (
                message.get("tool_calls")
                and {call["id"]: call for call in message["tool_calls"]} != native_calls
            ):
                raise MessageValidationError(
                    "unified tool_calls must match native response_output calls"
                )
            uniform_ids = {call["id"] for call in message.get("tool_calls", [])}
            if set(native_calls) & (known_calls - uniform_ids):
                raise MessageValidationError("duplicate native function call ID in history")
            known_calls.update(native_calls)
        if role == "tool":
            call_id = message.get("tool_call_id")
            if (
                not call_id
                or (call_id not in known_calls and not allow_external_calls)
                or call_id in used_calls
            ):
                raise MessageValidationError(
                    "tool result must reference a preceding, unanswered call ID"
                )
            used_calls.add(call_id)
        elif "tool_call_id" in message:
            raise MessageValidationError("tool_call_id belongs to tool messages")
        result.append(deepcopy(message))
    return result


def _expand(value: Any, effort: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand(item, effort) for key, item in value.items() if not isinstance(item, Omit)
        }
    if isinstance(value, list):
        return [_expand(item, effort) for item in value]
    return effort if value == "{effort}" else deepcopy(value)


def request_parameters(
    profile: ResolvedProfile,
    *,
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = None,
    reasoning_effort: str = "none",
    reasoning: bool | None = None,
) -> dict[str, Any]:
    from ..exceptions import UnsupportedCapabilityError

    if reasoning_effort not in EFFORTS:
        raise ConfigurationError(f"unsupported reasoning_effort: {reasoning_effort!r}")
    if reasoning is not None and not isinstance(reasoning, bool):
        raise ConfigurationError("reasoning must be a bool")
    if reasoning is False and reasoning_effort != "none":
        raise ConfigurationError("reasoning=False conflicts with a non-none effort")
    if reasoning is True and reasoning_effort == "none":
        reasoning_effort = "medium"
    headers = merge_dicts(profile.values["extra_headers"], extra_headers)
    body = merge_dicts(profile.values["extra_body"], extra_body)
    if set(body) & {"model", "stream", "messages", "input", "tools", "previous_response_id"}:
        raise ConfigurationError(
            "extra_body cannot override model, stream, messages, input, tools or previous_response_id"
        )
    kwargs: dict[str, Any] = {}
    if reasoning_effort != "none":
        if not profile.get("supports_reasoning", False):
            raise UnsupportedCapabilityError("model does not support reasoning")
        bindings = [profile.get("reasoning_enable_parameter", {})]
        mapped = reasoning_effort
        if profile.get("supports_reasoning_effort", False):
            mapped = profile.get("reasoning_effort_mapping", {}).get(reasoning_effort)
            if not isinstance(mapped, str) or not mapped:
                raise UnsupportedCapabilityError(
                    f"model does not support effort {reasoning_effort!r}"
                )
            bindings.append(profile.get("reasoning_effort_parameter", {}))
        for binding_index, binding in enumerate(bindings):
            if set(binding) - {"reasoning", "reasoning_effort", "extra_body"}:
                raise ConfigurationError(
                    "reasoning bindings may target reasoning, reasoning_effort or extra_body"
                )
            for key, raw_value in binding.items():
                if isinstance(raw_value, Omit):
                    continue
                value = _expand(raw_value, mapped)
                if key == "extra_body":
                    if isinstance(value, dict) and set(value) & {
                        "model",
                        "stream",
                        "messages",
                        "input",
                        "tools",
                        "previous_response_id",
                    }:
                        raise ConfigurationError(
                            "reasoning extra_body cannot override protocol parameters"
                        )
                    body = merge_dicts(body, value)
                elif key == "reasoning":
                    if profile.api_type != "responses":
                        raise ConfigurationError("reasoning SDK parameter requires Responses API")
                    combined = merge_dicts({"summary": "detailed"}, kwargs.get(key), raw_value)
                    if binding_index == 1 and "effort" not in raw_value:
                        combined["effort"] = mapped
                    kwargs[key] = _expand(combined, mapped)
                else:
                    if profile.api_type != "chat_completions" or not isinstance(value, str):
                        raise ConfigurationError(
                            "reasoning_effort SDK parameter requires Chat Completions"
                        )
                    kwargs[key] = value
    tokens = profile.get("max_tokens") if max_tokens is None else max_tokens
    if isinstance(tokens, Omit):
        tokens = None
    validate_max_tokens(tokens)
    if tokens is not None:
        kwargs[
            "max_output_tokens" if profile.api_type == "responses" else "max_completion_tokens"
        ] = tokens
    # SDK extra_body takes precedence over named kwargs: generated fields must win.
    for key, value in kwargs.items():
        if key in body:
            body[key] = deepcopy(value)
    if headers:
        kwargs["extra_headers"] = headers
    if body:
        kwargs["extra_body"] = body
    if profile.get("usage_mode", "none") == "stream_options":
        kwargs["stream_options"] = {"include_usage": True}
        if "stream_options" in body:
            body["stream_options"] = {"include_usage": True}
    return kwargs


def text_delta(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise IncompleteStreamError("configured text path did not return a string")
    return value
