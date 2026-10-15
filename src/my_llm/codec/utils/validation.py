"""Read-only validation for unified messages and function-call history."""

import json
from typing import Any

from my_llm.types import Message, ToolCall
from my_llm.types.exceptions import IncompleteStreamError, MessageValidationError


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
    return call


def validate_messages(messages: Any, *, allow_external_calls: bool = False) -> list[Message]:
    """Validate history without copying or mutating its message dictionaries."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise MessageValidationError("messages must be a non-empty list")
    result = []
    known_calls: set[str] = set()
    used_calls: set[str] = set()
    for message in messages:
        _validate_message_fields(message)
        calls = _validate_message_calls(message)
        if set(calls) & known_calls:
            raise MessageValidationError("duplicate function call ID in history")
        native = _validate_native_calls(message)
        if native is not None:
            if calls and calls != native:
                raise MessageValidationError(
                    "unified tool_calls must match native response_output calls"
                )
            if set(native) & known_calls:
                raise MessageValidationError("duplicate native function call ID in history")
            known_calls.update(native)
        known_calls.update(calls)
        _validate_tool_result(message, known_calls, used_calls, allow_external_calls)
        result.append(message)
    return result


def _validate_message_fields(message: Any) -> None:
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
    if role != "tool" and "tool_call_id" in message:
        raise MessageValidationError("tool_call_id belongs to tool messages")


def _validate_message_calls(message: Message) -> dict[str, ToolCall]:
    calls = message.get("tool_calls")
    if calls is None:
        return {}
    if message["role"] != "assistant" or not isinstance(calls, list):
        raise MessageValidationError("tool_calls belong to assistant messages")
    result = {}
    for call in calls:
        valid = validate_tool_call(call)
        if valid["id"] in result:
            raise MessageValidationError("duplicate function call ID in history")
        result[valid["id"]] = valid
    return result


def _validate_native_calls(message: Message) -> dict[str, ToolCall] | None:
    output = message.get("response_output")
    if output is None:
        return None
    if (
        message["role"] != "assistant"
        or not isinstance(output, list)
        or any(not isinstance(item, dict) for item in output)
    ):
        raise MessageValidationError("response_output must be assistant output items")
    result = {}
    for item in output:
        if item.get("type") != "function_call":
            continue
        call = validate_tool_call(
            {
                "id": item.get("call_id"),
                "type": "function",
                "function": {"name": item.get("name"), "arguments": item.get("arguments")},
            }
        )
        if call["id"] in result:
            raise MessageValidationError("duplicate native function call ID")
        result[call["id"]] = call
    return result


def _validate_tool_result(
    message: Message, known_calls: set[str], used_calls: set[str], allow_external_calls: bool
) -> None:
    if message["role"] != "tool":
        return
    call_id = message.get("tool_call_id")
    if (
        not call_id
        or (call_id not in known_calls and not allow_external_calls)
        or call_id in used_calls
    ):
        raise MessageValidationError("tool result must reference a preceding, unanswered call ID")
    used_calls.add(call_id)
