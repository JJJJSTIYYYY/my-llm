"""Role-specific message conversion over a request-owned history snapshot.

Encoders create protocol wrappers without copying already-owned nested values.
"""

from typing import Any

from my_llm.codec.openai.encoding.content import encode_content
from my_llm.codec.utils.input import NormalizedMessage, normalize_messages
from my_llm.codec.utils.validation import validate_messages
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types import Message, ToolCall
from my_llm.types.exceptions import MessageValidationError


def encode_messages(
    messages: Any, profile: ResolvedProfile, *, allow_external_calls: bool = False
) -> list[dict[str, Any]]:
    normalized = normalize_messages(messages)
    validate_messages(
        [entry.message for entry in normalized], allow_external_calls=allow_external_calls
    )
    if profile.api_type == "responses":
        result = [
            item
            for entry in normalized
            for item in (
                _encode_native_output(entry.response_items)
                if entry.response_items is not None
                else encode_response_message(entry.message)
            )
        ]
    else:
        result = _encode_chat_history(normalized, profile)
    if not result:
        raise MessageValidationError("input has no messages representable by the selected API")
    return result


def _encode_chat_history(
    entries: list[NormalizedMessage], profile: ResolvedProfile
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []
    last_native = False
    for entry in entries:
        native = entry.response_items is not None
        message = entry.message
        if native and any(item.get("type") == "item_reference" for item in entry.response_items):
            raise MessageValidationError("Chat cannot resolve Responses item_reference items")
        if native and not message.get("tool_calls") and message.get("reasoning_content"):
            pending_reasoning.append(message["reasoning_content"])
        if native and message.get("content") is None and not message.get("tool_calls"):
            last_native = True
            continue
        context_native = native or bool(pending_reasoning)
        if pending_reasoning and message["role"] == "assistant" and message.get("tool_calls"):
            message = {
                **message,
                "reasoning_content": "".join(pending_reasoning)
                + message.get("reasoning_content", ""),
            }
            pending_reasoning.clear()
        elif message["role"] != "assistant":
            pending_reasoning.clear()
        encoded = encode_chat_message(message, profile)
        if (
            result
            and encoded["role"] == result[-1]["role"] == "assistant"
            and (context_native or last_native)
            and (encoded.get("tool_calls") or result[-1].get("tool_calls"))
        ):
            result[-1] = _merge_chat_assistant(result[-1], encoded)
            last_native = True
        else:
            result.append(encoded)
            last_native = context_native
    return result


def _merge_chat_assistant(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    content = first.get("content")
    added = second.get("content")
    if content is None:
        content = added
    elif added is not None:
        if isinstance(content, str) and isinstance(added, str):
            content += added
        else:
            before = [{"type": "text", "text": content}] if isinstance(content, str) else content
            after = [{"type": "text", "text": added}] if isinstance(added, str) else added
            content = [*before, *after]
    result = {
        **first,
        **second,
        "content": content,
        "tool_calls": [*first.get("tool_calls", []), *second.get("tool_calls", [])],
    }
    reasoning = first.get("reasoning_content", "") + second.get("reasoning_content", "")
    if reasoning:
        result["reasoning_content"] = reasoning
    return result


def encode_chat_message(message: Message, profile: ResolvedProfile) -> dict[str, Any]:
    encoded = {
        key: message[key]
        for key in (
            "role",
            "content",
            "tool_call_id",
            "tool_calls",
            "refusal",
            "audio",
            "function_call",
        )
        if key in message
    }
    if "content" in encoded:
        encoded["content"] = encode_content(encoded["content"], message["role"], "chat_completions")
    if profile.get("supports_message_name", False) and "name" in message:
        encoded["name"] = message["name"]
    if (
        message.get("tool_calls")
        and profile.get("replay_reasoning_content_when_tool_calls", False)
        and "reasoning_content" in message
    ):
        encoded["reasoning_content"] = message["reasoning_content"]
    return encoded


def encode_response_message(message: Message) -> list[dict[str, Any]]:
    if message["role"] == "tool":
        return [_encode_tool_result(message)]
    if message["role"] == "assistant":
        return _encode_assistant(message)
    return [_encode_content_message(message)]


def _encode_tool_result(message: Message) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": message["tool_call_id"],
        "output": encode_content(message["content"], "tool", "responses"),
    }


def _encode_assistant(message: Message) -> list[dict[str, Any]]:
    if message.get("response_output"):
        return _encode_native_output(message["response_output"])
    if message.get("audio") is not None or message.get("function_call") is not None:
        raise MessageValidationError(
            "Responses cannot represent Chat audio or legacy function_call history"
        )
    result = [_encode_content_message(message)] if message.get("content") is not None else []
    if message.get("refusal") is not None:
        if not isinstance(message["refusal"], str):
            raise MessageValidationError("refusal must be a string")
        result.append(
            {"role": "assistant", "content": [{"type": "refusal", "refusal": message["refusal"]}]}
        )
    result.extend(_encode_function_call(call) for call in message.get("tool_calls", []))
    return result


def _encode_native_output(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_encode_native_item(item) for item in items]


def _encode_native_item(item: dict[str, Any]) -> dict[str, Any]:
    kind = item.get("type")
    if kind in (None, "message") and "role" in item:
        content = encode_content(item.get("content"), item["role"], "responses")
        return item if content is item.get("content") else {**item, "content": content}
    if kind == "function_call_output":
        output = encode_content(item.get("output"), "tool", "responses")
        return item if output is item.get("output") else {**item, "output": output}
    if kind in ("reasoning", "function_call", "item_reference"):
        return item
    raise MessageValidationError(
        "only message, reasoning, function_call, function_call_output and item_reference replay items are supported"
    )


def _encode_function_call(call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call["id"],
        "name": call["function"]["name"],
        "arguments": call["function"]["arguments"],
    }


def _encode_content_message(message: Message) -> dict[str, Any]:
    return {
        "role": message["role"],
        "content": encode_content(message.get("content"), message["role"], "responses"),
    }
