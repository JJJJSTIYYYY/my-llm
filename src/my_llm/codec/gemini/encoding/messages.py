"""Gemini role-specific content history and signed model-part replay."""

import json
from typing import Any

from google.genai import types

from my_llm.codec.gemini.encoding.content import encode_content
from my_llm.codec.gemini.encoding.tools import encode_function_call
from my_llm.codec.utils.native import native_messages, text_content, validate_replay_calls
from my_llm.types.exceptions import ConfigurationError


def encode_messages(messages: Any) -> tuple[list[types.Content], list[str]]:
    history = []
    system = []
    names = {}
    for message in native_messages(messages):
        if message["role"] == "system":
            system.append(text_content(message["content"]))
            continue
        encoded = encode_message(message, names)
        names.update({c["id"]: c["function"]["name"] for c in message.get("tool_calls", [])})
        if history and history[-1].role == encoded.role:
            history[-1].parts.extend(encoded.parts)
        else:
            history.append(encoded)
    if not history:
        raise ConfigurationError("Gemini requires a non-system message")
    return history, system


def encode_message(message: dict[str, Any], names: dict[str, str]) -> types.Content:
    role = message["role"]
    if role == "tool":
        encoded = _encode_tool_result(message, names)
    elif role == "assistant":
        encoded = _encode_assistant(message)
    else:
        encoded = types.Content(role="user", parts=encode_content(message.get("content"), role))
    if not encoded.parts:
        raise ConfigurationError("Gemini messages must contain text or function parts")
    return encoded


def _encode_tool_result(message: dict[str, Any], names: dict[str, str]) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=message["tool_call_id"],
                    name=names[message["tool_call_id"]],
                    response={"result": text_content(message["content"])},
                )
            )
        ],
    )


def _encode_assistant(message: dict[str, Any]) -> types.Content:
    native = message.get("response_output", [])
    if native and all(block.get("type") == "gemini_part" for block in native):
        parts = _replay_output(message, native)
    else:
        parts = encode_content(message.get("content"), "assistant")
        parts.extend(encode_function_call(call) for call in message.get("tool_calls", []))
    return types.Content(role="model", parts=parts)


def _replay_output(message: dict[str, Any], output: list[dict[str, Any]]) -> list[types.Part]:
    parts = [types.Part.model_validate(block["part"]) for block in output]
    validate_replay_calls(
        message,
        [
            {
                "id": part.function_call.id,
                "type": "function",
                "function": {
                    "name": part.function_call.name,
                    "arguments": json.dumps(part.function_call.args or {}),
                },
            }
            for part in parts
            if part.function_call is not None
        ],
    )
    return parts
