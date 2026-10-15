"""Anthropic role-specific history encoding and signed block replay."""

import json
from typing import Any

from my_llm.codec.anthropic.encoding.content import encode_content
from my_llm.codec.anthropic.encoding.tools import encode_function_call
from my_llm.codec.utils.native import native_messages, text_content, validate_replay_calls
from my_llm.types.exceptions import ConfigurationError


def encode_messages(messages: Any) -> tuple[list[dict[str, Any]], list[str]]:
    history = []
    system = []
    for message in native_messages(messages):
        if message["role"] == "system":
            system.append(text_content(message["content"]))
            continue
        encoded = encode_message(message)
        if history and history[-1]["role"] == encoded["role"]:
            history[-1]["content"] = [*history[-1]["content"], *encoded["content"]]
        else:
            history.append(encoded)
    if not history:
        raise ConfigurationError("Anthropic requires a non-system message")
    return history, system


def encode_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]
    if role == "tool":
        encoded = _encode_tool_result(message)
    elif role == "assistant":
        encoded = _encode_assistant(message)
    else:
        encoded = {"role": role, "content": encode_content(message.get("content"), role)}
    if not encoded["content"]:
        raise ConfigurationError("Anthropic messages must contain text or tool blocks")
    return encoded


def _encode_tool_result(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": text_content(message["content"]),
            }
        ],
    }


def _encode_assistant(message: dict[str, Any]) -> dict[str, Any]:
    native = message.get("response_output", [])
    if native and all(
        b.get("type") in ("thinking", "redacted_thinking", "text", "tool_use") for b in native
    ):
        blocks = _replay_output(message, native)
    else:
        blocks = encode_content(message.get("content"), "assistant")
        blocks.extend(encode_function_call(call) for call in message.get("tool_calls", []))
    return {"role": "assistant", "content": blocks}


def _replay_output(message: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_replay_calls(
        message,
        [
            {
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                },
            }
            for block in blocks
            if block["type"] == "tool_use"
        ],
    )
    return blocks
