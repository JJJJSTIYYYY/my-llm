"""Ollama role-specific history encoding and local tool-result association."""

from typing import Any

from my_llm.codec.ollama.encoding.content import encode_content
from my_llm.codec.ollama.encoding.tools import encode_function_call
from my_llm.codec.utils.native import native_messages
from my_llm.profile.utils.resolution import ResolvedProfile


def encode_messages(messages: Any, profile: ResolvedProfile) -> list[dict[str, Any]]:
    history = []
    names = {}
    for message in native_messages(messages):
        if message["role"] == "assistant":
            encoded = _encode_assistant(message, profile)
            names.update({c["id"]: c["function"]["name"] for c in message.get("tool_calls", [])})
        elif message["role"] == "tool":
            encoded = _encode_tool_result(message, names)
        else:
            encoded = encode_content(message)
        history.append(encoded)
    return history


def _encode_assistant(message: dict[str, Any], profile: ResolvedProfile) -> dict[str, Any]:
    encoded = encode_content(message)
    if message.get("tool_calls"):
        encoded["tool_calls"] = [encode_function_call(call) for call in message["tool_calls"]]
        if profile.get("replay_reasoning_content_when_tool_calls", False) and message.get(
            "reasoning_content"
        ):
            encoded["thinking"] = message["reasoning_content"]
    return encoded


def _encode_tool_result(message: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    return {**encode_content(message), "tool_name": names[message["tool_call_id"]]}
