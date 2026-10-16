"""ollama messages: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    chat_call,
    coalesce_response_messages,
    extras,
    original,
    text,
    thinking_enabled,
    tool_output,
)
from my_llm.codec.utils.keys import OPAQUE
from my_llm.exceptions import MessageValidationError
from my_llm.profile import ResolvedProfile

PROTOCOL = "ollama"


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    names = {}
    for message in coalesce_response_messages(messages):
        if message.get(OPAQUE):
            result.append(extras(message, OPAQUE))
            continue
        content = message.get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
        native = original(message, PROTOCOL)
        if native is not None:
            result.extend(native)
            continue
        calls = []
        images = []
        reasoning = ""
        unknown = []
        results = []
        for block in content:
            kind = block.get("type")
            if kind == "tool_use":
                call = chat_call(block)
                call["function"]["arguments"] = block["input"]
                calls.append({"function": call["function"]})
            elif kind == "tool_result":
                results.append(
                    {
                        "role": "tool",
                        "tool_name": block.get("name") or names.get(block["tool_use_id"]),
                        "content": tool_output(block.get("content")),
                    }
                )
            elif kind == "thinking":
                reasoning += block.get("thinking", "")
            elif kind == "image":
                source = block.get("source", {})
                if source.get("type") != "base64":
                    raise MessageValidationError("Ollama images require base64 data")
                images.append(source.get("data"))
            elif kind != "text":
                unknown.append(block)
        if len(content) != len(results) or not results:
            value = {**extras(message, "content", "parts", "tool_calls"), "content": text(content)}
            if calls:
                value["tool_calls"] = calls
            if images:
                value["images"] = images
            if reasoning:
                value["thinking"] = reasoning
            if unknown:
                value["content"] = (
                    [{"type": "text", "text": value["content"]}] if value["content"] else []
                ) + unknown
            result.append(value)
        result.extend(results)
    if profile:
        for value in result:
            if value.get("role") != "assistant":
                continue
            if not profile.get("replay_reasoning_content_when_tool_calls", False):
                value.pop("thinking", None)
            elif value.get("tool_calls") and thinking_enabled(profile, parameters):
                # Only active tool-call turns need a missing reasoning field padded.
                value.setdefault("thinking", "")
    return result
