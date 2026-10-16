"""openai/chat messages: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    blocks,
    chat_call,
    chat_content,
    coalesce_response_messages,
    extras,
    original,
    thinking_enabled,
    tool_output,
)
from my_llm.codec.utils.keys import OPAQUE
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_chat"


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in coalesce_response_messages(messages):
        if message.get(OPAQUE):
            result.append(extras(message, OPAQUE))
            continue
        native = original(message, PROTOCOL)
        if native is not None:
            result.extend(native)
            continue
        content = blocks(message.get("content"))
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        other = [b for b in content if b.get("type") != "tool_result"]
        if other or not tool_results:
            value = {
                **extras(message, "content", "parts", "tool_calls"),
                "content": chat_content(other, role=message["role"]),
            }
            calls = [chat_call(b) for b in other if b.get("type") == "tool_use"]
            if calls:
                value["tool_calls"] = calls
            annotations = [annotation for b in other for annotation in b.get("annotations", [])]
            if annotations:
                value["annotations"] = annotations
            reasoning = "".join(b.get("thinking", "") for b in other if b.get("type") == "thinking")
            if reasoning:
                value["reasoning_content"] = reasoning
            result.append(value)
        result.extend(
            {
                **extras(b, "type", "tool_use_id", "name"),
                "role": "tool",
                "tool_call_id": b["tool_use_id"],
                "content": tool_output(b.get("content")),
            }
            for b in tool_results
        )
    if profile:
        for value in result:
            if not profile.get("supports_message_name", False):
                value.pop("name", None)
            if value.get("role") != "assistant":
                continue
            if not profile.get("replay_reasoning_content_when_tool_calls", False):
                value.pop("reasoning_content", None)
            elif value.get("tool_calls") and thinking_enabled(profile, parameters):
                # Only active tool-call turns need a missing reasoning field padded.
                value.setdefault("reasoning_content", "")
    return result
