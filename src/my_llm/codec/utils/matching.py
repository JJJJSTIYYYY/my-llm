"""Whole-list protocol recognition. Overlapping text-only inputs prefer Chat."""

from typing import Any

from my_llm.codec.utils.conversion import dictionaries
from my_llm.exceptions import MessageValidationError
from my_llm.profile.utils.common import PROTOCOLS


def candidates(values: list[Any]) -> set[str]:
    try:
        items = dictionaries(values)
    except MessageValidationError:
        return set()
    if not items:
        return set()
    possible = set(PROTOCOLS)
    for item in items:
        if "parts" in item or "function_declarations" in item:
            possible &= {"gemini"}
        elif "input_schema" in item:
            possible &= {"anthropic"}
        elif item.get("type") == "function":
            possible &= {"openai_chat", "ollama"} if "function" in item else {"openai_responses"}
        elif item.get("type") in (
            "message",
            "function_call",
            "function_call_output",
            "reasoning",
            "item_reference",
        ):
            possible &= {"openai_responses"}
        elif "role" in item:
            role = item["role"]
            if role not in ("user", "assistant", "system", "developer", "tool"):
                return set()
            calls = item.get("tool_calls")
            if calls is None:
                calls = []
            if not isinstance(calls, list) or any(
                not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                for call in calls
            ):
                return set()
            content = item.get("content")
            if isinstance(content, list) and any(not isinstance(block, dict) for block in content):
                return set()
            if any(k in item for k in ("thinking", "images", "tool_name")) or any(
                isinstance(c.get("function", {}).get("arguments"), dict) for c in calls
            ):
                possible &= {"ollama"}
            elif any(k in item for k in ("tool_calls", "tool_call_id", "reasoning_content")):
                possible &= {"openai_chat"}
            elif isinstance(item.get("content"), list):
                kinds = {b["type"] for b in content if isinstance(b.get("type"), str)}
                if kinds & {
                    "thinking",
                    "redacted_thinking",
                    "tool_use",
                    "tool_result",
                    "image",
                    "document",
                }:
                    possible &= {"anthropic"}
                elif kinds & {"input_text", "output_text", "input_image", "input_file"}:
                    possible &= {"openai_responses"}
                elif kinds & {"image_url", "file", "input_audio"}:
                    possible &= {"openai_chat"}
            possible.discard("gemini")
            if role in ("tool", "developer"):
                possible.discard("anthropic")
                possible.discard("openai_responses")
        else:
            return set()
    return possible
