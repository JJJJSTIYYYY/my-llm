"""openai/chat chunks: decoding conversion."""

from typing import Any

from my_llm.codec.utils.chunks import envelope, events
from my_llm.codec.utils.conversion import chat_call
from my_llm.codec.utils.usage import convert_usage
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_chat"


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = convert_usage(event.get("usage"), event.get("usage_protocol"), PROTOCOL)
        if kind == "opaque":
            result.append(event["value"])
            continue
        delta = {}
        finish = None
        if kind == "start":
            delta = {"role": "assistant"}
        elif kind == "delta":
            if event["field"] in ("text", "thinking"):
                delta = {
                    "content" if event["field"] == "text" else "reasoning_content": event["value"]
                }
        elif kind == "tool":
            delta = {
                "tool_calls": [{**chat_call(event["block"]), "index": event.get("tool_index", 0)}]
            }
        elif kind == "finish":
            finish = event["reason"]
        if not delta and kind != "finish":
            continue
        value = {
            **envelope(**parameters),
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        value["usage"] = None
        result.append(value)
        if kind == "finish" and usage is not None:
            result.append({**value, "choices": [], "usage": usage})
    return result
