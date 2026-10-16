"""ollama chunks: decoding conversion."""

from typing import Any

from my_llm.codec.utils.chunks import events
from my_llm.codec.utils.conversion import chat_call
from my_llm.codec.utils.usage import convert_usage
from my_llm.profile import ResolvedProfile

PROTOCOL = "ollama"


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = convert_usage(event.get("usage"), event.get("usage_protocol"), PROTOCOL)
        if kind == "opaque":
            result.append(event["value"])
            continue
        message = {"role": "assistant", "content": ""}
        if kind == "delta":
            message["content" if event["field"] == "text" else "thinking"] = event["value"]
        elif kind == "tool":
            block = event["block"]
            call = chat_call(block)
            call["function"]["arguments"] = block["input"]
            message["tool_calls"] = [{"function": call["function"]}]
        elif kind != "finish":
            continue
        value = {"model": parameters.get("model", ""), "message": message, "done": kind == "finish"}
        if kind == "finish":
            value["done_reason"] = "stop" if event["reason"] == "tool_calls" else event["reason"]
            value.update(usage or {})
        result.append(value)
    return result
