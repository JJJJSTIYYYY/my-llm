"""ollama chunks: decoding conversion."""

from typing import Any

from my_llm.codec.ollama.decoding.messages.projection import foreign_block
from my_llm.codec.ollama.decoding.stream.usage import decode as decode_usage
from my_llm.codec.ollama.decoding.tools.calls import chat_call
from my_llm.codec.shared.chunks import events
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = decode_usage(event.get("usage"))
        if kind == "opaque":
            result.append(event["value"])
            continue
        message = {"role": "assistant", "content": ""}
        if kind == "delta":
            message["thinking" if event["field"] == "thinking" else "content"] = event["value"]
        elif kind == "tool":
            block = event["block"]
            if parameters.get("source_protocol") != PROTOCOL:
                block = foreign_block(block)
            call = chat_call(block)
            message["tool_calls"] = [{"function": call["function"]}]
        elif kind != "finish":
            continue
        value = {"model": parameters.get("model", ""), "message": message, "done": kind == "finish"}
        if kind == "finish":
            value["done_reason"] = "stop" if event["reason"] == "tool_calls" else event["reason"]
            value.update(usage or {})
        result.append(value)
    return result
