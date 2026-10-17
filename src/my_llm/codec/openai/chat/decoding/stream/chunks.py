"""openai/chat chunks: decoding conversion."""

from typing import Any

from my_llm.codec.openai.chat.decoding.messages.projection import foreign_block
from my_llm.codec.openai.chat.decoding.stream.usage import decode as decode_usage
from my_llm.codec.openai.chat.decoding.tools.calls import chat_call
from my_llm.codec.shared.chunks import events
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"


def envelope(**parameters: Any) -> dict:
    return {
        "id": parameters.get("response_id", "generated"),
        "model": parameters.get("model", ""),
        "created": parameters.get("created", 0),
    }


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = decode_usage(event.get("usage"))
        if kind == "opaque":
            result.append(event["value"])
            continue
        delta = {}
        finish = None
        if kind == "start":
            delta = {"role": "assistant"}
        elif kind == "delta":
            field = {"text": "content", "thinking": "reasoning_content", "refusal": "refusal"}.get(
                event["field"]
            )
            if event["field"] == "thinking" and parameters.get("source_protocol") != PROTOCOL:
                field = "content"
            if field:
                delta = {field: event["value"]}
        elif kind == "tool":
            block = event["block"]
            if parameters.get("source_protocol") != PROTOCOL:
                block = foreign_block(block)
            delta = {"tool_calls": [{**chat_call(block), "index": event.get("tool_index", 0)}]}
        elif kind == "finish":
            finish = event["reason"]
            if finish not in ("stop", "length", "tool_calls", "content_filter", "function_call"):
                raise ResponseFailedError(
                    "Chat cannot represent the source finish reason", details={"reason": finish}
                )
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
