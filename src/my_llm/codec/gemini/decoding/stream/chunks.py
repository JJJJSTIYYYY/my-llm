"""gemini chunks: decoding conversion."""

from typing import Any

from my_llm.codec.gemini.decoding.messages import conversion as messages
from my_llm.codec.gemini.decoding.messages.projection import foreign_block
from my_llm.codec.gemini.decoding.stream.usage import decode as decode_usage
from my_llm.codec.shared.chunks import events
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "gemini"


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = decode_usage(event.get("usage"))
        if kind == "opaque":
            result.append(event["value"])
            continue
        parts = []
        reason = None
        if kind == "delta":
            parts = [
                {
                    "text": event["value"],
                    **(
                        {"thought": True}
                        if event["field"] == "thinking"
                        and parameters.get("source_protocol") == PROTOCOL
                        else {}
                    ),
                }
            ]
        elif kind in ("tool", "block"):
            block = event["block"]
            foreign = parameters.get("source_protocol") != PROTOCOL
            if foreign:
                block = foreign_block(block)
                if block is None:
                    continue
            parts = [messages.decode_part(block, foreign=foreign)]
        elif kind == "block_stop":
            signature = event["block"].get("thought_signature")
            if signature and parameters.get("source_protocol") == PROTOCOL:
                parts = [
                    {
                        "text": "",
                        "thought_signature": signature,
                        **({"thought": True} if event["block"].get("type") == "thinking" else {}),
                    }
                ]
        elif kind == "finish":
            reason = {
                "length": "MAX_TOKENS",
                "content_filter": "SAFETY",
                "stop": "STOP",
                "tool_calls": "STOP",
            }.get(event["reason"], "OTHER")
        if parts or reason:
            value = {
                "response_id": parameters.get("response_id", "generated"),
                "model_version": parameters.get("model", ""),
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": parts},
                        **({"finish_reason": reason} if reason else {}),
                    }
                ],
            }
            if kind == "finish" and usage is not None:
                value["usage_metadata"] = usage
            result.append(value)
    return result
