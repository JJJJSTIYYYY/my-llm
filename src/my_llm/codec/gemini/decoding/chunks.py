"""gemini chunks: decoding conversion."""

from typing import Any

from my_llm.codec.gemini.decoding import messages
from my_llm.codec.utils.chunks import events
from my_llm.codec.utils.usage import convert_usage
from my_llm.profile import ResolvedProfile

PROTOCOL = "gemini"


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = convert_usage(event.get("usage"), event.get("usage_protocol"), PROTOCOL)
        if kind == "opaque":
            result.append(event["value"])
            continue
        parts = []
        reason = None
        if kind == "delta":
            parts = [
                {
                    "text": event["value"],
                    **({"thought": True} if event["field"] == "thinking" else {}),
                }
            ]
        elif kind in ("tool", "block"):
            parts = [messages.decode_part(event["block"])]
        elif kind == "block_stop":
            signature = event["block"].get("thought_signature")
            if signature:
                parts = [
                    {
                        "text": "",
                        "thought_signature": signature,
                        **({"thought": True} if event["block"].get("type") == "thinking" else {}),
                    }
                ]
        elif kind == "finish":
            reason = "MAX_TOKENS" if event["reason"] == "length" else "STOP"
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
