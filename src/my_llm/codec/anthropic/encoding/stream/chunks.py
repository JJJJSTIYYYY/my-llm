"""anthropic chunks: encoding conversion."""

from typing import Any

from my_llm.codec.anthropic.encoding.stream.usage import encode as encode_usage
from my_llm.codec.shared.chunks import BlockRef, batch
from my_llm.codec.shared.conversion import dictionaries
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"
REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def metadata(event: dict) -> dict:
    return {key: event[key] for key in ("id", "model") if key in event}


def consumed_fields(event: dict) -> dict:
    fields = {
        "message_start": {"message": dict.fromkeys(("id", "model"), True)},
        "content_block_start": {"index": True},
        "content_block_delta": {
            "index": True,
            "delta": {
                "type": True,
                "text": True,
                "thinking": True,
                "signature": True,
                "partial_json": True,
                "citation": True,
            },
        },
        "content_block_stop": {"index": True},
        "message_delta": {"delta": {"stop_reason": True, "stop_sequence": True}},
        "message_stop": {},
    }
    kind = event.get("type")
    return {"type": True, **fields[kind]} if kind in fields else {}


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        if event.get("type") == "error" or event.get("error"):
            raise ResponseFailedError(
                "provider generation failed", details=event.get("error", event)
            )
        kind = event.get("type")
        index = event.get("index", 0)
        ops = []
        if kind == "message_start":
            message = event["message"]
            ops.append({"type": "metadata", **metadata(message)})
            if message.get("usage") and (profile is None or profile.get("usage_mode") != "none"):
                ops.append({"type": "usage", "usage": message["usage"]})
        elif kind == "content_block_start":
            block = event["content_block"]
            ops.append({"type": "block_set", "ref": BlockRef("content", index), "block": block})
        elif kind == "content_block_delta":
            delta = event["delta"]
            if delta.get("type") == "citations_delta":
                ops.append(
                    {
                        "type": "block_annotation",
                        "ref": BlockRef("content", index),
                        "field": "citations",
                        "value": delta["citation"],
                    }
                )
            field = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "signature",
                "input_json_delta": "arguments",
            }.get(delta.get("type"))
            if field:
                ops.append(
                    {
                        "type": "block_delta",
                        "ref": BlockRef("content", index),
                        "field": field,
                        "kind": {"arguments": "tool_use", "signature": "thinking"}.get(
                            field, field
                        ),
                        "value": delta.get("partial_json" if field == "arguments" else field, ""),
                    }
                )
        elif kind == "content_block_stop":
            ops.append({"type": "block_stop", "ref": BlockRef("content", index)})
        elif kind == "message_delta":
            delta = event.get("delta", {})
            if delta.get("stop_reason"):
                reason = delta["stop_reason"]
                ops.append({"type": "reason", "reason": REASONS.get(reason, reason)})
                ops.append(
                    {
                        "type": "metadata",
                        "native_stop_reason": reason,
                        "native_stop_sequence": delta.get("stop_sequence"),
                    }
                )
            if event.get("usage") and (profile is None or profile.get("usage_mode") != "none"):
                ops.append({"type": "usage", "usage": event["usage"]})
        elif kind == "message_stop":
            ops.append({"type": "finish"})
        if not ops:
            ops = [{"type": "opaque", "value": event}]
        for operation in ops:
            if operation["type"] == "usage":
                operation["usage"] = encode_usage(operation["usage"])
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
