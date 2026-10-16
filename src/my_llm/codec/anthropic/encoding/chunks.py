"""anthropic chunks: encoding conversion."""

from typing import Any

from my_llm.codec.anthropic._constants import _REASONS
from my_llm.codec.utils.chunks import BlockRef, batch, check_error, metadata
from my_llm.codec.utils.conversion import dictionaries
from my_llm.profile import ResolvedProfile

PROTOCOL = "anthropic"


def consumed_fields(event: dict) -> dict:
    fields = {
        "message_start": {
            "message": dict.fromkeys(("id", "model", "created", "created_at", "response_id"), True)
        },
        "content_block_start": {"index": True},
        "content_block_delta": {
            "index": True,
            "delta": {
                "type": True,
                "text": True,
                "thinking": True,
                "signature": True,
                "partial_json": True,
            },
        },
        "content_block_stop": {"index": True},
        "message_delta": {"delta": {"stop_reason": True}},
        "message_stop": {},
    }
    kind = event.get("type")
    return {"type": True, **fields[kind]} if kind in fields else {}


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        check_error(event)
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
                ops.append({"type": "reason", "reason": _REASONS.get(reason, reason)})
            if event.get("usage") and (profile is None or profile.get("usage_mode") != "none"):
                ops.append({"type": "usage", "usage": event["usage"]})
        elif kind == "message_stop":
            ops.append({"type": "finish"})
        if not ops:
            ops = [{"type": "opaque", "value": event}]
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
