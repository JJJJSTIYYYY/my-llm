"""ollama chunks: encoding conversion."""

from typing import Any

from my_llm.codec.utils.chunks import BlockRef, batch, check_error
from my_llm.codec.utils.conversion import dictionaries, tool_block
from my_llm.codec.utils.preservation import CALL
from my_llm.profile import ResolvedProfile

PROTOCOL = "ollama"


def consumed_fields(event: dict) -> dict:
    return {
        "model": True,
        "done": True,
        "done_reason": True,
        "message": {"content": True, "thinking": True, "tool_calls": [CALL]},
        **{key: True for key in event if key.endswith(("_count", "_duration"))},
    }


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        check_error(event)
        if not any(key in event for key in ("message", "done")):
            result.append(batch(event, [{"type": "opaque", "value": event}], PROTOCOL))
            continue
        ops = [{"type": "metadata", "model": event.get("model")}]
        message = event.get("message", {})
        for field, kind in (("content", "text"), ("thinking", "thinking")):
            if message.get(field):
                ops.append(
                    {
                        "type": "block_delta",
                        "ref": BlockRef(kind),
                        "field": kind,
                        "kind": kind,
                        "value": message[field],
                    }
                )
        for call in message.get("tool_calls") or []:
            ops.append({"type": "block_append", "block": tool_block(call)})
        if event.get("done"):
            ops.append({"type": "finish", "reason": event.get("done_reason", "stop")})
            if profile is None or profile.get("usage_mode") != "none":
                ops.append(
                    {
                        "type": "usage",
                        "usage": {
                            k: v for k, v in event.items() if k.endswith(("_count", "_duration"))
                        },
                    }
                )
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
