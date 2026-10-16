"""openai/chat chunks: encoding conversion."""

from typing import Any

from my_llm.codec.utils.chunks import batch, check_error, metadata
from my_llm.codec.utils.common import path_value
from my_llm.codec.utils.conversion import dictionaries
from my_llm.codec.utils.preservation import CALL
from my_llm.exceptions import ConfigurationError
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_chat"


def consumed_fields(event: dict) -> dict:
    return {
        **dict.fromkeys(("id", "model", "created", "created_at", "response_id"), True),
        "usage": {},
        "choices": [
            {
                "index": True,
                "finish_reason": True,
                "delta": {
                    "content": True,
                    "reasoning_content": True,
                    "tool_calls": [{**CALL, "index": True}],
                },
            }
        ],
    }


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        check_error(event)
        if not any(key in event for key in ("choices", "usage")):
            result.append(batch(event, [{"type": "opaque", "value": event}], PROTOCOL))
            continue
        ops = [{"type": "metadata", **metadata(event)}]
        if event.get("usage") is not None and (
            profile is None or profile.get("usage_mode") != "none"
        ):
            ops.append({"type": "usage", "usage": event["usage"]})
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ConfigurationError("only one generated candidate is supported")
            delta = choice.get("delta", {})
            for field, kind, default in (
                ("content_path", "text", "$.choices[0].delta.content"),
                ("reasoning_content_path", "thinking", "$.choices[0].delta.reasoning_content"),
            ):
                value = path_value(event, profile.values[field] if profile else default)
                if value:
                    ops.append(
                        {
                            "type": "delta",
                            "index": kind,
                            "field": "text" if kind == "text" else "thinking",
                            "value": value,
                            "kind": kind,
                        }
                    )
            for call in delta.get("tool_calls") or []:
                ops.append(
                    {
                        "type": "tool_delta",
                        "index": f"tool_{call.get('index', 0)}",
                        **{k: v for k, v in call.items() if k not in ("index", "type")},
                    }
                )
            if choice.get("finish_reason") is not None:
                ops.append({"type": "finish", "reason": choice["finish_reason"]})
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
