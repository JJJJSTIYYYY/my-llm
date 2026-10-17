"""openai/chat chunks: encoding conversion."""

from typing import Any

from my_llm.codec.openai.chat.encoding.messages.fields import CALL
from my_llm.codec.openai.chat.encoding.stream.usage import encode as encode_usage
from my_llm.codec.openai.chat.encoding.tools.calls import tool_operations
from my_llm.codec.shared.chunks import BlockRef, batch
from my_llm.codec.shared.conversion import dictionaries
from my_llm.codec.shared.values import path_value
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"


def metadata(event: dict) -> dict:
    return {key: event[key] for key in ("id", "model", "created") if key in event}


def consumed_fields(event: dict) -> dict:
    return {
        **dict.fromkeys(("id", "model", "created"), True),
        "usage": {},
        "choices": [
            {
                "index": True,
                "finish_reason": True,
                "delta": {
                    "content": True,
                    "reasoning_content": True,
                    "refusal": True,
                    "tool_calls": [{**CALL, "index": True}],
                },
            }
        ],
    }


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        if event.get("type") == "error" or event.get("error"):
            raise ResponseFailedError(
                "provider generation failed", details=event.get("error", event)
            )
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
            if delta.get("refusal") is not None:
                ops.append(
                    {
                        "type": "block_delta",
                        "ref": BlockRef("refusal"),
                        "field": "refusal",
                        "kind": "refusal",
                        "value": delta["refusal"],
                    }
                )
            for field, kind, default in (
                ("content_path", "text", "$.choices[0].delta.content"),
                ("reasoning_content_path", "thinking", "$.choices[0].delta.reasoning_content"),
            ):
                value = path_value(event, profile.values[field] if profile else default)
                if value:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef(kind),
                            "field": "text" if kind == "text" else "thinking",
                            "value": value,
                            "kind": kind,
                        }
                    )
            for call in delta.get("tool_calls") or []:
                ops.extend(tool_operations(BlockRef("tool_use", call.get("index", 0)), call))
            if choice.get("finish_reason") is not None:
                ops.append(
                    {
                        "type": "finish",
                        "reason": choice["finish_reason"],
                        "allow_tail": ["metadata", "usage"],
                    }
                )
        for operation in ops:
            if operation["type"] == "usage":
                operation["usage"] = encode_usage(operation["usage"])
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
