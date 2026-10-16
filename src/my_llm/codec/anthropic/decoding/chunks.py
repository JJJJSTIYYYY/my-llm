"""anthropic chunks: decoding conversion."""

from typing import Any

from my_llm.codec.anthropic._constants import _REASONS
from my_llm.codec.utils.chunks import events
from my_llm.codec.utils.conversion import extras, json_arguments
from my_llm.codec.utils.usage import convert_usage
from my_llm.profile import ResolvedProfile

PROTOCOL = "anthropic"


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = convert_usage(event.get("usage"), event.get("usage_protocol"), PROTOCOL)
        index = event.get("index", 0)
        if kind == "opaque":
            result.append(event["value"])
        elif kind == "start":
            result.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": parameters.get("response_id", "generated"),
                        "type": "message",
                        "role": "assistant",
                        "model": parameters.get("model", ""),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0, **(usage or {})},
                    },
                }
            )
        elif kind == "block_start":
            block = event["block"]
            result.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                    if block["type"] == "thinking"
                    else {"type": "text", "text": ""},
                }
            )
        elif kind == "delta":
            field = event["field"]
            result.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": f"{field}_delta", field: event["value"]},
                }
            )
        elif kind == "block_stop":
            signature = event["block"].get("signature")
            if signature:
                result.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "signature_delta", "signature": signature},
                    }
                )
            result.append({"type": "content_block_stop", "index": index})
        elif kind == "tool":
            block = extras(event["block"], "thought_signature")
            result.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json_arguments(block["input"]),
                        },
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
        elif kind == "block":
            result.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": extras(event["block"], "thought_signature"),
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
        elif kind == "finish":
            reason = {v: k for k, v in _REASONS.items()}.get(event["reason"], event["reason"])
            result.extend(
                [
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": None},
                        "usage": usage or {"output_tokens": 0},
                    },
                    {"type": "message_stop"},
                ]
            )
    return result
