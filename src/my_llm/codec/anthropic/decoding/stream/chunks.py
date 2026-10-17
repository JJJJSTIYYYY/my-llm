"""anthropic chunks: decoding conversion."""

from typing import Any

from my_llm.codec.anthropic.decoding.messages.projection import foreign_block
from my_llm.codec.anthropic.decoding.stream.usage import decode as decode_usage
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import extras, json_arguments
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"
REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = decode_usage(event.get("usage"))
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
                    if block["type"] == "thinking" and parameters.get("source_protocol") == PROTOCOL
                    else {"type": "text", "text": ""},
                }
            )
        elif kind == "delta":
            field = event["field"]
            if field == "refusal" or (
                field == "thinking" and parameters.get("source_protocol") != PROTOCOL
            ):
                field = "text"
            result.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": f"{field}_delta", field: event["value"]},
                }
            )
        elif kind == "block_annotation":
            if event["field"] == "citations" and parameters.get("source_protocol") == PROTOCOL:
                result.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "citations_delta", "citation": event["value"]},
                    }
                )
        elif kind == "block_stop":
            signature = event["block"].get("signature")
            if signature and parameters.get("source_protocol") == PROTOCOL:
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
            if parameters.get("source_protocol") != PROTOCOL:
                block = foreign_block(block)
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
            block = event["block"]
            if parameters.get("source_protocol") != PROTOCOL:
                block = foreign_block(block)
                if block is None:
                    continue
            result.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": extras(block, "thought_signature"),
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
        elif kind == "finish":
            reason = REASONS.get(event["reason"], event["reason"])
            stop_sequence = None
            if parameters.get("source_protocol") == PROTOCOL:
                reason = parameters.get("native_stop_reason", reason)
                stop_sequence = parameters.get("native_stop_sequence")
            elif event["reason"] not in REASONS:
                raise ResponseFailedError(
                    "Anthropic cannot represent the source finish reason",
                    details={"reason": event["reason"]},
                )
            result.extend(
                [
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
                        "usage": usage or {"output_tokens": 0},
                    },
                    {"type": "message_stop"},
                ]
            )
    return result
