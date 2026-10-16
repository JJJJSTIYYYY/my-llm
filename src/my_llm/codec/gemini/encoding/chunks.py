"""gemini chunks: encoding conversion."""

from typing import Any

from my_llm.codec.gemini.encoding import messages
from my_llm.codec.utils.chunks import batch, check_error
from my_llm.codec.utils.conversion import dictionaries
from my_llm.codec.utils.preservation import part_fields
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile import ResolvedProfile

PROTOCOL = "gemini"


def consumed_fields(event: dict) -> dict:
    return {
        "response_id": True,
        "model_version": True,
        "prompt_feedback": {"block_reason": True},
        "candidates": [{"index": True, "finish_reason": True, "content": {"parts": [part_fields]}}],
    }


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        check_error(event)
        if not any(
            key in event
            for key in (
                "candidates",
                "usage_metadata",
                "prompt_feedback",
                "response_id",
                "model_version",
            )
        ):
            result.append(batch(event, [{"type": "opaque", "value": event}], PROTOCOL))
            continue
        if event.get("prompt_feedback", {}).get("block_reason") not in (
            None,
            "BLOCK_REASON_UNSPECIFIED",
        ):
            raise ResponseFailedError("Gemini blocked the prompt", details=event["prompt_feedback"])
        ops = [
            {
                "type": "metadata",
                "id": event.get("response_id"),
                "model": event.get("model_version"),
            }
        ]
        if event.get("usage_metadata") is not None and (
            profile is None or profile.get("usage_mode") != "none"
        ):
            ops.append({"type": "usage", "usage": event["usage_metadata"]})
        for candidate in event.get("candidates", []):
            if candidate.get("index", 0) not in (None, 0):
                raise ConfigurationError("only one generated candidate is supported")
            for index, part in enumerate(candidate.get("content", {}).get("parts", [])):
                if "function_call" in part:
                    ops.append(
                        {
                            "type": "tool_complete",
                            "block": messages.encode_part(part, ""),
                            "index": None,
                        }
                    )
                elif "text" in part:
                    thinking = part.get("thought", False)
                    kind = "thinking" if thinking else "text"
                    ops.append(
                        {
                            "type": "delta",
                            "index": None,
                            "kind": kind,
                            "field": kind,
                            "value": part["text"],
                            "extras": {
                                k: v for k, v in part.items() if k not in ("text", "thought")
                            },
                        }
                    )
                else:
                    ops.append(
                        {
                            "type": "complete_block",
                            "block": messages.encode_part(part, f"call_{index}"),
                        }
                    )
            reason = candidate.get("finish_reason")
            if reason not in (None, "FINISH_REASON_UNSPECIFIED"):
                if reason not in ("STOP", "MAX_TOKENS"):
                    raise ResponseFailedError("Gemini generation failed", details=candidate)
                ops.append(
                    {"type": "finish", "reason": "length" if reason == "MAX_TOKENS" else "stop"}
                )
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
