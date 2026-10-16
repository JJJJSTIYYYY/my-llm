"""openai/responses chunks: encoding conversion."""

from typing import Any

from openai import Omit

from my_llm.codec.openai.responses.encoding import messages
from my_llm.codec.utils.chunks import batch, check_error, metadata
from my_llm.codec.utils.common import path_value
from my_llm.codec.utils.conversion import dictionaries
from my_llm.codec.utils.preservation import response_fields
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_responses"


def response_metadata(response: dict) -> dict:
    return {
        **metadata(response),
        **{
            # `tools` is also MyCodec.decode's input argument. Keep response
            # metadata separate so it cannot be mistaken for tool conversion.
            "response_tools" if key == "tools" else key: response[key]
            for key in ("tools", "tool_choice", "parallel_tool_calls")
            if key in response
        },
    }


def consumed_fields(event: dict) -> dict:
    kind = event.get("type", "")
    return {
        "type": True,
        "output_index": True,
        "content_index": True,
        "response": {
            **dict.fromkeys(("id", "model", "created", "created_at", "response_id"), True),
            **({"output": [response_fields]} if kind == "response.completed" else {}),
        },
        "item": response_fields if kind == "response.output_item.done" else {},
        **(
            {"delta": True}
            if kind
            in (
                "response.output_text.delta",
                "response.reasoning_summary_text.delta",
                "response.function_call_arguments.delta",
            )
            else {}
        ),
    }


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        check_error(event)
        kind = event.get("type", "")
        index = event.get("output_index", 0)
        ops = []
        if kind in ("response.created", "response.in_progress"):
            ops.append({"type": "metadata", **response_metadata(event.get("response", {}))})
        elif (
            kind == "response.output_item.added"
            and event.get("item", {}).get("type") == "function_call"
        ):
            item = event["item"]
            ops.append(
                {
                    "type": "tool_delta",
                    "index": f"tool_{index}",
                    "id": item.get("call_id"),
                    "function": {"name": item.get("name"), "arguments": item.get("arguments", "")},
                }
            )
        elif kind == "response.function_call_arguments.delta":
            ops.append(
                {
                    "type": "tool_delta",
                    "index": f"tool_{index}",
                    "function": {"arguments": event.get("delta", "")},
                }
            )
        elif (
            kind == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ):
            block = dictionaries(messages.encode([event["item"]]))[0]["content"][0]
            ops.append({"type": "tool_complete", "index": f"tool_{index}", "block": block})
        elif kind == "response.completed":
            response = event["response"]
            output = response.get("output", [])
            ops.extend(
                [
                    {"type": "metadata", **response_metadata(response)},
                    {"type": "snapshot", "messages": messages.encode(output)},
                    {
                        "type": "finish",
                        "reason": "tool_calls"
                        if any(i.get("type") == "function_call" for i in output)
                        else "stop",
                    },
                ]
            )
            if response.get("usage") is not None and (
                profile is None or profile.get("usage_mode") != "none"
            ):
                ops.append({"type": "usage", "usage": response["usage"]})
        for field, default, block_kind in (
            ("content_path", "response.output_text.delta", "text"),
            ("reasoning_content_path", "response.reasoning_summary_text.delta", "thinking"),
        ):
            selector = profile.values[field] if profile else default
            if isinstance(selector, Omit):
                continue
            # Profiles may select either a JSON path or a native delta event type.
            value = (
                path_value(event, selector)
                if selector.startswith("$")
                else event.get("delta")
                if kind == selector
                else None
            )
            if value:
                ops.append(
                    {
                        "type": "delta",
                        "index": f"{block_kind}_{index}_{event.get('content_index', 0)}",
                        "kind": block_kind,
                        "field": "text" if block_kind == "text" else "thinking",
                        "value": value,
                    }
                )
        structural = {
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.function_call_arguments.done",
        }
        # Known structural events need no operation: their payload is represented
        # by deltas or the final snapshot. Only unknown events pass through opaquely.
        if not ops and kind not in structural:
            ops = [{"type": "opaque", "value": event}]
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
