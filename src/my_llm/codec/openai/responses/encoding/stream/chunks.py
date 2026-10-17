"""openai/responses chunks: encoding conversion."""

from typing import Any

from openai import Omit

from my_llm.codec.openai.responses.encoding.messages import conversion as messages
from my_llm.codec.openai.responses.encoding.messages.content import encode_content
from my_llm.codec.openai.responses.encoding.messages.fields import response_fields
from my_llm.codec.openai.responses.encoding.stream.usage import encode as encode_usage
from my_llm.codec.openai.responses.encoding.tools.calls import tool_operations
from my_llm.codec.shared.chunks import BlockRef, batch
from my_llm.codec.shared.conversion import dictionaries, extras
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.values import path_value
from my_llm.exceptions import ResponseFailedError, ResponseIncompleteError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_responses"


def metadata(event: dict) -> dict:
    return {key: event[key] for key in ("id", "model", "created_at") if key in event}


def snapshot(output: list) -> dict:
    """Address final blocks with the same item/part coordinates as live deltas."""
    encoded = messages.encode(output)
    blocks = []
    for index, (item, message) in enumerate(zip(output, dictionaries(encoded))):
        if message.get(OPAQUE):
            continue
        for part, block in enumerate(message.get("content", [])):
            kind = block.get("type", "unknown")
            if kind == "thinking":
                # Message conversion groups summaries; live operations address
                # each summary part separately using the native summary_index.
                for summary_index, summary in enumerate(item.get("summary") or [{}]):
                    blocks.append(
                        {
                            "ref": BlockRef(kind, index, summary_index),
                            "block": {
                                **block,
                                "thinking": summary.get("text", ""),
                                "summary": [summary] if summary else [],
                            },
                        }
                    )
            else:
                blocks.append({"ref": BlockRef(kind, index, part), "block": block})
    return {"type": "snapshot", "messages": encoded, "blocks": blocks}


def response_metadata(response: dict) -> dict:
    return {
        **metadata(response),
        "native_response": extras(response, "output", "usage"),
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
            **dict.fromkeys(("id", "model", "created_at"), True),
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
                "response.refusal.delta",
            )
            else {}
        ),
    }


def encode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for event in dictionaries(chunks):
        kind = event.get("type")
        if kind in ("response.failed", "response.incomplete", "error") or event.get("error"):
            error = (
                ResponseIncompleteError if kind == "response.incomplete" else ResponseFailedError
            )
            response = event.get("response", {})
            raise error(
                "provider generation failed",
                details=response.get(
                    "incomplete_details", response.get("error", event.get("error", event))
                ),
            )
        kind = event.get("type", "")
        index = event.get("output_index", 0)
        ops = []
        if kind in ("response.created", "response.in_progress"):
            ops.append({"type": "metadata", **response_metadata(event.get("response", {}))})
            if kind == "response.created":
                ops.append({"type": "start"})
        elif (
            kind == "response.output_item.added"
            and event.get("item", {}).get("type") == "function_call"
        ):
            item = event["item"]
            ops.extend(
                tool_operations(
                    BlockRef("tool_use", index),
                    call_id=item.get("call_id"),
                    name=item.get("name"),
                    arguments=item.get("arguments", ""),
                )
            )
        elif kind == "response.function_call_arguments.delta":
            ops.extend(
                tool_operations(
                    BlockRef("tool_use", index),
                    arguments=event.get("delta", ""),
                )
            )
        elif (
            kind == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ):
            block = dictionaries(messages.encode([event["item"]]))[0]["content"][0]
            ops.append({"type": "block_set", "ref": BlockRef("tool_use", index), "block": block})
        elif kind in ("response.content_part.added", "response.content_part.done"):
            part = event.get("part", {})
            if part.get("type") in ("output_text", "refusal"):
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef(
                            "refusal" if part["type"] == "refusal" else "text",
                            index,
                            event.get("content_index", 0),
                        ),
                        "block": encode_content([part])[0],
                    }
                )
        elif kind == "response.refusal.delta":
            ops.append(
                {
                    "type": "block_delta",
                    "ref": BlockRef("refusal", index, event.get("content_index", 0)),
                    "kind": "refusal",
                    "field": "refusal",
                    "value": event["delta"],
                }
            )
        elif kind in (
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        ):
            part = event["part"]
            ops.append(
                {
                    "type": "block_set",
                    "ref": BlockRef("thinking", index, event.get("summary_index", 0)),
                    "block": {
                        "type": "thinking",
                        "thinking": part.get("text", ""),
                        "summary": [part],
                    },
                }
            )
        elif kind == "response.completed":
            response = event["response"]
            output = response.get("output", [])
            ops.extend(
                [
                    {"type": "metadata", **response_metadata(response)},
                    {
                        "type": "metadata",
                        "response_items": {str(i): item for i, item in enumerate(output)},
                    },
                    snapshot(output),
                    {
                        "type": "finish",
                        "reason": "tool_calls"
                        if any(i.get("type") == "function_call" for i in output)
                        else "stop",
                        "allow_tail": ["usage"],
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
            if value is not None:
                ops.append(
                    {
                        "type": "block_delta",
                        "ref": BlockRef(
                            block_kind,
                            index,
                            event.get(
                                "summary_index" if block_kind == "thinking" else "content_index", 0
                            ),
                        ),
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
            "response.refusal.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.function_call_arguments.done",
        }
        # Known structural events need no operation: their payload is represented
        # by deltas or the final snapshot. Only unknown events pass through opaquely.
        if not ops and kind not in structural:
            ops = [{"type": "opaque", "value": event}]
        if "output_index" in event:
            item = event.get("item") or {}
            item_id = item.get("id") or event.get("item_id")
            if item_id:
                header = item or {"id": item_id}
                if not item:
                    if "reasoning_summary" in kind:
                        header["type"] = "reasoning"
                    elif "function_call_arguments" in kind:
                        header["type"] = "function_call"
                    elif kind.startswith(
                        ("response.output_text", "response.content_part", "response.refusal")
                    ):
                        header["type"] = "message"
                ops.insert(
                    0,
                    {
                        "type": "metadata",
                        "merge": ["response_items"],
                        "response_items": {str(index): header},
                    },
                )
            if kind in ("response.output_item.added", "response.output_item.done") and item.get(
                "type"
            ) not in ("message", "reasoning", "function_call"):
                ops.append({"type": "opaque", "value": event})
        for operation in ops:
            if operation["type"] == "usage":
                operation["usage"] = encode_usage(operation["usage"])
        result.append(batch(event, ops, PROTOCOL, consumed_fields(event)))
    return result
