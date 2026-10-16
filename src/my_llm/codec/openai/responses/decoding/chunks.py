"""openai/responses chunks: decoding conversion."""

from typing import Any

from my_llm.codec.openai.responses.decoding import messages
from my_llm.codec.utils.chunks import events
from my_llm.codec.utils.conversion import extras
from my_llm.codec.utils.keys import OPAQUE
from my_llm.codec.utils.usage import convert_usage
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_responses"


def response_envelope(**parameters: Any) -> dict:
    # Other protocols do not report these request settings. Supply valid defaults
    # only when unavailable, preserving native Responses metadata when present.
    return {
        "id": parameters.get("response_id", "generated"),
        "object": "response",
        "model": parameters.get("model", ""),
        "created_at": parameters.get("created_at", parameters.get("created", 0)),
        "parallel_tool_calls": parameters.get("parallel_tool_calls", True),
        "tool_choice": parameters.get("tool_choice", "auto"),
        "tools": parameters.get("response_tools", []),
    }


def output_item(block: dict, index: int) -> dict:
    item = messages.decode([{"role": "assistant", "content": [block]}])[0]
    item.update(id=f"item_{index}", status="completed")
    return item


def completed_output(event: dict) -> list[dict]:
    if "output_blocks" not in event:
        return messages.decode(event.get("messages", []))
    result = [output_item(entry["block"], entry["index"]) for entry in event["output_blocks"]]
    result.extend(extras(message, OPAQUE) for message in event["messages"] if message.get(OPAQUE))
    return result


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    passthrough = set()
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = convert_usage(event.get("usage"), event.get("usage_protocol"), PROTOCOL)
        index = event.get("index", 0)
        item_id = f"item_{index}"
        if kind == "opaque":
            result.append(event["value"])
            passthrough.add(id(event["value"]))
        elif kind == "start":
            result.append(
                {
                    "type": "response.created",
                    "response": {
                        **response_envelope(**parameters),
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
        elif kind == "block_start":
            thinking = event["block"].get("type") == "thinking"
            item = (
                {"id": item_id, "type": "reasoning", "summary": []}
                if thinking
                else {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
            )
            result.append(
                {"type": "response.output_item.added", "output_index": index, "item": item}
            )
            result.append(
                {
                    "type": "response.reasoning_summary_part.added"
                    if thinking
                    else "response.content_part.added",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": 0,
                    "part": {"type": "summary_text", "text": ""}
                    if thinking
                    else {"type": "output_text", "text": "", "annotations": []},
                }
            )
        elif kind == "delta":
            thinking = event["field"] == "thinking"
            result.append(
                {
                    "type": "response.reasoning_summary_text.delta"
                    if thinking
                    else "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": 0,
                    "delta": event["value"],
                    **({"logprobs": []} if not thinking else {}),
                }
            )
        elif kind == "block_stop":
            block = event["block"]
            thinking = block.get("type") == "thinking"
            value = block.get("thinking" if thinking else "text", "")
            result.append(
                {
                    "type": "response.reasoning_summary_text.done"
                    if thinking
                    else "response.output_text.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": 0,
                    "text": value,
                    **({"logprobs": block.get("logprobs") or []} if not thinking else {}),
                }
            )
            part = (
                {"type": "summary_text", "text": value}
                if thinking
                else {
                    "type": "output_text",
                    "text": value,
                    "annotations": block.get("annotations", []),
                }
            )
            result.append(
                {
                    "type": "response.reasoning_summary_part.done"
                    if thinking
                    else "response.content_part.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": 0,
                    "part": part,
                }
            )
            item = output_item(block, index)
            result.append(
                {"type": "response.output_item.done", "output_index": index, "item": item}
            )
        elif kind == "tool":
            item = output_item(event["block"], index)
            result.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": {**item, "arguments": "", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": index,
                        "item_id": item_id,
                        "delta": item["arguments"],
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": index,
                        "item_id": item_id,
                        "arguments": item["arguments"],
                        "name": item["name"],
                    },
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                ]
            )
        elif kind == "block":
            item = output_item(event["block"], index)
            result.extend(
                [
                    {"type": "response.output_item.added", "output_index": index, "item": item},
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                ]
            )
        elif kind == "finish":
            # Truncation and filtering are incomplete Responses, not successful ends.
            incomplete = {"length": "max_output_tokens", "content_filter": "content_filter"}.get(
                event["reason"]
            )
            result.append(
                {
                    "type": "response.incomplete" if incomplete else "response.completed",
                    "response": {
                        **response_envelope(**parameters),
                        "status": "incomplete" if incomplete else "completed",
                        **({"incomplete_details": {"reason": incomplete}} if incomplete else {}),
                        "output": completed_output(event),
                        "usage": usage,
                    },
                }
            )
    sequence = parameters.get("sequence_number", 0)
    for offset, event in enumerate(result):
        if id(event) not in passthrough:
            event.setdefault("sequence_number", sequence + offset)
    return result
