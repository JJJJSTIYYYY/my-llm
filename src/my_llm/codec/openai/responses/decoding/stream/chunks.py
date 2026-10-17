"""openai/responses chunks: decoding conversion."""

from copy import deepcopy
from typing import Any

from my_llm.codec.openai.responses.decoding.messages import conversion as messages
from my_llm.codec.openai.responses.decoding.stream.usage import decode as decode_usage
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import extras
from my_llm.codec.shared.keys import OPAQUE, SOURCE
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_responses"


def response_envelope(**parameters: Any) -> dict:
    # Other protocols do not report these request settings. Supply valid defaults
    # only when unavailable, preserving native Responses metadata when present.
    return deepcopy(
        {
            "access_programs": None,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "temperature": None,
            "top_p": None,
            "id": parameters.get("response_id", "generated"),
            "object": "response",
            "model": parameters.get("model", ""),
            "created_at": parameters.get("created_at", parameters.get("created", 0)),
            "parallel_tool_calls": parameters.get("parallel_tool_calls", True),
            "tool_choice": parameters.get("tool_choice", "auto"),
            "tools": parameters.get("response_tools", []),
            **(
                parameters.get("native_response", {})
                if parameters.get("source_protocol") == PROTOCOL
                else {}
            ),
        }
    )


def output_item(block: dict, index: int, origin: str | None) -> dict:
    item = messages.decode(
        [{"role": "assistant", "content": [block], SOURCE: {"protocol": origin}}]
    )[0]
    item.update(id=f"item_{index}", status="completed")
    return item


def completed_output(event: dict, origin: str | None) -> list[dict]:
    if "output_blocks" not in event:
        return messages.decode(event.get("messages", []))
    result = [
        output_item(entry["block"], entry["index"], origin)
        for entry in event["output_blocks"]
        if origin == PROTOCOL or entry["block"].get("type") != "redacted_thinking"
    ]
    result.extend(extras(message, OPAQUE) for message in event["messages"] if message.get(OPAQUE))
    return result


def native_items(parameters: dict) -> bool:
    return parameters.get("source_protocol") == PROTOCOL and bool(parameters.get("response_items"))


def address(event: dict, parameters: dict) -> tuple[int, int, str, bool]:
    index, part = event.get("index", 0), 0
    ref = event.get("source_ref")
    if native_items(parameters) and ref is not None:
        parent = parameters["response_items"].get(str(ref[1]), {})
        if parent.get("id"):
            return ref[1], ref[2], parent["id"], True
    return index, part, f"item_{index}", False


def added_item(item: dict) -> dict:
    result = deepcopy(item)
    result["status"] = "in_progress"
    if item["type"] == "message":
        result.setdefault("role", "assistant")
        result["content"] = []
    elif item["type"] == "reasoning":
        result["summary"] = []
    elif item["type"] == "function_call":
        result["arguments"] = ""
    return result


def decode(chunks: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    passthrough = set()
    for event in events(chunks, PROTOCOL):
        kind = event.get("type")
        usage = decode_usage(event.get("usage"))
        index, part_index, item_id, native = address(event, parameters)
        if kind == "opaque":
            value = event["value"]
            if event.get("protocol") == PROTOCOL and "sequence_number" in value:
                value = {key: child for key, child in value.items() if key != "sequence_number"}
            else:
                passthrough.add(id(value))
            result.append(value)
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
            thinking = (
                event["block"].get("type") == "thinking"
                and parameters.get("source_protocol") == PROTOCOL
            )
            refusal = event["block"].get("type") == "refusal"
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
            if not native or part_index == 0:
                if native:
                    item = added_item(parameters["response_items"][str(index)])
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
                    "summary_index" if thinking else "content_index": part_index,
                    "part": {"type": "summary_text", "text": ""}
                    if thinking
                    else {"type": "refusal", "refusal": ""}
                    if refusal
                    else {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                }
            )
        elif kind == "delta":
            thinking = (
                event["field"] == "thinking" and parameters.get("source_protocol") == PROTOCOL
            )
            refusal = event["field"] == "refusal"
            result.append(
                {
                    "type": "response.reasoning_summary_text.delta"
                    if thinking
                    else "response.refusal.delta"
                    if refusal
                    else "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": part_index,
                    "delta": event["value"],
                    **({"logprobs": []} if not thinking and not refusal else {}),
                }
            )
        elif kind == "block_stop":
            block = event["block"]
            thinking = (
                block.get("type") == "thinking" and parameters.get("source_protocol") == PROTOCOL
            )
            refusal = block.get("type") == "refusal"
            value = block.get(
                "thinking" if block.get("type") == "thinking" else "refusal" if refusal else "text",
                "",
            )
            result.append(
                {
                    "type": "response.reasoning_summary_text.done"
                    if thinking
                    else "response.refusal.done"
                    if refusal
                    else "response.output_text.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": part_index,
                    "refusal" if refusal else "text": value,
                    **(
                        {"logprobs": block.get("logprobs") or []}
                        if not thinking and not refusal
                        else {}
                    ),
                }
            )
            part = (
                {"type": "summary_text", "text": value}
                if thinking
                else {"type": "refusal", "refusal": value}
                if refusal
                else {
                    "type": "output_text",
                    "text": value,
                    "annotations": block.get("annotations", []),
                    "logprobs": block.get("logprobs") or [],
                }
            )
            result.append(
                {
                    "type": "response.reasoning_summary_part.done"
                    if thinking
                    else "response.content_part.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": part_index,
                    "part": part,
                }
            )
            if not native:
                item = output_item(block, index, parameters.get("source_protocol"))
                result.append(
                    {"type": "response.output_item.done", "output_index": index, "item": item}
                )
        elif kind == "tool":
            item = (
                deepcopy(parameters["response_items"][str(index)])
                if native
                else output_item(event["block"], index, parameters.get("source_protocol"))
            )
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
                ]
            )
            if not native:
                result.append(
                    {"type": "response.output_item.done", "output_index": index, "item": item}
                )
        elif kind == "block":
            if (
                event["block"].get("type") == "redacted_thinking"
                and parameters.get("source_protocol") != PROTOCOL
            ):
                continue
            item = output_item(event["block"], index, parameters.get("source_protocol"))
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
            if incomplete is None and event["reason"] not in ("stop", "tool_calls"):
                raise ResponseFailedError(
                    "Responses cannot represent the source finish reason",
                    details={"reason": event["reason"]},
                )
            output = (
                messages.decode(event.get("messages", []))
                if native_items(parameters)
                else completed_output(event, parameters.get("source_protocol"))
            )
            if native_items(parameters):
                represented = {
                    entry["source_ref"][1]
                    for entry in event.get("output_blocks", [])
                    if entry.get("source_ref") is not None
                }
                for output_index, item in enumerate(output):
                    if item.get("type") not in ("message", "reasoning", "function_call"):
                        continue  # Native opaque item envelopes were relayed in order.
                    if output_index not in represented:
                        result.append(
                            {
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": added_item(item),
                            }
                        )
                    result.append(
                        {
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": item,
                        }
                    )
            result.append(
                {
                    "type": "response.incomplete" if incomplete else "response.completed",
                    "response": {
                        **response_envelope(**parameters),
                        "status": "incomplete" if incomplete else "completed",
                        **({"incomplete_details": {"reason": incomplete}} if incomplete else {}),
                        "output": output,
                        "usage": usage,
                    },
                }
            )
    sequence = parameters.get("sequence_number", 0)
    for offset, event in enumerate(result):
        if id(event) not in passthrough:
            event.setdefault("sequence_number", sequence + offset)
    return result
