"""openai/responses: intermediate to native output."""

from copy import deepcopy
from typing import Any

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.responses.decoding.content import foreign_block, render_content
from my_llm.codec.openai.responses.utils.metadata import response_parameters
from my_llm.codec.shared.conversion import (
    decode_functions,
    dictionaries,
    json_arguments,
    project_message,
    select,
    select_present,
    source_protocol,
    tool_output,
)
from my_llm.codec.shared.stream import counts, events
from my_llm.exceptions import ResponseFailedError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit

PROTOCOL = "openai_responses"


class ResponsesDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    body_container = "extra_body"
    body_overlay = "extra_body"
    omit_empty_body = True

    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        request: dict,
        **parameters: Any,
    ) -> dict:
        previous = parameters.pop("previous_response_id", None)
        has_previous = previous is not None and not isinstance(previous, Omit)
        result = request
        result.update(model=profile.model, stream=True, input=messages)
        if tools is not None:
            result["tools"] = tools
        if has_previous:
            result["previous_response_id"] = previous
        return result

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for message in dictionaries(messages):
            group_start = len(result)
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
            content = message.get("content", [])
            pending = []

            def flush():
                if pending:
                    if message["role"] == "assistant" and profile is None:
                        pending[:] = [
                            {"type": "text", "text": block["source"]["url"]}
                            if block.get("type") in ("image", "document")
                            and block.get("source", {}).get("url")
                            else block
                            for block in pending
                        ]
                    media = any(block.get("type") in ("image", "document") for block in pending)
                    input_media = media and profile is not None
                    if media and message["role"] == "assistant" and profile is None:
                        raise UnsupportedCapabilityError(
                            "Responses output messages cannot represent foreign binary media"
                        )
                    result.append(
                        {
                            "type": "message",
                            **(
                                {"id": f"msg_{len(result)}", "status": "completed"}
                                if message["role"] == "assistant" and not input_media
                                else {}
                            ),
                            **select(message, "id", "status"),
                            **({"id": pending[0]["item_id"]} if pending[0].get("item_id") else {}),
                            **(
                                {"status": pending[0]["item_status"]}
                                if pending[0].get("item_status")
                                else {}
                            ),
                            "role": message["role"],
                            "content": render_content(
                                pending, role="user" if input_media else message["role"]
                            ),
                        }
                    )
                    pending.clear()

            for block in content:
                kind = block.get("type")
                if kind == "item_reference":
                    flush()
                    result.append(select(block, "type", "id"))
                    continue
                if kind not in ("tool_use", "tool_result", "thinking"):
                    if pending and block.get("item_id") != pending[0].get("item_id"):
                        flush()
                    pending.append(block)
                    continue
                flush()
                if kind == "tool_use":
                    result.append(
                        {
                            **select(block, "name", "status"),
                            **({"id": block["item_id"]} if "item_id" in block else {}),
                            "type": "function_call",
                            "call_id": block["id"],
                            "arguments": json_arguments(block["input"]),
                        }
                    )
                elif kind == "tool_result":
                    result.append(
                        {
                            **select(block, "id", "status"),
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": render_content(block["content"], role="user")
                            if isinstance(block.get("content"), list)
                            else tool_output(block.get("content", "")),
                        }
                    )
                else:
                    item = {
                        **select(block, "status", "encrypted_content"),
                        "type": "reasoning",
                        "id": block.get("id", f"rs_{len(result)}"),
                        "summary": [
                            select(part, "type", "text")
                            for part in block.get(
                                "summary",
                                [{"type": "summary_text", "text": block.get("thinking", "")}],
                            )
                        ],
                    }
                    if (
                        result
                        and result[-1].get("type") == "reasoning"
                        and result[-1].get("id") == item["id"]
                    ):
                        result[-1]["summary"].extend(item["summary"])
                        result[-1].update(select(item, "status", "encrypted_content"))
                    else:
                        result.append(item)
            flush()
            if len(result) == group_start:
                result.append(
                    {
                        "type": "message",
                        "role": message["role"],
                        **(
                            {"id": f"msg_{len(result)}", "status": "completed"}
                            if message["role"] == "assistant"
                            else {}
                        ),
                        "content": render_content(
                            [{"type": "text", "text": ""}], role=message["role"]
                        ),
                    }
                )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [
                {
                    "type": "function",
                    **select_present(f, "name", "description", "strict"),
                    "parameters": f.get("parameters"),
                    "strict": f.get("strict") or False,
                }
                for f in functions
            ],
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            index, part_index, item_id, native = self._address(event, parameters)
            if kind == "start":
                result.append(
                    {
                        "type": "response.created",
                        "response": {
                            **self._response_envelope(**parameters),
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
                        item = self._added_item(parameters["response_items"][str(index)])
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
            elif kind == "delta" and event.get("field") in ("text", "thinking", "refusal"):
                thinking = (
                    event["field"] == "thinking" and parameters.get("source_protocol") == PROTOCOL
                )
                refusal = event["field"] == "refusal"
                value = {
                    "type": "response.reasoning_summary_text.delta"
                    if thinking
                    else "response.refusal.delta"
                    if refusal
                    else "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": part_index,
                    "delta": deepcopy(event["value"]),
                    **({"logprobs": []} if not thinking and not refusal else {}),
                }
                result.append(value)
            elif kind == "block_stop":
                block = event["block"]
                thinking = (
                    block.get("type") == "thinking"
                    and parameters.get("source_protocol") == PROTOCOL
                )
                refusal = block.get("type") == "refusal"
                rendered_text = (
                    deepcopy(
                        render_content(
                            [
                                {"type": "text", "text": block.get("thinking", "")}
                                if block.get("type") == "thinking"
                                else block
                            ],
                            role="assistant",
                        )[0]
                    )
                    if not thinking and not refusal
                    else {}
                )
                value = block.get(
                    "thinking"
                    if block.get("type") == "thinking"
                    else "refusal"
                    if refusal
                    else "text",
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
                            {"logprobs": rendered_text.get("logprobs", [])}
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
                        "annotations": rendered_text.get("annotations", []),
                        "logprobs": rendered_text.get("logprobs", []),
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
                    item = self._output_item(block, index, parameters.get("source_protocol"))
                    result.append(
                        {"type": "response.output_item.done", "output_index": index, "item": item}
                    )
            elif kind == "tool":
                item = (
                    {
                        **self._output_item(
                            event["block"], index, parameters.get("source_protocol")
                        ),
                        **select(
                            parameters["response_items"][str(index)],
                            "id",
                            "status",
                            "name",
                            "call_id",
                        ),
                    }
                    if native
                    else self._output_item(event["block"], index, parameters.get("source_protocol"))
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
                item = self._output_item(event["block"], index, parameters.get("source_protocol"))
                result.extend(
                    [
                        {"type": "response.output_item.added", "output_index": index, "item": item},
                        {"type": "response.output_item.done", "output_index": index, "item": item},
                    ]
                )
            elif kind == "finish":
                # Truncation and filtering are incomplete Responses, not successful ends.
                incomplete = {
                    "length": "max_output_tokens",
                    "content_filter": "content_filter",
                }.get(event["reason"])
                if incomplete is None and event["reason"] not in ("stop", "tool_calls"):
                    raise ResponseFailedError(
                        "Responses cannot represent the source finish reason",
                        details={"reason": event["reason"]},
                    )
                output = self._completed_output(
                    event, parameters.get("source_protocol"), parameters
                )
                if self._native_items(parameters):
                    represented = {
                        entry["source_ref"][1]
                        for entry in event.get("output_blocks", [])
                        if entry.get("source_ref") is not None
                    }
                    for output_index, item in enumerate(output):
                        if output_index not in represented:
                            result.append(
                                {
                                    "type": "response.output_item.added",
                                    "output_index": output_index,
                                    "item": self._added_item(item),
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
                            **self._response_envelope(**parameters),
                            "status": "incomplete" if incomplete else "completed",
                            **(
                                {"incomplete_details": {"reason": incomplete}} if incomplete else {}
                            ),
                            "output": output,
                            "usage": usage,
                        },
                    }
                )
        sequence = parameters.get("sequence_number", 0)
        for offset, event in enumerate(result):
            event.setdefault("sequence_number", sequence + offset)
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        if input_tokens is None or output_tokens is None:
            return None
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "input_tokens_details": {
                "cached_tokens": cached or 0,
                "cache_write_tokens": cache_write or 0,
            },
            "output_tokens_details": {"reasoning_tokens": reasoning or 0},
        }

    def result(self, messages: list) -> list:
        return messages

    def _response_envelope(self, **parameters: Any) -> dict:
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
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                **(
                    response_parameters(parameters.get("response_parameters", {}))
                    if parameters.get("source_protocol") == PROTOCOL
                    else {}
                ),
            }
        )

    def _output_item(self, block: dict, index: int, origin: str | None) -> dict:
        item = self._messages([{"role": "assistant", "content": [block], "protocol": origin}])[0]
        if origin != PROTOCOL:
            item["id"] = f"item_{index}"
        item.setdefault("id", f"item_{index}")
        item.setdefault("status", "completed")
        return deepcopy(item)

    def _completed_output(self, event: dict, origin: str | None, parameters: dict) -> list[dict]:
        entries = event.get("output_blocks", [])
        if not self._native_items(parameters):
            return [
                self._output_item(entry["block"], entry["index"], origin)
                for entry in entries
                if origin == PROTOCOL or entry["block"].get("type") != "redacted_thinking"
            ]
        grouped = {}
        for entry in entries:
            ref = entry.get("source_ref")
            index = ref[1] if ref is not None else entry["index"]
            grouped.setdefault(index, []).append(entry["block"])
        result = []
        for index, header in sorted(
            parameters["response_items"].items(), key=lambda pair: int(pair[0])
        ):
            blocks = grouped.get(int(index), [])
            kind = header.get("type")
            if kind == "message":
                item = {
                    "type": kind,
                    "role": "assistant",
                    "content": render_content(blocks, role="assistant"),
                }
            elif kind == "reasoning":
                item = {
                    "type": kind,
                    "summary": [
                        select(part, "type", "text")
                        for block in blocks
                        for part in block.get(
                            "summary", [{"type": "summary_text", "text": block.get("thinking", "")}]
                        )
                    ],
                }
                for block in blocks:
                    item.update(select(block, "encrypted_content"))
            elif kind == "function_call" and blocks:
                item = self._output_item(blocks[0], int(index), origin)
            else:
                continue
            item.update(select(header, "id", "status", "name", "call_id"))
            result.append(item)
        return deepcopy(result)

    def _native_items(self, parameters: dict) -> bool:
        return parameters.get("source_protocol") == PROTOCOL and bool(
            parameters.get("response_items")
        )

    def _address(self, event: dict, parameters: dict) -> tuple[int, int, str, bool]:
        index, part = event.get("index", 0), 0
        ref = event.get("source_ref")
        if self._native_items(parameters) and ref is not None:
            parent = parameters["response_items"].get(str(ref[1]), {})
            if parent.get("id"):
                return ref[1], ref[2], parent["id"], True
        return index, part, f"item_{index}", False

    def _added_item(self, item: dict) -> dict:
        result = select(item, "type", "id", "role", "status", "call_id", "name")
        result["status"] = "in_progress"
        if item["type"] == "message":
            result.setdefault("role", "assistant")
            result["content"] = []
        elif item["type"] == "reasoning":
            result["summary"] = []
        elif item["type"] == "function_call":
            result["arguments"] = ""
        return result
