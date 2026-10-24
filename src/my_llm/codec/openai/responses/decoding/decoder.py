"""openai/responses: intermediate to native output."""

from typing import Any

from openai.resources.responses import AsyncResponses
from openai.types.responses import ResponseInputItemParam, ToolParam

from my_llm.basic.data import dictionaries
from my_llm.basic.exceptions import ConfigurationError, ResponseFailedError
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.responses.decoding.content import (
    _tool_output,
    foreign_block,
    render_content,
)
from my_llm.codec.shared.conversion import (
    decode_functions,
    json_arguments,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.schema import parameter_types
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens
from my_llm.profile.types import Omit

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}

PROTOCOL = "openai_responses"


class ResponsesDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    tool_type = ToolParam
    message_type = ResponseInputItemParam
    message_path = ("input",)
    tool_path = ("tools",)

    supports_previous_response_id = True

    request_schema = parameter_types(AsyncResponses.create)

    def allows_external_calls(self, previous_response_id) -> bool:
        return previous_response_id is not None and not isinstance(previous_response_id, Omit)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("extra_body",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        if "max_output_tokens" in request:
            validate_max_tokens(request["max_output_tokens"])
        reasoning = request.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise ConfigurationError("reasoning must be a dictionary")
            for field in ("effort", "summary", "generate_summary"):
                if reasoning.get(field) is not None and not isinstance(reasoning[field], str):
                    raise ConfigurationError(f"reasoning.{field} must be a string")
        return request

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
        for index, message in enumerate(dictionaries(messages)):
            if message.get("name") is not None:
                report_loss(
                    message["name"],
                    "Responses cannot represent message names",
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            group_start = len(result)
            context_owner = source_protocol(message)
            content = []
            for block in message["content"]:
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                if block is not None:
                    content.append(block)
            pending = []

            def flush():
                if pending:
                    media = any(block.get("type") == "image" for block in pending)
                    input_media = media and profile is not None
                    if media and message["role"] == "assistant" and profile is None:
                        retained = []
                        for block in pending:
                            if block.get("type") == "image":
                                report_loss(
                                    block, "Responses assistant output cannot represent images"
                                )
                            else:
                                retained.append(block)
                        pending[:] = retained
                        if not pending:
                            return
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
                            else _tool_output(block.get("content", "")),
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
            if kind in ("block_start", "block_stop", "tool", "block"):
                owner = event["block"].get("protocol")
                native_owner = (
                    parameters.get("source_protocol") if owner is None else owner
                ) == PROTOCOL
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
                        else {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                    }
                )
            elif kind == "delta" and event["field"] in ("text", "thinking"):
                thinking = event["field"] == "thinking"
                value = {
                    "type": "response.reasoning_summary_text.delta"
                    if thinking
                    else "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index" if thinking else "content_index": part_index,
                    "delta": event["value"],
                    **({"logprobs": []} if not thinking else {}),
                }
                result.append(value)
            elif kind == "block_stop":
                block = event["block"]
                if not native_owner:
                    foreign_block(block)
                thinking = block.get("type") == "thinking"
                value = block["thinking" if thinking else "text"]
                result.append(
                    {
                        "type": "response.reasoning_summary_text.done"
                        if thinking
                        else "response.output_text.done",
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index" if thinking else "content_index": part_index,
                        "text": value,
                        **({"logprobs": []} if not thinking else {}),
                    }
                )
                part = (
                    {"type": "summary_text", "text": value}
                    if thinking
                    else {
                        "type": "output_text",
                        "text": value,
                        "annotations": [],
                        "logprobs": [],
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
                            "item": {**item, "status": "in_progress"},
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
                if event["block"].get("type") == "redacted_thinking":
                    report_loss(
                        event["block"],
                        "Responses cannot represent opaque reasoning from another protocol",
                    )
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
        # Required native envelope defaults carry no unsupported source metadata.
        return {
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
        }

    def _output_item(self, block: dict, index: int, origin: str | None) -> dict:
        item = self._messages([{"role": "assistant", "content": [block], "protocol": origin}])[0]
        if origin != PROTOCOL:
            item["id"] = f"item_{index}"
        item.setdefault("id", f"item_{index}")
        item.setdefault("status", "completed")
        return item

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
                retained = []
                for block in blocks:
                    owner = block.get("protocol")
                    if (origin if owner is None else owner) != PROTOCOL:
                        block = foreign_block(block)
                    if block is not None:
                        retained.append(block)
                blocks = retained
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
        return result

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
