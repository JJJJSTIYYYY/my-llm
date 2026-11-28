"""openai/responses: intermediate to native output."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any
from uuid import uuid4

from openai.resources.responses import AsyncResponses
from openai.types.responses import ResponseInputItemParam, ToolParam

import my_llm.codec.openai.responses.constants as constants
from my_llm.basic.exceptions import ConfigurationError, ResponseFailedError
from my_llm.basic.semantic_loss import report_loss
from my_llm.cache.compiled_cache import _MISSING, _plan, parameter_types
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.responses.decoding.content import (
    _tool_output,
    foreign_block,
    reasoning_summary,
    render_content,
)
from my_llm.codec.shared.capabilities import content_capability
from my_llm.codec.shared.conversion import (
    decode_functions,
    json_arguments,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.input import (
    InputPolicy,
    input_content,
    message_feedback,
)
from my_llm.codec.shared.output import output_chunks
from my_llm.codec.shared.stream import counts, events
from my_llm.middleware.protocol import wire_values
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens
from my_llm.profile.types import Omit
from my_llm.utils.data import capture, dictionary_views


class ResponsesDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = constants.PROTOCOL

    tool_type = ToolParam
    message_type = ResponseInputItemParam
    message_path = constants.MESSAGE_PATH
    tool_path = constants.TOOL_PATH

    supports_previous_response_id = True

    request_schema = parameter_types(AsyncResponses.create)
    protected_request_fields = constants.REQUEST_ENVELOPE_FIELDS
    request_body_containers = constants.BODY_CONTAINERS

    def _message_inputs(
        self, messages: list, policy: InputPolicy
    ) -> Iterator[tuple[int, dict, bool]]:
        """Group native assistant items only when reasoning feedback needs later calls."""
        unconditional = policy.reasoning_feedback(has_tool_calls=False)
        conditional = policy.reasoning_feedback(has_tool_calls=True)
        if unconditional or not conditional:
            for index, message in enumerate(dictionary_views(messages)):
                yield (index, message, unconditional)
            return
        pending = []
        has_calls = False
        for index, message in enumerate(dictionary_views(messages)):
            kind = message.get("type")
            assistant = (
                kind in constants.REASONING_FUNCTION_CALL_ITEM_TYPES
                or message.get("role") == "assistant"
            )
            if not assistant:
                for item_index, item in pending:
                    yield (item_index, item, has_calls)
                pending = []
                has_calls = False
            pending.append((index, message))
            has_calls = has_calls or kind == "function_call"
        for index, message in pending:
            yield (index, message, has_calls)

    def _filter_message(
        self,
        message: dict,
        policy: InputPolicy,
        *,
        path: str,
        decode: Callable,
        feedback: bool,
        **parameters: Any,
    ) -> dict | None:
        """Filter one Responses item using its assistant-group reasoning context."""
        kind = constants.INPUT_ITEM_TYPE_TO_MIDDLEWARE_BLOCK_TYPE.get(message.get("type"), "text")
        if not policy.allows(message, kind, feedback=feedback, path=path):
            return None
        if message.get("name") is not None and kind == "text":
            policy.allows(message["name"], "message_name", path=f"{path}.name")
        field = "output" if kind == "tool_result" else "content"
        if isinstance(content := message.get(field), list):
            retained = [
                block
                for offset, block in enumerate(content)
                if not isinstance(block, dict)
                or block.get("type") != "input_image"
                or policy.allows(block, "image", path=f"{path}.{field}[{offset}]")
            ]
            if len(retained) != len(content):
                message = _plan(
                    self.message_type, delegated_paths=constants.MESSAGE_DELEGATED_PATHS
                ).decode(message, path)
                if message is _MISSING:
                    return None
                message[field] = retained
        value = decode(message, path)
        return None if value is _MISSING else value

    def allows_external_calls(self, previous_response_id) -> bool:
        """Allow unmatched local result IDs when previous_response_id supplies server history."""
        return previous_response_id is not None and not isinstance(previous_response_id, Omit)

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Validate output-token limits and the types of supported reasoning controls."""
        if "max_output_tokens" in request:
            validate_max_tokens(request["max_output_tokens"])
        reasoning = request.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise ConfigurationError("reasoning must be a dictionary")
            for field in constants.REASONING_FIELDS:
                if reasoning.get(field) is not None and not isinstance(reasoning[field], str):
                    raise ConfigurationError(f"reasoning.{field} must be a string")
        return request

    body_container = constants.BODY_CONTAINER
    body_overlay = constants.BODY_CONTAINER
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
        """Assemble streaming Responses input, optional tools and server-history reference."""
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
        self, messages: Iterable[dict], profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render history as Responses message, reasoning and function items.

        Group consecutive assistant items sourced from Responses only to decide
        reasoning feedback; merge=False keeps those items separate in the output.
        Preserve native item identity and ordering rather than coalescing assistant
        fragments. Foreign blocks pass through Responses compatibility handling.
        Generate fallback item identities independently for cached incremental
        batches; explicit native IDs take precedence. Tool results become separate
        function_call_output items keyed by the original call ID.
        """
        result = []
        # Each incremental batch needs distinct generated identities; explicit
        # native item IDs below continue to take precedence over these fallbacks.
        id_prefix = f"{uuid4().hex}_" if parameters.get("enable_message_cache", False) else ""
        for index, message, feedback in message_feedback(messages, profile, merge=False):
            if message.get("name") is not None:
                content_capability(
                    message["name"],
                    "message_name",
                    profile,
                    native_support=False,
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            group_start = len(result)
            context_owner = source_protocol(message) or parameters.get("source_protocol")
            content = []
            for block in message["content"]:
                if not input_content(block, profile, feedback=feedback):
                    continue
                if context_owner != constants.PROTOCOL:
                    block = foreign_block(block)
                if block is not None:
                    content.append(block)
            pending = []

            def flush():
                """Render buffered text/image blocks while retaining compatible native message identity."""
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
                                {"id": f"msg_{id_prefix}{len(result)}", "status": "completed"}
                                if message["role"] == "assistant" and not input_media
                                else {}
                            ),
                            **select(
                                pending[0].get("item") or message.get("item") or {},
                                *constants.ITEM_ID_STATUS_FIELDS,
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
                    if pending:
                        identity = block.get("item", {})
                        first_identity = pending[0].get("item", {})
                        if identity.get("id") != first_identity.get("id") or identity.get(
                            "status"
                        ) != first_identity.get("status"):
                            flush()
                    pending.append(block)
                    continue
                flush()
                if kind == "tool_use":
                    result.append(
                        {
                            **select(block, "name"),
                            **select(block.get("item", {}), *constants.ITEM_ID_STATUS_FIELDS),
                            "type": "function_call",
                            "call_id": block["id"],
                            "arguments": json_arguments(block["input"]),
                        }
                    )
                elif kind == "tool_result":
                    result.append(
                        {
                            **select(block.get("item", {}), *constants.ITEM_ID_STATUS_FIELDS),
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": render_content(block["content"], role="user", profile=profile)
                            if isinstance(block.get("content"), list)
                            else _tool_output(block.get("content", "")),
                        }
                    )
                else:
                    feedback_item = block.get("feedback_item")
                    if isinstance(feedback_item, dict):
                        result.append(feedback_item)
                    else:
                        if feedback_item is not None:
                            report_loss(
                                block,
                                "Responses reasoning feedback requires a native reasoning item",
                            )
                        if profile is None and block.get("reasoning"):
                            result.append(
                                {
                                    "id": f"rs_{id_prefix}{len(result)}",
                                    **select(
                                        block.get("item", {}), *constants.ITEM_ID_STATUS_FIELDS
                                    ),
                                    "type": "reasoning",
                                    "summary": reasoning_summary(block),
                                }
                            )
            flush()
            if len(result) == group_start and message["role"] != "tool":
                result.append(
                    {
                        "type": "message",
                        "role": message["role"],
                        **(
                            {"id": f"msg_{id_prefix}{len(result)}", "status": "completed"}
                            if message["role"] == "assistant"
                            else {}
                        ),
                        **select(message.get("item", {}), *constants.ITEM_ID_STATUS_FIELDS),
                        "content": [],
                    }
                )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render canonical function declarations as Responses tools and report unsupported fields."""
        return decode_functions(
            tools,
            lambda functions: [
                {
                    "type": "function",
                    **select_present(f, *constants.MIDDLEWARE_FUNCTION_METADATA_FIELDS),
                    "parameters": f.get("parameters"),
                    "strict": f.get("strict") or False,
                }
                for f in functions
            ],
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render public middleware events as native Responses stream events.

        Text/reasoning deltas are already assembled into ordered public events;
        tool events contain complete calls validated at successful source EOF.
        Use source metadata for native identity where available and report
        content that the target protocol cannot express as semantic losses.
        """
        result = []
        parameters.setdefault("stream_indices", {})
        for event in events(chunks):
            kind = event.get("type")
            if kind in ("block_start", "block_stop", "tool", "block"):
                owner = (
                    event["block"].get("protocol")
                    if event["block"].get("type") != "thinking"
                    else None
                )
                native_owner = (
                    parameters.get("source_protocol") if owner is None else owner
                ) == constants.PROTOCOL
            if (
                kind == "block"
                and not native_owner
                and event["block"].get("type") == "thinking"
                and event["block"].get("reasoning") is None
            ):
                foreign_block(event["block"])
                continue
            index, part_index, item_id, native = self._address(event, parameters)
            source_index = event["source_ref"][1] if native else event.get("index", 0)
            if kind == "start":
                result.append(
                    {
                        "type": constants.RESPONSE_CREATED,
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
                        item = self._added_item(parameters["response_items"][str(source_index)])
                    result.append(
                        {"type": constants.OUTPUT_ITEM_ADDED, "output_index": index, "item": item}
                    )
                result.append(
                    {
                        "type": constants.REASONING_SUMMARY_PART_ADDED
                        if thinking
                        else constants.CONTENT_PART_ADDED,
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index" if thinking else "content_index": part_index,
                        "part": {"type": "summary_text", "text": ""}
                        if thinking
                        else {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                    }
                )
            elif kind == "delta" and event["field"] in ("text", "reasoning"):
                thinking = event["field"] == "reasoning"
                value = {
                    "type": constants.REASONING_SUMMARY_TEXT_DELTA
                    if thinking
                    else constants.OUTPUT_TEXT_DELTA,
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
                value = block.get("reasoning" if thinking else "text") or ""
                result.append(
                    {
                        "type": constants.REASONING_SUMMARY_TEXT_DONE
                        if thinking
                        else constants.OUTPUT_TEXT_DONE,
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
                        "type": constants.REASONING_SUMMARY_PART_DONE
                        if thinking
                        else constants.CONTENT_PART_DONE,
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index" if thinking else "content_index": part_index,
                        "part": part,
                    }
                )
                if not native:
                    item = self._output_item(block, index, parameters.get("source_protocol"))
                    result.append(
                        {"type": constants.OUTPUT_ITEM_DONE, "output_index": index, "item": item}
                    )
            elif kind == "tool":
                item = (
                    {
                        **self._output_item(
                            event["block"], index, parameters.get("source_protocol")
                        ),
                        **select(
                            parameters["response_items"][str(source_index)],
                            *constants.FUNCTION_CALL_ITEM_IDENTITY_FIELDS,
                        ),
                    }
                    if native
                    else self._output_item(event["block"], index, parameters.get("source_protocol"))
                )
                result.extend(
                    [
                        {
                            "type": constants.OUTPUT_ITEM_ADDED,
                            "output_index": index,
                            "item": {**item, "status": "in_progress"},
                        },
                        {
                            "type": constants.FUNCTION_CALL_ARGUMENTS_DONE,
                            "output_index": index,
                            "item_id": item_id,
                            "arguments": item["arguments"],
                            "name": item["name"],
                        },
                    ]
                )
                if not native:
                    result.append(
                        {"type": constants.OUTPUT_ITEM_DONE, "output_index": index, "item": item}
                    )
            elif kind == "block":
                block = event["block"]
                item = self._output_item(block, index, parameters.get("source_protocol"))
                result.append(
                    {
                        "type": constants.OUTPUT_ITEM_ADDED,
                        "output_index": index,
                        "item": self._added_item(item),
                    }
                )
                if not native:
                    result.append(
                        {"type": constants.OUTPUT_ITEM_DONE, "output_index": index, "item": item}
                    )
            elif kind == "finish":
                # Truncation and filtering are incomplete Responses, not successful ends.
                incomplete = constants.MIDDLEWARE_REASON_TO_INCOMPLETE_REASON.get(event["reason"])
                if (
                    incomplete is None
                    and event["reason"] not in constants.STOP_AND_TOOL_CALLS_REASONS
                ):
                    raise ResponseFailedError(
                        "Responses cannot represent the source finish reason",
                        details={"reason": event["reason"]},
                    )
                output = self._completed_output(
                    event, parameters.get("source_protocol"), parameters
                )
                if self._native_items(parameters):
                    represented = {
                        parameters["stream_indices"][("item", entry["source_ref"][1])]
                        for entry in event.get("output_blocks", [])
                        if entry.get("source_ref") is not None
                    }
                    for output_index, item in enumerate(output):
                        if output_index not in represented:
                            result.append(
                                {
                                    "type": constants.OUTPUT_ITEM_ADDED,
                                    "output_index": output_index,
                                    "item": self._added_item(item),
                                }
                            )
                        result.append(
                            {
                                "type": constants.OUTPUT_ITEM_DONE,
                                "output_index": output_index,
                                "item": item,
                            }
                        )
                result.append(
                    {
                        "type": constants.RESPONSE_INCOMPLETE
                        if incomplete
                        else constants.RESPONSE_COMPLETED,
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
        """Render canonical token counters in the Responses usage format."""
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

    def result(self, messages: list, *, final_event: dict | None = None, **parameters: Any) -> list:
        """Render completed indexed output, or canonical messages for a partial result."""
        if final_event is None:
            return self._render_messages(wire_values(messages), **parameters)
        final_event = output_chunks([final_event], parameters.pop("semantic_profile", None))[0]
        parameters.setdefault("stream_indices", {})
        return self._completed_output(final_event, parameters.get("source_protocol"), parameters)

    def _response_envelope(self, **parameters: Any) -> dict:
        # Required native envelope defaults carry no unsupported source metadata.
        """Build a minimal native response envelope from supported metadata."""
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
            "created_at": parameters.get("created_at", 0),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

    def _output_item(self, block: dict, index: int, origin: str | None) -> dict:
        """Render a completed block without message-level empty-content fallback."""
        if block.get("type") == "thinking":
            feedback = block.get("feedback_item")
            if origin == constants.PROTOCOL and isinstance(feedback, dict):
                item = capture(feedback)
            else:
                if origin != constants.PROTOCOL:
                    foreign_block(block)
                elif feedback is not None:
                    report_loss(
                        block, "Responses reasoning feedback requires a native reasoning item"
                    )
                item = {"type": "reasoning", "summary": reasoning_summary(block)}
                if origin == constants.PROTOCOL:
                    item.update(select(block.get("item", {}), *constants.ITEM_ID_STATUS_FIELDS))
        else:
            item = self._messages([{"role": "assistant", "content": [block], "protocol": origin}])[
                0
            ]
        if origin != constants.PROTOCOL:
            item["id"] = f"item_{index}"
        item.setdefault("id", f"item_{index}")
        if origin != constants.PROTOCOL or item["type"] != "reasoning":
            item.setdefault("status", "completed")
        return item

    def _completed_output(self, event: dict, origin: str | None, parameters: dict) -> list[dict]:
        """Render the indexed final snapshot using the addresses already emitted."""
        entries = event.get("output_blocks", [])
        if not self._native_items(parameters):
            result = []
            for entry in entries:
                block = entry["block"]
                if (
                    origin != constants.PROTOCOL
                    and block.get("type") == "thinking"
                    and block.get("reasoning") is None
                ):
                    foreign_block(block)
                    continue
                index, _, _, _ = self._address(entry, parameters)
                result.append((index, self._output_item(block, index, origin)))
            return [item for _, item in sorted(result, key=lambda pair: pair[0])]
        grouped = {}
        for entry in entries:
            ref = entry.get("source_ref")
            index = ref[1] if ref is not None else entry["index"]
            grouped.setdefault(index, []).append(entry["block"])
            self._address(entry, parameters)
        result = []
        for source_index, header in sorted(
            parameters["response_items"].items(), key=lambda pair: int(pair[0])
        ):
            source_index = int(source_index)
            blocks = grouped.get(source_index, [])
            kind = header.get("type")
            if kind == "message" and (blocks or header.get("empty")):
                item = {
                    "type": kind,
                    "role": "assistant",
                    "content": render_content(blocks, role="assistant"),
                }
            elif kind in constants.REASONING_FUNCTION_CALL_ITEM_TYPES and blocks:
                item = self._output_item(blocks[0], source_index, origin)
            else:
                continue
            item.update(select(header, *constants.FUNCTION_CALL_ITEM_IDENTITY_FIELDS))
            indices = parameters["stream_indices"]
            key = ("item", source_index)
            if key in indices:
                index = indices[key]
            else:
                index, _, _, _ = self._address({"source_ref": (kind, source_index, 0)}, parameters)
            item.setdefault("id", f"item_{index}")
            if kind != "reasoning":
                item.setdefault("status", "completed")
            result.append((index, item))
        return [item for _, item in sorted(result, key=lambda pair: pair[0])]

    def _native_items(self, parameters: dict) -> bool:
        """Whether source Responses metadata supports preserving native item identities."""
        return parameters.get("source_protocol") == constants.PROTOCOL and bool(
            parameters.get("response_items")
        )

    def _address(self, event: dict, parameters: dict) -> tuple[int, int, str, bool]:
        """Map source blocks/items and parts to stable, contiguous target addresses."""
        if event.get("type") in ("start", "finish"):
            return 0, 0, "item_0", False
        indices = parameters["stream_indices"]
        ref = event.get("source_ref")
        native = self._native_items(parameters) and ref is not None
        key = ("item", ref[1]) if native else ("block", event.get("index", 0))
        if key not in indices:
            indices[key] = indices.get(("next_item",), 0)
            indices[("next_item",)] = indices[key] + 1
        index = indices[key]
        if not native:
            return index, 0, f"item_{index}", False
        part_key = ("part", ref[1], ref[2])
        if part_key not in indices:
            counter = ("next_part", ref[1])
            indices[part_key] = indices.get(counter, 0)
            indices[counter] = indices[part_key] + 1
        parent = parameters["response_items"][str(ref[1])]
        return index, indices[part_key], parent.get("id", f"item_{index}"), True

    def _added_item(self, item: dict) -> dict:
        """Build an in-progress item header with empty content, summary or arguments."""
        result = select(item, *constants.ITEM_HEADER_FIELDS)
        result["status"] = "in_progress"
        if item["type"] == "message":
            result.setdefault("role", "assistant")
            result["content"] = []
        elif item["type"] == "reasoning":
            result["summary"] = []
        elif item["type"] == "function_call":
            result["arguments"] = ""
        return result
