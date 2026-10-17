"""openai/responses: intermediate to native output."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from typing import Any

from openai import Omit

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.responses.decoding.content import foreign_block, render_content
from my_llm.codec.openai.responses.decoding.transport import open_stream
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import (
    dictionaries,
    extras,
    json_arguments,
    original,
    source_protocol,
    tool_output,
)
from my_llm.codec.shared.keys import OPAQUE, SOURCE
from my_llm.codec.shared.options import (
    bindings,
    check_protected,
    overlay_generated,
    resolve_options,
)
from my_llm.codec.shared.rendering import project_message, select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.codec.shared.usage import counts
from my_llm.codec.shared.values import _expand
from my_llm.exceptions import ConfigurationError, ResponseFailedError, UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_responses"
PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


class ResponsesDecoder(BaseDecoder):
    """Render intermediate values and build request-owned native SDK calls."""

    __slots__ = ()
    usage_modes = ("none", "native")

    def messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for message in dictionaries(messages):
            if message.get(OPAQUE):
                result.extend(original(message, PROTOCOL))
                continue
            native = original(message, PROTOCOL)
            if native is not None:
                result.extend(native)
                continue
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
                            **extras(message, "role", "content"),
                            "type": "message",
                            **(
                                {"id": f"msg_{len(result)}", "status": "completed"}
                                if message["role"] == "assistant" and not input_media
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
                    pending.append(block)
                    continue
                flush()
                if kind == "tool_use":
                    result.append(
                        {
                            **extras(block, "type", "id", "input", "thought_signature"),
                            "type": "function_call",
                            "call_id": block["id"],
                            "arguments": json_arguments(block["input"]),
                        }
                    )
                elif kind == "tool_result":
                    result.append(
                        {
                            **extras(block, "type", "tool_use_id", "content", "name"),
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": tool_output(block.get("content", "")),
                        }
                    )
                else:
                    result.append(
                        {
                            **extras(block, "type", "thinking", "signature", "thought_signature"),
                            "type": "reasoning",
                            "id": block.get("id", f"rs_{len(result)}"),
                            "summary": block.get(
                                "summary",
                                [{"type": "summary_text", "text": block.get("thinking", "")}],
                            ),
                        }
                    )
            flush()
        return result

    def tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            "openai_responses",
            lambda functions: [
                {
                    "type": "function",
                    **select_present(f, "name", "description", "parameters", "strict"),
                    "strict": f.get("strict") or False,
                }
                for f in functions
            ],
        )

    def chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        passthrough = set()
        for event in events(chunks, PROTOCOL):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            index, part_index, item_id, native = self._address(event, parameters)
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
                    block.get("type") == "thinking"
                    and parameters.get("source_protocol") == PROTOCOL
                )
                refusal = block.get("type") == "refusal"
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
                    item = self._output_item(block, index, parameters.get("source_protocol"))
                    result.append(
                        {"type": "response.output_item.done", "output_index": index, "item": item}
                    )
            elif kind == "tool":
                item = (
                    deepcopy(parameters["response_items"][str(index)])
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
                output = (
                    self.messages(event.get("messages", []))
                    if self._native_items(parameters)
                    else self._completed_output(event, parameters.get("source_protocol"))
                )
                if self._native_items(parameters):
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
            if id(event) not in passthrough:
                event.setdefault("sequence_number", sequence + offset)
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        native = original(value, "openai_responses")
        if native is not None:
            return (
                native
                if {
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_tokens_details",
                    "output_tokens_details",
                }
                <= native.keys()
                else None
            )
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

    def profile_defaults(self) -> dict:
        return {
            "replay_reasoning_content_when_tool_calls": True,
            "supports_reasoning_effort": True,
            "content_path": "response.output_text.delta",
            "reasoning_content_path": "response.reasoning_summary_text.delta",
            "reasoning_enable_parameter": {
                "reasoning": {"summary": "detailed", "effort": "medium"}
            },
            "reasoning_effort_parameter": {
                "reasoning": {"summary": "detailed", "effort": "{effort}"}
            },
        }

    def result(self, messages: list) -> list:
        return messages

    def allows_external_calls(self, previous_response_id) -> bool:
        return previous_response_id is not None and not isinstance(previous_response_id, Omit)

    def request_parameters(self, profile: ResolvedProfile, **parameters: Any) -> dict:
        tokens, headers, body, effort = resolve_options(profile, protected=PROTECTED, **parameters)
        request = {}
        for index, binding in enumerate(bindings(profile, effort, expand=False)):
            binding = {key: value for key, value in binding.items() if not isinstance(value, Omit)}
            if set(binding) - {"reasoning", "extra_body"}:
                raise ConfigurationError("reasoning bindings contain unsupported SDK parameters")
            extra = _expand(binding.get("extra_body", {}), effort)
            check_protected(extra, PROTECTED)
            body = merge_dicts(body, extra)
            if "reasoning" in binding:
                value = binding["reasoning"]
                if not isinstance(value, dict):
                    raise ConfigurationError("reasoning must be a dictionary")
                combined = merge_dicts({"summary": "detailed"}, request.get("reasoning"), value)
                if index == 1 and "effort" not in value:
                    combined["effort"] = effort
                request["reasoning"] = _expand(combined, effort)
        if tokens is not None:
            request["max_output_tokens"] = tokens
        body = overlay_generated(request, body)
        if headers:
            request["extra_headers"] = headers
        if body:
            request["extra_body"] = body
        return request

    def assemble_request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        previous = parameters.pop("previous_response_id", None)
        has_previous = previous is not None and not isinstance(previous, Omit)
        if has_previous and (not isinstance(previous, str) or not previous):
            raise ConfigurationError("previous_response_id requires Responses and a non-empty ID")
        result = self.request_parameters(profile, **parameters)
        result.update(model=profile.model, stream=True, input=messages)
        if tools is not None:
            result["tools"] = tools
        if has_previous:
            result["previous_response_id"] = previous
        return result

    def upstream(
        self, request: dict, profile: ResolvedProfile, **parameters: Any
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        return open_stream(request, profile, **parameters)

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

    def _output_item(self, block: dict, index: int, origin: str | None) -> dict:
        item = self.messages(
            [{"role": "assistant", "content": [block], SOURCE: {"protocol": origin}}]
        )[0]
        item.update(id=f"item_{index}", status="completed")
        return item

    def _completed_output(self, event: dict, origin: str | None) -> list[dict]:
        if "output_blocks" not in event:
            return self.messages(event.get("messages", []))
        result = [
            self._output_item(entry["block"], entry["index"], origin)
            for entry in event["output_blocks"]
            if origin == PROTOCOL or entry["block"].get("type") != "redacted_thinking"
        ]
        result.extend(
            extras(message, OPAQUE) for message in event["messages"] if message.get(OPAQUE)
        )
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
