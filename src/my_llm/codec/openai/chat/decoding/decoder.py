"""openai/chat: intermediate to native output."""

from typing import Any

from openai import Omit

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.chat.decoding.content import (
    chat_call,
    foreign_block,
    render_content,
    thinking_enabled,
)
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    extras,
    original,
    source_protocol,
    tool_output,
)
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.options import (
    apply_max_tokens,
    bindings,
    check_protected,
    overlay_generated,
    resolve_options,
)
from my_llm.codec.shared.rendering import project_message, select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.codec.shared.usage import counts
from my_llm.exceptions import ConfigurationError, ResponseFailedError, UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"
PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


class ChatCompletionsDecoder(BaseDecoder):
    """Render intermediate values and assemble native request dictionaries."""

    __slots__ = ()
    usage_modes = ("none", "stream_options")

    def messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for message in coalesce_messages(messages):
            if message.get(OPAQUE):
                result.extend(original(message, PROTOCOL))
                continue
            native = original(message, PROTOCOL)
            if native is not None:
                result.extend(native)
                continue
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
            content = blocks(message.get("content"))
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            other = [b for b in content if b.get("type") != "tool_result"]
            if other or not tool_results:
                value = {
                    **extras(message, "content", "parts", "tool_calls"),
                    "content": render_content(
                        [b for b in other if b.get("type") != "refusal"], role=message["role"]
                    ),
                }
                refusals = [b.get("refusal", "") for b in other if b.get("type") == "refusal"]
                if refusals:
                    value["refusal"] = "".join(refusals)
                calls = [chat_call(b) for b in other if b.get("type") == "tool_use"]
                if calls:
                    value["tool_calls"] = calls
                annotations = [annotation for b in other for annotation in b.get("annotations", [])]
                if annotations:
                    value["annotations"] = annotations
                reasoning = "".join(
                    b.get("thinking", "") for b in other if b.get("type") == "thinking"
                )
                if reasoning:
                    value["reasoning_content"] = reasoning
                result.append(value)
            result.extend(
                {
                    **extras(b, "type", "tool_use_id", "name"),
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": tool_output(b.get("content")),
                }
                for b in tool_results
            )
        if profile:
            for value in result:
                if not profile.get("supports_message_name", False):
                    value.pop("name", None)
                if value.get("role") != "assistant":
                    continue
                if not profile.get("replay_reasoning_content_when_tool_calls", False):
                    value.pop("reasoning_content", None)
                elif value.get("tool_calls") and thinking_enabled(profile, parameters):
                    # Only active tool-call turns need a missing reasoning field padded.
                    value.setdefault("reasoning_content", "")
        return result

    def tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            "openai_chat",
            lambda functions: [
                {
                    "type": "function",
                    "function": select_present(f, "name", "description", "parameters", "strict"),
                }
                for f in functions
            ],
        )

    def chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks, PROTOCOL):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            if kind == "opaque":
                result.append(event["value"])
                continue
            delta = {}
            finish = None
            if kind == "start":
                delta = {"role": "assistant"}
            elif kind == "delta":
                field = {
                    "text": "content",
                    "thinking": "reasoning_content",
                    "refusal": "refusal",
                }.get(event["field"])
                if event["field"] == "thinking" and parameters.get("source_protocol") != PROTOCOL:
                    field = "content"
                if field:
                    delta = {field: event["value"]}
            elif kind == "tool":
                block = event["block"]
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                delta = {"tool_calls": [{**chat_call(block), "index": event.get("tool_index", 0)}]}
            elif kind == "finish":
                finish = event["reason"]
                if finish not in (
                    "stop",
                    "length",
                    "tool_calls",
                    "content_filter",
                    "function_call",
                ):
                    raise ResponseFailedError(
                        "Chat cannot represent the source finish reason", details={"reason": finish}
                    )
            if not delta and kind != "finish":
                continue
            value = {
                **self._envelope(**parameters),
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            value["usage"] = None
            result.append(value)
            if kind == "finish" and usage is not None:
                result.append({**value, "choices": [], "usage": usage})
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        native = original(value, "openai_chat")
        if native is not None:
            return (
                native
                if {"completion_tokens", "total_tokens", "prompt_tokens"} <= native.keys()
                else None
            )
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        if input_tokens is None or output_tokens is None:
            return None
        result = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total,
        }
        if cached is not None:
            result["prompt_tokens_details"] = {"cached_tokens": cached}
        if reasoning is not None:
            result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
        return result

    def profile_defaults(self) -> dict:
        return {
            "replay_reasoning_content_when_tool_calls": False,
            "supports_reasoning_effort": False,
            "max_tokens_path": "$.max_completion_tokens",
            "content_path": "$.choices[0].delta.content",
            "reasoning_content_path": "$.choices[0].delta.reasoning_content",
            "reasoning_enable_parameter": {"reasoning_effort": "medium"},
            "reasoning_effort_parameter": {"reasoning_effort": "{effort}"},
        }

    def request_parameters(self, profile: ResolvedProfile, **parameters: Any) -> dict:
        tokens, headers, body, effort = resolve_options(profile, protected=PROTECTED, **parameters)
        request = {}
        for binding in bindings(profile, effort):
            if set(binding) - {"reasoning_effort", "extra_body"}:
                raise ConfigurationError("reasoning bindings contain unsupported SDK parameters")
            extra = binding.get("extra_body", {})
            check_protected(extra, PROTECTED)
            body = merge_dicts(body, extra)
            if "reasoning_effort" in binding:
                value = binding["reasoning_effort"]
                if not isinstance(value, str):
                    raise ConfigurationError("reasoning_effort must be a string")
                request["reasoning_effort"] = value
        if body.get("n") not in (None, 1):
            raise UnsupportedCapabilityError("only one generated candidate is supported")
        if profile.get("usage_mode", "none") == "stream_options":
            request["stream_options"] = {"include_usage": True}
        body = overlay_generated(request, body)
        if headers:
            request["extra_headers"] = headers
        if body:
            request["extra_body"] = body
        apply_max_tokens(request, profile, tokens, protected=PROTECTED)
        return request

    def assemble_request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        previous = parameters.pop("previous_response_id", None)
        has_previous = previous is not None and not isinstance(previous, Omit)
        if has_previous:
            raise ConfigurationError("previous_response_id requires Responses and a non-empty ID")
        result = self.request_parameters(profile, **parameters)
        result.update(model=profile.model, stream=True, messages=messages)
        if tools is not None:
            result["tools"] = tools

        return result

    def _envelope(self, **parameters: Any) -> dict:
        return {
            "id": parameters.get("response_id", "generated"),
            "model": parameters.get("model", ""),
            "created": parameters.get("created", 0),
        }
