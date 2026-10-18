"""openai/chat: intermediate to native output."""

from typing import Any

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.chat.decoding.content import (
    chat_call,
    foreign_block,
    render_content,
)
from my_llm.codec.shared.conversion import (
    OPAQUE,
    Translations,
    blocks,
    coalesce_groups,
    decode_functions,
    extras,
    original,
    project_message,
    select_present,
    source_protocol,
    tool_output,
)
from my_llm.codec.shared.options import (
    PreparedOptions,
    merge_bindings,
    overlay_request_parameters,
)
from my_llm.codec.shared.stream import counts, events
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"


def thinking_enabled(profile: ResolvedProfile, parameters: dict) -> bool:
    if parameters.get("reasoning") is True or parameters.get("reasoning_effort", "none") != "none":
        return True
    body = merge_dicts(profile.get("extra_body", {}), parameters.get("extra_body"))
    thinking = body.get("thinking", {})
    return (
        isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
    ) or body.get("enable_thinking") is True


class ChatCompletionsDecoder(BaseDecoder):
    """Translate intermediate values into protocol dictionaries, retaining replay data."""

    __slots__ = ()
    protocol = PROTOCOL

    def request_parameters(self, profile: ResolvedProfile, options: PreparedOptions) -> dict:
        request = merge_bindings(
            *(
                {key: value for key, value in binding.items() if key != "extra_body"}
                for binding in options.bindings
            )
        )
        if profile.get("usage_mode", "none") == "stream_options":
            request["stream_options"] = {"include_usage": True}
        return overlay_request_parameters(
            request, profile, options, overlay="extra_body", headers_field="extra_headers"
        )

    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        options: PreparedOptions,
        **parameters: Any,
    ) -> dict:
        result = self.request_parameters(profile, options)
        result.update(model=profile.model, stream=True, messages=messages)
        if tools is not None:
            result["tools"] = tools

        return result

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> Translations:
        result = Translations()
        for sources, message in coalesce_groups(messages):
            result.group(
                sources,
                structure={
                    "content": [{"image_url": {}, "file": {}}],
                    "tool_calls": [{"function": {}}],
                },
            )
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
        if profile is not None and profile.get("replay_reasoning_content_when_tool_calls", False):
            if thinking_enabled(profile, parameters):
                for value in result:
                    if value.get("role") == "assistant" and value.get("tool_calls"):
                        value.setdefault("reasoning_content", "")
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> Translations:
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
            structure={"function": {}},
        )

    def _chunks(
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

    def _envelope(self, **parameters: Any) -> dict:
        return {
            "id": parameters.get("response_id", "generated"),
            "model": parameters.get("model", ""),
            "created": parameters.get("created", 0),
        }
