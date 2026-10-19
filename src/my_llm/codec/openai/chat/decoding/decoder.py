"""openai/chat: intermediate to native output."""

from typing import Any

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.chat.decoding.content import (
    chat_call,
    foreign_block,
    render_content,
)
from my_llm.codec.openai.chat.utils.content import annotations as render_annotations
from my_llm.codec.openai.chat.utils.content import text_blocks
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    decode_functions,
    project_message,
    select,
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
    """Translate recognized intermediate values into protocol dictionaries."""

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
    ) -> list[dict]:
        result = []
        for message in coalesce_messages(messages):
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
            content = blocks(message.get("content"))
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            other = [b for b in content if b.get("type") != "tool_result"]
            if other or not tool_results:
                value = {
                    **select(message, "role", "name"),
                    **(
                        {
                            "audio": select(
                                message["audio"], "id", "expires_at", "data", "transcript"
                            )
                        }
                        if message.get("audio") is not None
                        else {}
                    ),
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
                citations = render_annotations(
                    [
                        *message.get("annotations", []),
                        *(annotation for b in other for annotation in b.get("annotations", [])),
                    ]
                )
                if citations:
                    value["annotations"] = citations
                reasoning = "".join(
                    b.get("thinking", "") for b in other if b.get("type") == "thinking"
                )
                if reasoning:
                    value["reasoning_content"] = reasoning
                result.append(value)
            result.extend(
                {
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": text_blocks(b["content"])
                    if isinstance(b.get("content"), list)
                    else tool_output(b.get("content")),
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
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [
                {
                    "type": "function",
                    "function": select_present(f, "name", "description", "parameters", "strict"),
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
