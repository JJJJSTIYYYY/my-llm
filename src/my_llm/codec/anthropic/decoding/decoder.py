"""anthropic: intermediate to native output."""

from copy import deepcopy
from typing import Any

from my_llm.codec.anthropic.decoding.content import foreign_block
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.shared.conversion import (
    OPAQUE,
    Translations,
    blocks,
    coalesce_groups,
    decode_functions,
    extras,
    json_arguments,
    original,
    project_message,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import PreparedOptions, apply_max_tokens
from my_llm.codec.shared.stream import counts, events
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"
REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class AnthropicDecoder(BaseDecoder):
    """Translate intermediate values into protocol dictionaries, retaining replay data."""

    __slots__ = ()
    protocol = PROTOCOL

    def request_parameters(
        self, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict[str, Any]:
        tokens, headers, effort = options.tokens, options.headers, options.effort
        body = deepcopy(options.body)
        request = dict(body)
        if headers:
            request["extra_headers"] = deepcopy(headers)
        if effort:
            self._apply_reasoning(request, body, profile, effort)
        apply_max_tokens(request, profile, tokens)
        return request

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
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        result.update(model=profile.model, stream=True, messages=history)
        if system:
            result["system"] = [
                block
                for message in system
                for block in (
                    [{"type": "text", "text": message["content"]}]
                    if isinstance(message["content"], str)
                    else message["content"]
                )
            ]
        if tools is not None:
            result["tools"] = tools
        return result

    def _apply_reasoning(
        self,
        request: dict[str, Any],
        body: dict[str, Any],
        profile: ResolvedProfile,
        effort: str,
    ) -> None:
        thinking = body.get("thinking", {"type": "enabled", "budget_tokens": 1024})
        if thinking.get("type") == "enabled" and "budget_tokens" not in thinking:
            thinking = {**thinking, "budget_tokens": 1024}
        request["thinking"] = thinking
        if profile.get("supports_reasoning_effort", False):
            request["output_config"] = {**body.get("output_config", {}), "effort": effort}

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = Translations()
        for sources, message in coalesce_groups(messages):
            result.group(sources, structure={"content": [{"source": {}}]})
            if message.get(OPAQUE):
                result.extend(original(message, PROTOCOL))
                continue
            native = original(message, PROTOCOL)
            if native is not None:
                result.extend(native)
            else:
                if source_protocol(message) != PROTOCOL:
                    message = project_message(message, foreign_block)
                content = blocks(message.get("content"))
                result.append({**extras(message, "parts", "tool_calls"), "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            "anthropic",
            lambda functions: [
                {
                    **select_present(f, "name", "description", "strict"),
                    "input_schema": f.get("parameters") or {"type": "object"},
                }
                for f in functions
            ],
            structure={},
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks, PROTOCOL):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            index = event.get("index", 0)
            if kind == "opaque":
                result.append(event["value"])
            elif kind == "start":
                result.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": parameters.get("response_id", "generated"),
                            "type": "message",
                            "role": "assistant",
                            "model": parameters.get("model", ""),
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0, **(usage or {})},
                        },
                    }
                )
            elif kind == "block_start":
                block = event["block"]
                result.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                        if block["type"] == "thinking"
                        and parameters.get("source_protocol") == PROTOCOL
                        else {"type": "text", "text": ""},
                    }
                )
            elif kind == "delta":
                field = event["field"]
                if field == "refusal" or (
                    field == "thinking" and parameters.get("source_protocol") != PROTOCOL
                ):
                    field = "text"
                result.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": f"{field}_delta", field: event["value"]},
                    }
                )
            elif kind == "block_annotation":
                if event["field"] == "citations" and parameters.get("source_protocol") == PROTOCOL:
                    result.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "citations_delta", "citation": event["value"]},
                        }
                    )
            elif kind == "block_stop":
                signature = event["block"].get("signature")
                if signature and parameters.get("source_protocol") == PROTOCOL:
                    result.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "signature_delta", "signature": signature},
                        }
                    )
                result.append({"type": "content_block_stop", "index": index})
            elif kind == "tool":
                block = extras(event["block"], "thought_signature")
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                result.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {**block, "input": {}},
                        },
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json_arguments(block["input"]),
                            },
                        },
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            elif kind == "block":
                block = event["block"]
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                    if block is None:
                        continue
                result.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": extras(block, "thought_signature"),
                        },
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            elif kind == "finish":
                reason = REASONS.get(event["reason"], event["reason"])
                stop_sequence = None
                if parameters.get("source_protocol") == PROTOCOL:
                    reason = parameters.get("native_stop_reason", reason)
                    stop_sequence = parameters.get("native_stop_sequence")
                elif event["reason"] not in REASONS:
                    raise ResponseFailedError(
                        "Anthropic cannot represent the source finish reason",
                        details={"reason": event["reason"]},
                    )
                result.extend(
                    [
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
                            "usage": usage or {"output_tokens": 0},
                        },
                        {"type": "message_stop"},
                    ]
                )
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        native = original(value, "anthropic")
        if native is not None:
            return native if {"output_tokens"} <= native.keys() else None
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        if output_tokens is None:
            return None
        result = {"output_tokens": output_tokens}
        if input_tokens is not None:
            result["input_tokens"] = input_tokens - (cached or 0) - (cache_write or 0)
        if cached is not None:
            result["cache_read_input_tokens"] = cached
        if cache_write is not None:
            result["cache_creation_input_tokens"] = cache_write
        return result
