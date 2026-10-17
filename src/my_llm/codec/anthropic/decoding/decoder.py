"""anthropic: intermediate to native output."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from inspect import signature
from typing import Any

from anthropic.resources.messages import AsyncMessages
from openai import Omit

from my_llm.codec.anthropic.decoding.content import foreign_block
from my_llm.codec.anthropic.decoding.transport import open_stream
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    extras,
    json_arguments,
    original,
    source_protocol,
)
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.options import native_options
from my_llm.codec.shared.rendering import project_message, select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.codec.shared.usage import counts
from my_llm.exceptions import ConfigurationError, ResponseFailedError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"
PROTECTED = {"model", "stream", "messages", "input", "tools", "system", "previous_response_id"}
REQUEST_FIELDS = (
    set(signature(AsyncMessages.create).parameters)
    - PROTECTED
    - {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
)
REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class AnthropicDecoder(BaseDecoder):
    """Render intermediate values and build request-owned native SDK calls."""

    __slots__ = ()
    usage_modes = ("none", "native")

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
            else:
                if source_protocol(message) != PROTOCOL:
                    message = project_message(message, foreign_block)
                content = blocks(message.get("content"))
                result.append({**extras(message, "parts", "tool_calls"), "content": content})
        return result

    def tools(
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
        )

    def chunks(
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

    def profile_defaults(self) -> dict:
        return {
            "replay_reasoning_content_when_tool_calls": True,
            "supports_reasoning_effort": False,
            "content_path": "$.delta.text",
            "reasoning_content_path": "$.delta.thinking",
            "reasoning_enable_parameter": {},
            "reasoning_effort_parameter": {},
        }

    def request_parameters(self, profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
        tokens, headers, body, effort = native_options(
            profile, request_fields=REQUEST_FIELDS, protected=PROTECTED, **parameters
        )
        if tokens is None:
            raise ConfigurationError("Anthropic requires max_tokens in the model or generation")
        request = {**body, "max_tokens": tokens}
        if headers:
            request["extra_headers"] = headers
        if effort:
            self._apply_reasoning(request, body, profile, tokens, effort)
        return request

    def assemble_request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        previous = parameters.pop("previous_response_id", None)
        if previous is not None and not isinstance(previous, Omit):
            raise ConfigurationError("previous_response_id requires OpenAI Responses API")
        result = self.request_parameters(profile, **parameters)
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        if not history:
            raise ConfigurationError("a non-system message is required")
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

    def upstream(
        self, request: dict, profile: ResolvedProfile, **parameters: Any
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        return open_stream(request, profile, **parameters)

    def _apply_reasoning(
        self,
        request: dict[str, Any],
        body: dict[str, Any],
        profile: ResolvedProfile,
        tokens: int,
        effort: str,
    ) -> None:
        thinking = body.get("thinking", {"type": "enabled", "budget_tokens": 1024})
        if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "adaptive"):
            raise ConfigurationError("reasoning requires enabled or adaptive Anthropic thinking")
        if thinking.get("type") == "enabled" and "budget_tokens" not in thinking:
            thinking = {**thinking, "budget_tokens": 1024}
        if thinking.get("type") == "enabled" and (
            not isinstance(thinking.get("budget_tokens"), int)
            or thinking["budget_tokens"] < 1024
            or thinking["budget_tokens"] >= tokens
        ):
            raise ConfigurationError("Anthropic thinking budget must be >=1024 and < max_tokens")
        request["thinking"] = thinking
        if profile.get("supports_reasoning_effort", False):
            if effort not in ("low", "medium", "high", "xhigh", "max"):
                raise UnsupportedCapabilityError(
                    "map reasoning effort to an Anthropic-supported level"
                )
            request["output_config"] = {**body.get("output_config", {}), "effort": effort}
