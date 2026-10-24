"""anthropic: intermediate to native output."""

from typing import Any

from anthropic.resources.messages import AsyncMessages
from anthropic.types import (
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
)
from pydantic import ValidationError

from my_llm.basic.data import capture
from my_llm.basic.exceptions import ConfigurationError, MessageValidationError, ResponseFailedError
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.anthropic.content import native_block
from my_llm.codec.anthropic.decoding.content import foreign_block
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.shared.conversion import (
    coalesce_messages,
    decode_functions,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.schema import _plan, adapter, parameter_types
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens

PROTECTED = {"model", "stream", "messages", "input", "tools", "system", "previous_response_id"}

REQUEST_SCHEMA = parameter_types(AsyncMessages.create)

PROTOCOL = "anthropic"
REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class AnthropicDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    tool_type = ToolParam
    message_type = MessageParam
    message_path = ("messages",)
    tool_path = ("tools",)

    request_schema = REQUEST_SCHEMA

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        # The SDK's InputSchema TypedDict names only some JSON Schema keys.
        # Delegate this open leaf so it is captured once rather than filtered
        # and copied before replacing it with the full application schema.
        apply = _plan(self.tool_type, delegated_paths=(("input_schema",),)).apply
        result = [apply(tool) for tool in tools]
        for original, filtered in zip(tools, result):
            # InputSchema is an open JSON Schema, not just the SDK TypedDict's
            # named type/properties/required fields. Keep its application keys.
            if "input_schema" in original:
                filtered["input_schema"] = capture(original["input_schema"])
        return result

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(profile, options, protected=PROTECTED, containers=())

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        if not request["messages"]:
            raise ConfigurationError("a non-system message is required")
        if request.get("max_tokens") is None:
            raise ConfigurationError("Anthropic requires max_tokens in the model or generation")
        validate_max_tokens(request["max_tokens"])
        if "system" in request and not isinstance(request["system"], str):
            try:
                validator = adapter(list[TextBlockParam])
                parsed = validator.validate_python(request["system"])
                validator.dump_json(parsed, warnings="error")
            except (ValidationError, TypeError, ValueError) as exc:
                raise MessageValidationError(
                    "system does not conform to Anthropic text blocks"
                ) from exc
        for field, schema in (
            ("thinking", ThinkingConfigParam),
            ("output_config", OutputConfigParam),
        ):
            if field not in request:
                continue
            try:
                adapter(schema).validate_python(request[field], strict=True)
            except (ValidationError, TypeError, ValueError) as exc:
                raise ConfigurationError(f"invalid Anthropic {field}") from exc
        thinking = request.get("thinking", {})
        if (
            (binding := profile.parameter("reasoning_enable", "input")) is not None
            and binding.path == "$.thinking"
            and (
                parameters.get("reasoning") is True
                or parameters.get("reasoning_effort", "none") != "none"
            )
            and thinking.get("type") not in ("enabled", "adaptive")
        ):
            raise ConfigurationError("reasoning requires enabled or adaptive Anthropic thinking")
        if thinking.get("type") == "enabled":
            budget = thinking["budget_tokens"]
            tokens = request.get("max_tokens")
            if (
                isinstance(budget, bool)
                or budget < 1024
                or (tokens is not None and budget >= tokens)
            ):
                raise ConfigurationError(
                    "Anthropic thinking budget must be >=1024 and < max_tokens"
                )
        return request

    body_overlay = "extra_body"

    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        request: dict,
        **parameters: Any,
    ) -> dict:
        result = request
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

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for index, message in enumerate(coalesce_messages(messages)):
            if message.get("name") is not None:
                report_loss(
                    message["name"],
                    "Anthropic cannot represent message names",
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            context_owner = source_protocol(message)
            content = []
            for block in message["content"]:
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                    if block is None:
                        continue
                if block.get("type") == "thinking" and not block.get("signature"):
                    report_loss(block, "Anthropic reasoning requires a native signature")
                    continue
                value = native_block(block)
                if value is not None:
                    content.append(value)
            result.append({"role": message["role"], "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [
                {
                    **select_present(f, "name", "description", "strict"),
                    "input_schema": f["parameters"]
                    if f.get("parameters") is not None
                    else {"type": "object"},
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
            index = event.get("index", 0)
            if kind == "start":
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
                        },
                    }
                )
            elif kind == "block_start":
                block = event["block"]
                if block.get("type") == "thinking" and not native_owner:
                    report_loss(
                        block, "Anthropic cannot replay reasoning without its native signature"
                    )
                    continue
                result.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                        if block["type"] == "thinking" and native_owner
                        else {"type": "text", "text": ""},
                    }
                )
            elif kind == "delta":
                field = {"text": "text", "thinking": "thinking"}.get(event.get("field"))
                if field is None:
                    continue
                if field == "thinking" and parameters.get("source_protocol") != PROTOCOL:
                    continue
                value = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": f"{field}_delta", field: event["value"]},
                }
                result.append(value)
            elif kind == "block_stop":
                if event["block"].get("type") == "thinking":
                    if not native_owner:
                        report_loss(
                            event["block"],
                            "Anthropic cannot replay reasoning without its native signature",
                        )
                        continue
                    if not event["block"].get("signature"):
                        report_loss(
                            event["block"], "Anthropic reasoning requires a native signature"
                        )
                signature = event["block"].get("signature")
                if signature and native_owner:
                    result.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "signature_delta", "signature": signature},
                        }
                    )
                result.append({"type": "content_block_stop", "index": index})
            elif kind == "tool":
                block = event["block"]
                if not native_owner:
                    block = foreign_block(block)
                block = native_block(block)
                result.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": capture(block),
                        },
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            elif kind == "block":
                block = event["block"]
                if not native_owner:
                    block = foreign_block(block)
                    if block is None:
                        continue
                result.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": capture(native_block(block)),
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
                        },
                        {"type": "message_stop"},
                    ]
                )
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
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
