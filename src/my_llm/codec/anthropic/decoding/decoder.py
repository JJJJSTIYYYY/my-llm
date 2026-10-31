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

from my_llm.basic.data import capture, dictionary_views
from my_llm.basic.exceptions import ConfigurationError, MessageValidationError, ResponseFailedError
from my_llm.basic.semantic_loss import report_loss
from my_llm.cache.compiled_cache import _MISSING, _plan, adapter, parameter_types
from my_llm.codec.anthropic.decoding.content import decode_block, foreign_block
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.shared.capabilities import content_capability, supports_content
from my_llm.codec.shared.conversion import (
    decode_functions,
    select_present,
    source_protocol,
    split_system_messages,
)
from my_llm.codec.shared.input import (
    feedback_reasoning,
    input_capability,
    input_content,
    message_feedback,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens
from my_llm.profile.types import Parameter

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

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        """Filter owned Anthropic messages to SDK fields and apply input capability policies."""
        if parameters.get("_translated") or (
            feedback_reasoning(profile, has_tool_calls=False)
            and supports_content("image", profile)
            and supports_content("tool_use", profile)
            and supports_content("message_name", profile)
        ):
            return super().filter_messages(messages, profile, **parameters)
        projected = []
        for index, message in enumerate(dictionary_views(messages)):
            path = f"$.messages[{index}]"
            if message.get("name") is not None:
                content_capability(message["name"], "message_name", profile, path=f"{path}.name")
            content = message.get("content")
            if isinstance(content, list):
                feedback = feedback_reasoning(
                    profile,
                    has_tool_calls=any(
                        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
                    ),
                )
                retained = []
                for offset, block in enumerate(content):
                    if not isinstance(block, dict):
                        retained.append(block)
                        continue
                    kind = block.get("type")
                    block_path = f"{path}.content[{offset}]"
                    if not input_capability(
                        block,
                        "thinking" if kind == "redacted_thinking" else kind,
                        profile,
                        feedback=feedback,
                        path=block_path,
                    ):
                        continue
                    if kind == "tool_result" and isinstance(children := block.get("content"), list):
                        block = {
                            **select_present(
                                block, "type", "tool_use_id", "is_error", "cache_control"
                            ),
                            "content": [
                                child
                                for child_index, child in enumerate(children)
                                if not isinstance(child, dict)
                                or input_capability(
                                    child,
                                    child.get("type"),
                                    profile,
                                    path=f"{block_path}.content[{child_index}]",
                                )
                            ],
                        }
                    retained.append(block)
                message = {"role": message.get("role"), "content": retained}
            projected.append(message)
        return super().filter_messages(projected, profile, **parameters)

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        """Filter native tool fields while preserving the complete open input_schema payload."""
        # The SDK's InputSchema TypedDict names only some JSON Schema keys.
        # Delegate this open leaf so it is captured once rather than filtered
        # and copied before replacing it with the full application schema.
        decode = _plan(self.tool_type, delegated_paths=(("input_schema",),)).decode
        result = []
        for index, original in enumerate(tools):
            filtered = decode(original, f"$.tools[{index}]")
            if filtered is _MISSING:
                continue
            # InputSchema is an open JSON Schema, not just the SDK TypedDict's
            # named type/properties/required fields. Keep its application keys.
            if "input_schema" in original:
                filtered["input_schema"] = capture(original["input_schema"])
            result.append(filtered)
        return result

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        """Validate shared controls and protect Anthropic fields from extra-body replacement."""
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(profile, options, protected=PROTECTED, containers=())

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Validate required history/token limits, system blocks and thinking configuration."""
        if not request["messages"]:
            raise ConfigurationError("a non-system message is required")
        tokens = profile.parameters.get("max_tokens")
        if request.get("max_tokens") is None and not (
            isinstance(tokens, Parameter) and not tokens.supports
        ):
            raise ConfigurationError("Anthropic requires max_tokens in the model or generation")
        validate_max_tokens(request.get("max_tokens"))
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
            and "thinking" in request
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
        """Extract system blocks and assemble a streaming Anthropic messages request."""
        result = request
        system, history = split_system_messages(messages)
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
        """Render middleware history as Anthropic content-block messages.

        Apply capabilities and reasoning-feedback policy, replay native signatures
        only where representable, and record foreign reasoning/name losses.
        Tool calls and results stay as tool_use/tool_result content blocks; request
        assembly later separates system instructions from the conversation.
        """
        result = []
        for index, message, feedback in message_feedback(messages, profile):
            if message.get("name") is not None:
                content_capability(
                    message["name"],
                    "message_name",
                    profile,
                    native_support=False,
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            context_owner = source_protocol(message)
            content = []
            for block in message["content"]:
                if not input_content(block, profile, feedback=feedback):
                    continue
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                    if block is None:
                        continue
                value = decode_block(block, profile=profile)
                if value is not None:
                    content.append(value)
            result.append({"role": message["role"], "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render canonical function declarations as Anthropic tools and report unsupported fields."""
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
        """Render public middleware events as native Anthropic stream events.

        Text/reasoning deltas are already assembled into ordered public events;
        tool events contain complete calls validated at successful source EOF.
        Use source metadata for native identity where available and report
        content that the target protocol cannot express as semantic losses.
        """
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
                block = decode_block(block)
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
                block = decode_block(block)
                if block is None:
                    continue
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
        """Render canonical token counters in the Anthropic usage format."""
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
