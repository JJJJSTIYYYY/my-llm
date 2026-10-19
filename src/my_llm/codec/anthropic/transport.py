"""Reusable Anthropic SDK client and request-owned Messages streams."""

from copy import deepcopy
from inspect import signature
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.resources.messages import AsyncMessages
from anthropic.types import (
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
)
from pydantic import ValidationError

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_max_tokens_path,
    validate_native_options,
)
from my_llm.codec.shared.transport import adapter, owned_stream, supported
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile, client_parameters, validate_max_tokens

PROTECTED = {"model", "stream", "messages", "input", "tools", "system", "previous_response_id"}
REQUEST_FIELDS = (
    set(signature(AsyncMessages.create).parameters)
    - PROTECTED
    - {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
)


class AnthropicTransport(BaseTransport):
    """Validate native Messages requests and execute them with the official SDK."""

    __slots__ = ()
    tool_type = ToolParam
    message_type = MessageParam
    message_path = ("messages",)
    tool_path = ("tools",)

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        result = super().filter_tools(tools, profile)
        for original, filtered in zip(tools, result):
            # InputSchema is an open JSON Schema, not just the SDK TypedDict's
            # named type/properties/required fields. Keep its application keys.
            if "input_schema" in original:
                filtered["input_schema"] = deepcopy(original["input_schema"])
        return result

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        options = validate_native_options(
            options, request_fields=REQUEST_FIELDS, protected=PROTECTED
        )
        tokens, body, effort = options.tokens, options.body, options.effort
        if tokens is None:
            raise ConfigurationError("Anthropic requires max_tokens in the model or generation")
        validate_max_tokens_path(profile, tokens, protected=PROTECTED, request=body)
        if effort:
            thinking = body.get("thinking", {"type": "enabled", "budget_tokens": 1024})
            if not isinstance(thinking, dict) or thinking.get("type") not in (
                "enabled",
                "adaptive",
            ):
                raise ConfigurationError(
                    "reasoning requires enabled or adaptive Anthropic thinking"
                )
            if thinking.get("type") == "enabled":
                budget = thinking.get("budget_tokens", 1024)
                if (
                    not isinstance(budget, int)
                    or isinstance(budget, bool)
                    or budget < 1024
                    or budget >= tokens
                ):
                    raise ConfigurationError(
                        "Anthropic thinking budget must be >=1024 and < max_tokens"
                    )
            if profile.get("supports_reasoning_effort", False) and effort not in (
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            ):
                raise UnsupportedCapabilityError(
                    "map reasoning effort to an Anthropic-supported level"
                )
        return options

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        if not request["messages"]:
            raise ConfigurationError("a non-system message is required")
        if "max_tokens" in request:
            validate_max_tokens(request["max_tokens"])
        if "system" in request and not isinstance(request["system"], str):
            request["system"] = supported(request["system"], list[TextBlockParam])
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
            request[field] = supported(request[field], schema)
            try:
                adapter(schema).validate_python(request[field], strict=True)
            except (ValidationError, TypeError, ValueError) as exc:
                raise ConfigurationError(f"invalid Anthropic {field}") from exc
        thinking = request.get("thinking", {})
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

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncAnthropic:
        return AsyncAnthropic(**client_parameters(profile))

    def upstream(
        self, client: AsyncAnthropic, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.messages.create(**request))
