"""Reusable OpenAI SDK client and request-owned Responses streams."""

from dataclasses import replace
from typing import Any

from openai import AsyncOpenAI, Omit
from openai.types.responses import ResponseInputItemParam, ToolParam
from openai.types.shared_params import Reasoning

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.options import (
    PreparedOptions,
    check_protected,
    merge_bindings,
    validate_bindings,
    validate_max_tokens_path,
)
from my_llm.codec.shared.transport import owned_stream, supported
from my_llm.exceptions import ConfigurationError
from my_llm.profile.resolution import ResolvedProfile, client_parameters, validate_max_tokens

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


class ResponsesTransport(BaseTransport):
    """Validate native Responses requests and execute them with the official SDK."""

    __slots__ = ()
    tool_type = ToolParam
    message_type = ResponseInputItemParam
    message_path = ("input",)
    tool_path = ("tools",)
    supports_previous_response_id = True

    def allows_external_calls(self, previous_response_id) -> bool:
        return previous_response_id is not None and not isinstance(previous_response_id, Omit)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        tokens, body = options.tokens, options.body
        check_protected(body, PROTECTED)
        bindings = validate_bindings(
            options, protected=PROTECTED, allowed={"reasoning", "extra_body"}
        )
        for binding in bindings:
            if "reasoning" in binding and not isinstance(binding["reasoning"], dict):
                raise ConfigurationError("reasoning must be a dictionary")
        body = merge_bindings(body, *(binding.get("extra_body") for binding in options.bindings))
        validate_max_tokens_path(
            profile,
            tokens,
            protected=PROTECTED,
            containers=("extra_body",),
            request={"extra_body": body},
        )
        return replace(options, body=body)

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        for field in ("max_output_tokens",):
            if field in request:
                validate_max_tokens(request[field])
        reasoning = request.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise ConfigurationError("reasoning must be a dictionary")
            reasoning = supported(reasoning, Reasoning)
            for field in ("effort", "summary", "generate_summary"):
                if reasoning.get(field) is not None and not isinstance(reasoning[field], str):
                    raise ConfigurationError(f"reasoning.{field} must be a string")
            request["reasoning"] = reasoning
        return request

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**client_parameters(profile))

    def upstream(
        self, client: AsyncOpenAI, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.responses.create(**request))
