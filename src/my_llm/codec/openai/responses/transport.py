"""Reusable OpenAI SDK client and request-owned Responses streams."""

from typing import Any

from openai import AsyncOpenAI
from openai.resources.responses import AsyncResponses
from openai.types.responses import ResponseInputItemParam, ToolParam

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_request_options,
)
from my_llm.codec.shared.transport import owned_stream, parameter_types
from my_llm.exceptions import ConfigurationError
from my_llm.profile.resolution import ResolvedProfile, client_parameters, validate_max_tokens
from my_llm.profile.types import Omit

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


class ResponsesTransport(BaseTransport):
    """Validate native Responses requests and execute them with the official SDK."""

    __slots__ = ()
    tool_type = ToolParam
    message_type = ResponseInputItemParam
    message_path = ("input",)
    tool_path = ("tools",)
    supports_previous_response_id = True
    request_schema = parameter_types(AsyncResponses.create)

    def allows_external_calls(self, previous_response_id) -> bool:
        return previous_response_id is not None and not isinstance(previous_response_id, Omit)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("extra_body",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        if "max_output_tokens" in request:
            validate_max_tokens(request["max_output_tokens"])
        reasoning = request.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise ConfigurationError("reasoning must be a dictionary")
            for field in ("effort", "summary", "generate_summary"):
                if reasoning.get(field) is not None and not isinstance(reasoning[field], str):
                    raise ConfigurationError(f"reasoning.{field} must be a string")
        return request

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**client_parameters(profile))

    def upstream(
        self, client: AsyncOpenAI, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.responses.create(**request))
