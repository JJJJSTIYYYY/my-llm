"""Reusable OpenAI SDK client and request-owned Chat streams."""

from typing import Any

from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_request_options,
)
from my_llm.codec.shared.transport import owned_stream, parameter_types
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile, client_parameters, validate_max_tokens

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


class ChatCompletionsTransport(BaseTransport):
    """Validate native Chat requests and execute them with the official SDK."""

    __slots__ = ()
    usage_modes = ("none", "stream_options")
    tool_type = ChatCompletionToolParam
    message_type = ChatCompletionMessageParam
    message_path = ("messages",)
    tool_path = ("tools",)
    request_schema = parameter_types(AsyncCompletions.create)

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        result = super().filter_messages(messages, profile, **parameters)
        for original, value in zip(messages, result):
            if not profile.get("supports_message_name", True):
                value.pop("name", None)
            if (
                value.get("role") == "assistant"
                and profile.get("replay_reasoning_content_when_tool_calls", False)
                and "reasoning_content" in original
            ):
                if not isinstance(original["reasoning_content"], str):
                    raise MessageValidationError("reasoning_content must be a string")
                value["reasoning_content"] = original["reasoning_content"]
        return result

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        result = super().filter_tools(tools, profile)
        for tool in result:
            function = tool.get("function", {})
            for field in ("description", "parameters"):
                if field in function and function[field] is None:
                    function.pop(field)
        return result

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("extra_body",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        for field in ("max_tokens", "max_completion_tokens"):
            if field in request:
                validate_max_tokens(request[field])
        effort = request.get("reasoning_effort")
        if effort is not None and not isinstance(effort, str):
            raise ConfigurationError("reasoning_effort must be a string")
        body = request.get("extra_body")
        if body is not None and body.get("n") not in (None, 1):
            raise UnsupportedCapabilityError("only one generated candidate is supported")
        return request

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**client_parameters(profile))

    def upstream(
        self, client: AsyncOpenAI, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.chat.completions.create(**request))
