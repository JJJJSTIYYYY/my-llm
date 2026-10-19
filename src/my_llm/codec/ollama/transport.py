"""Ollama host, authentication and client-header semantics."""

from inspect import signature
from typing import Any

from ollama import AsyncClient, Image, Message, Options, Tool
from ollama import AsyncClient as AsyncOllama
from pydantic import ValidationError

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.conversion import plain
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_max_tokens_path,
    validate_native_options,
)
from my_llm.codec.shared.transport import owned_stream, supported
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, client_parameters


def client_headers(profile: ResolvedProfile, **parameters: Any) -> dict:
    return merge_dicts(profile.get("extra_headers", {}), parameters.get("extra_headers"))


PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}
REQUEST_FIELDS = set(signature(AsyncClient.chat).parameters) - PROTECTED - {"self"}


def validate_schema(parameters: dict) -> dict:
    """Do not let the official SDK silently discard JSON Schema constraints."""
    try:
        parsed = Tool(function={"name": "schema_check", "parameters": parameters}).model_dump(
            exclude_none=True
        )["function"]["parameters"]
    except (ValidationError, KeyError) as exc:
        raise UnsupportedCapabilityError(
            "installed Ollama SDK cannot represent the tool parameter schema"
        ) from exc

    def preserves(expected, actual):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and preserves(value, actual[key]) for key, value in expected.items()
            )
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(expected) == len(actual)
                and all(preserves(left, right) for left, right in zip(expected, actual))
            )
        return expected == actual

    if not preserves(parameters, parsed):
        raise UnsupportedCapabilityError(
            "installed Ollama SDK would discard tool parameter schema constraints"
        )
    return parameters


class OllamaTransport(BaseTransport):
    """Official SDK lifecycle and native request execution for this protocol."""

    __slots__ = ()
    tool_type = Tool
    message_type = Message
    message_path = ("messages",)
    tool_path = ("tools",)

    def _validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        if schema is Message and isinstance(values, list):
            # The public chat API wraps wire-format base64 strings in Image
            # before validating Message. Mirror that SDK boundary without
            # changing the dictionaries used for conversion or serialization.
            try:
                values = [
                    {
                        **value,
                        "images": [
                            image if isinstance(image, Image) else Image(value=image)
                            for image in value["images"]
                        ],
                    }
                    if isinstance(value, dict) and value.get("images")
                    else value
                    for value in values
                ]
            except (ValidationError, TypeError, ValueError) as exc:
                raise MessageValidationError(
                    "images do not conform to the native protocol"
                ) from exc
        super()._validate_values(values, schema, filtered=filtered)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        options = validate_native_options(
            options, request_fields=REQUEST_FIELDS, protected=PROTECTED
        )
        tokens, body, effort = options.tokens, options.body, options.effort
        validate_max_tokens_path(profile, tokens, protected=PROTECTED, request=body)
        if (
            effort
            and not profile.get("supports_reasoning_effort", False)
            and body.get("think") is False
        ):
            raise ConfigurationError("reasoning conflicts with think=False")
        return options

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        thinking = request.get("think")
        if thinking is not None and not isinstance(thinking, (str, bool)):
            raise ConfigurationError("Ollama think must be a boolean or model-defined string")
        if request.get("options") is not None:
            try:
                options = supported(plain(request["options"]), Options)
                request["options"] = Options(**options).model_dump(exclude_none=True)
            except (ValidationError, TypeError, ValueError) as exc:
                raise ConfigurationError("invalid Ollama options") from exc
        return request

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        for tool in tools:
            if "parameters" in tool.get("function", {}):
                validate_schema(tool["function"]["parameters"])
        return super().filter_tools(tools, profile)

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        result = super().filter_messages(messages, profile, **parameters)
        for value in result:
            if not profile.get("supports_message_name", True):
                value.pop("name", None)
            if not profile.get("replay_reasoning_content_when_tool_calls", False):
                value.pop("thinking", None)
        return result

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncOllama:
        values = client_parameters(profile)
        options = {key: value for key, value in values.items() if key == "timeout"}
        headers = client_headers(profile, **parameters)
        if "api_key" in values:
            headers = {"Authorization": f"Bearer {values['api_key']}", **headers}
        if headers:
            options["headers"] = headers
        return AsyncOllama(host=values.get("base_url"), **options)

    def client_key(self, profile: ResolvedProfile, **parameters: Any) -> tuple:
        return tuple(sorted(client_headers(profile, **parameters).items()))

    def validate_profile(self, profile: ResolvedProfile) -> None:
        super().validate_profile(profile)
        retries = profile.get("max_retries")
        if isinstance(retries, int) and not isinstance(retries, bool) and retries > 0:
            raise UnsupportedCapabilityError("Ollama SDK does not support a positive max_retries")

    def upstream(
        self, client: AsyncOllama, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.chat(**request), close_stream="aclose")
