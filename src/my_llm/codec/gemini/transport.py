"""Reusable Gemini SDK client with request-local response-body cleanup."""

from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from google import genai
from google.genai import types
from google.genai.models import AsyncModels

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_request_options,
)
from my_llm.codec.shared.transport import parameter_types
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile, client_parameters, validate_max_tokens

_responses: ContextVar[AsyncExitStack | None] = ContextVar("gemini_responses", default=None)


async def _track_response(response: Any) -> None:
    resources = _responses.get()
    if resources is not None:
        resources.push_async_callback(response.aclose)


@contextmanager
def _response_scope(resources: AsyncExitStack) -> Iterator[None]:
    token = _responses.set(resources)
    try:
        yield
    finally:
        _responses.reset(token)


PROTECTED = {
    "model",
    "stream",
    "messages",
    "contents",
    "input",
    "tools",
    "system_instruction",
    "previous_response_id",
    "config",
    "automatic_function_calling",
    "http_options",
}


class GeminiTransport(BaseTransport):
    """Official SDK lifecycle and native request execution for this protocol."""

    __slots__ = ()
    tool_type = types.Tool
    message_type = types.Content
    message_path = ("contents",)
    tool_path = ("config", "tools")
    request_schema = {
        **parameter_types(AsyncModels.generate_content_stream),
        "contents": list[types.Content],
        "config": types.GenerateContentConfig,
    }

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("config",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        if not request["contents"]:
            raise ConfigurationError("a non-system message is required")
        config = request.get("config")
        if config is None:
            return request
        validate_max_tokens(config.get("max_output_tokens"))
        if config.get("candidate_count") not in (None, 1):
            raise UnsupportedCapabilityError("only one generated candidate is supported")
        thinking = types.GenerateContentConfig(**config).thinking_config
        if thinking is not None and thinking.thinking_budget == 0 and thinking.include_thoughts:
            raise ConfigurationError("reasoning conflicts with a zero Gemini thinking budget")
        return request

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> genai.Client:
        values = client_parameters(profile)
        options = {key: value for key, value in values.items() if key == "base_url"}
        options["async_client_args"] = {"event_hooks": {"response": [_track_response]}}
        if "timeout" in values:
            options["timeout"] = max(1, round(values["timeout"] * 1000))
        if "max_retries" in values:
            options["retry_options"] = types.HttpRetryOptions(attempts=values["max_retries"] + 1)
        return genai.Client(
            **({"api_key": values["api_key"]} if "api_key" in values else {}),
            http_options=types.HttpOptions(**options),
        )

    async def close_client(self, client: genai.Client) -> None:
        try:
            await client.aio.aclose()
        finally:
            client.close()

    @asynccontextmanager
    async def upstream(
        self, client: genai.Client, request: dict, profile: ResolvedProfile, **parameters: Any
    ) -> AsyncIterator[Any]:
        async with AsyncExitStack() as resources:
            with _response_scope(resources):
                upstream = await client.aio.models.generate_content_stream(**request)
            resources.push_async_callback(upstream.aclose)

            async def events():
                while True:
                    # Reset before yielding: consumers may resume/close a stream
                    # from a different task. Only SDK reads use this request's scope.
                    with _response_scope(resources):
                        try:
                            event = await anext(upstream)
                        except StopAsyncIteration:
                            return
                    yield event

            iterator = events()
            resources.push_async_callback(iterator.aclose)
            yield iterator
