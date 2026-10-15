"""Request execution using AsyncOpenAI streaming for both public modes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from openai import AsyncOpenAI, Omit, omit

from .codec import ChatCompletionsCodec, ResponsesCodec
from .exceptions import ConfigurationError, StreamError, UnsupportedProviderError
from .profiles import (
    ModelProfile,
    Protocol,
    ProviderProfile,
    ReasoningEffort,
    client_parameters,
    resolve_profiles,
)
from .types import Message, MessageChunk


class MessageStream(AsyncIterator[MessageChunk]):
    """Closable result iterator; use async with for deterministic early cleanup."""

    def __init__(self, iterator: AsyncIterator[MessageChunk]) -> None:
        self._iterator = iterator
        self._closed = False

    def __aiter__(self) -> MessageStream:
        return self

    async def __anext__(self) -> MessageChunk:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._iterator)
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._iterator.aclose()

    async def __aenter__(self) -> MessageStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


class Invoke:
    """Codec/execution boundary, configured exclusively through user profiles.

    A supplied AsyncOpenAI is borrowed and never closed. Without one, a fresh
    SDK client is created lazily for each stream and always closed afterwards.
    """

    def __init__(
        self, *, provider: ProviderProfile, model: ModelProfile, client: AsyncOpenAI | None = None
    ) -> None:
        self.profile = resolve_profiles(provider, model)
        self.client = client
        self.codec = (
            ResponsesCodec(self.profile)
            if self.profile.api_type == "responses"
            else ChatCompletionsCodec(self.profile)
        )

    async def create(
        self,
        *,
        messages: list[Message],
        protocol: Protocol = "openai",
        base_url: str | Omit | None = None,
        api_key: str | Omit | None = None,
        timeout: float | Omit | None = None,
        max_retries: int | Omit | None = None,
        extra_headers: dict[str, Any] | Omit | None = None,
        extra_body: dict[str, Any] | Omit | None = None,
        max_tokens: int | Omit | None = None,
        reasoning: bool | None = None,
        reasoning_effort: ReasoningEffort = "none",
        stream: bool | None = None,
        tools: list[dict[str, Any]] | Omit | None = omit,
        previous_response_id: str | Omit | None = None,
    ) -> MessageStream | Message:
        if protocol != "openai":
            raise UnsupportedProviderError(f"protocol {protocol!r} is reserved")
        if stream is not None and not isinstance(stream, bool):
            raise ConfigurationError("stream must be True, False or None")
        overrides = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if self.client is not None and any(value is not None for value in overrides.values()):
            raise ConfigurationError(
                "client initialization overrides cannot be used with an injected SDK client"
            )
        sdk_kwargs = client_parameters(self.profile, **overrides)
        request = self.codec.encode(
            messages,
            tools=tools,
            extra_headers=extra_headers,
            extra_body=extra_body,
            max_tokens=max_tokens,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            previous_response_id=previous_response_id,
        )
        result = MessageStream(self._stream(request, sdk_kwargs))
        if stream:
            return result
        combined = MessageChunk()
        async with aclosing(result):
            async for chunk in result:
                combined += chunk
        return combined.to_message(require_finished=True)

    async def _stream(
        self, request: dict[str, Any], sdk_kwargs: dict[str, Any]
    ) -> AsyncIterator[MessageChunk]:
        client = self.client
        owned = client is None
        upstream = None
        combined = MessageChunk()
        decoder = self.codec.decoder()
        try:
            if client is None:
                client = AsyncOpenAI(**sdk_kwargs)
            endpoint = (
                client.responses
                if self.profile.api_type == "responses"
                else client.chat.completions
            )
            upstream = await endpoint.create(**request)
            async for event in upstream:
                for chunk in decoder.feed(event):
                    combined += chunk
                    yield chunk
            final = decoder.finish()
            combined += final
            yield final
        except StreamError as exc:
            exc.partial_message = {**combined.to_message(), **(exc.partial_message or {})}
            raise
        finally:
            try:
                if upstream is not None:
                    await upstream.close()
            finally:
                if owned and client is not None:
                    await client.close()


async def create(
    *,
    provider: ProviderProfile,
    model: ModelProfile | str,
    messages: list[Message],
    protocol: Protocol = "openai",
    base_url: str | Omit | None = None,
    api_key: str | Omit | None = None,
    timeout: float | Omit | None = None,
    max_retries: int | Omit | None = None,
    extra_headers: dict[str, Any] | Omit | None = None,
    extra_body: dict[str, Any] | Omit | None = None,
    max_tokens: int | Omit | None = None,
    reasoning: bool | None = None,
    reasoning_effort: ReasoningEffort = "none",
    stream: bool | None = None,
    tools: list[dict[str, Any]] | Omit | None = omit,
    previous_response_id: str | Omit | None = None,
    client: AsyncOpenAI | None = None,
) -> MessageStream | Message:
    """Invoke a configured model; stream=None/False collects the same stream.

    tools describes caller-owned local functions; my-llm returns function calls
    and never executes tools or enables provider-native cloud services.
    """
    if isinstance(model, str):
        model = ModelProfile(model=model)
    invoke = Invoke(provider=provider, model=model, client=client)
    return await invoke.create(
        messages=messages,
        protocol=protocol,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        extra_headers=extra_headers,
        extra_body=extra_body,
        max_tokens=max_tokens,
        reasoning=reasoning,
        reasoning_effort=reasoning_effort,
        stream=stream,
        tools=tools,
        previous_response_id=previous_response_id,
    )
