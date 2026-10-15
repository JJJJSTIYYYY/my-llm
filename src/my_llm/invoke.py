"""Request execution using official asynchronous SDK streams for both public modes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from copy import deepcopy
from dataclasses import replace
from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from ollama import AsyncClient as AsyncOllama
from openai import AsyncOpenAI, Omit, omit

from my_llm.codec import (
    AnthropicCodec,
    ChatCompletionsCodec,
    GeminiCodec,
    OllamaCodec,
    ResponsesCodec,
)
from my_llm.profile import ModelProfile, Protocol, ProviderProfile, ReasoningEffort
from my_llm.profile.utils.common import merge_dicts
from my_llm.profile.utils.resolution import client_parameters, resolve_profiles
from my_llm.types import Message, MessageChunk, MessageInput
from my_llm.types.exceptions import ConfigurationError, StreamError, UnsupportedProviderError


class _MessageAccumulator:
    """Buffer text fragments; join and snapshot only when producing a message."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._metadata = MessageChunk()

    def add(self, chunk: MessageChunk) -> None:
        if chunk.content:
            self._content.append(chunk.content)
        if chunk.reasoning_content:
            self._reasoning.append(chunk.reasoning_content)
        if (
            chunk.tool_calls
            or chunk.is_finished
            or chunk.usage is not None
            or chunk.response_id is not None
            or chunk.response_output
            or chunk.response_status is not None
        ):
            self._metadata += replace(chunk, content="", reasoning_content="")

    def to_message(self, *, require_finished: bool = False) -> Message:
        return replace(
            self._metadata,
            content="".join(self._content),
            reasoning_content="".join(self._reasoning),
        ).to_message(require_finished=require_finished)


class MessageStream(AsyncIterator[MessageChunk]):
    """An asynchronous stream of assistant-message fragments.

    Returned by ``await generate(..., stream=True)`` and
    ``await llm.generate(..., stream=True)``. Use ``async with stream`` when
    consuming it, especially if iteration may stop early. Text and reasoning
    arrive incrementally; tool calls are returned only when complete.

    Args:
        iterator: A closable asynchronous iterator yielding MessageChunk values.
    """

    def __init__(self, iterator: AsyncIterator[MessageChunk]) -> None:
        """Wrap a chunk iterator that provides an asynchronous ``aclose()``."""
        self._iterator = iterator
        self._closed = False

    def __aiter__(self) -> MessageStream:
        """Return this stream for use in ``async for``."""
        return self

    async def __anext__(self) -> MessageChunk:
        """Read the next chunk, closing the stream on completion or failure."""
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._iterator)
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Release this request's resources; repeated calls are safe.

        Closing an unconsumed stream does not start a request. A caller-supplied
        SDK client remains open and must be closed by its owner.
        """
        if not self._closed:
            self._closed = True
            await self._iterator.aclose()

    async def __aenter__(self) -> MessageStream:
        """Enter a context that closes the stream automatically on exit."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close the stream after normal completion, early exit or an error."""
        await self.aclose()


class MyLLM:
    """Generate assistant messages using a configured provider and model.

    Args:
        provider: Connection, API selection and provider-wide defaults.
        model: Explicit ModelProfile to use for every generation. The provider's
            model catalogue is not searched automatically.
        client: Optional caller-owned official SDK client. Configure and close
            this client yourself; per-call connection overrides are unavailable
            when it is supplied.

    Configuration is captured at construction. To change it, create a new
    instance with updated profiles. Concurrent calls have independent histories
    and result streams. Without an injected client, each request's client is
    created and closed automatically.

    Raises:
        ConfigurationError: The supplied profiles are invalid.
        UnsupportedProviderError: The provider uses an unsupported protocol or
            does not support upstream streaming.

    Example:
        llm = MyLLM(provider=ProviderProfile(), model=ModelProfile(model="your-model"))
        message = await llm.generate(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        *,
        provider: ProviderProfile,
        model: ModelProfile,
        client: AsyncOpenAI | AsyncAnthropic | genai.Client | AsyncOllama | None = None,
    ) -> None:
        """Capture provider/model settings and optionally borrow an SDK client."""
        self.profile = resolve_profiles(provider, model)
        self.client = client
        native = {"anthropic": AnthropicCodec, "gemini": GeminiCodec, "ollama": OllamaCodec}
        self.codec = (
            native[self.profile.protocol](self.profile)
            if self.profile.protocol in native
            else (
                ResponsesCodec(self.profile)
                if self.profile.api_type == "responses"
                else ChatCompletionsCodec(self.profile)
            )
        )

    async def generate(
        self,
        *,
        messages: Sequence[MessageInput],
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
        """Generate one assistant message or return its asynchronous stream.

        Args:
            messages: Non-empty sequence of Chat Completions messages, Responses
                items, or a mixture. Roles are system/user/assistant/tool.
            protocol: Optional consistency check against ProviderProfile.protocol.
                The default "openai" follows the configured provider protocol.
            base_url: Connection URL override; None uses the provider setting.
            api_key: Credential override; None uses the provider setting.
            timeout: Positive timeout in seconds; None uses the provider setting.
            max_retries: Non-negative SDK retry count; None inherits the provider.
            extra_headers: Request headers merged over profile defaults.
            extra_body: Extra JSON fields merged over profile defaults. Generated
                reasoning parameters take precedence over matching fields.
            max_tokens: Positive output-token limit; None inherits the model,
                while omit removes the explicit limit.
            reasoning: Optional switch. True selects medium when effort is none;
                False conflicts with a non-none effort.
            reasoning_effort: One of none/low/medium/high/xhigh/max. Model
                capability flags and effort mappings determine availability.
            stream: True returns MessageStream; False or None returns Message.
            tools: Local function descriptions in either supported API format.
                None or omit leaves tools unspecified. Tools are not executed.
            previous_response_id: Responses-only conversation reference. Send
                only the new messages or function results when using it.

        Returns:
            A unified assistant-message dictionary, or a MessageStream when
            stream=True. Always await this method before iterating the stream.
            Equivalent finish reasons use stop, length, tool_calls or
            content_filter in both modes; other reasons are preserved.

        Raises:
            ConfigurationError: Parameters conflict or have invalid values.
            MessageValidationError: Messages/tools cannot be converted safely.
            UnsupportedCapabilityError: The model lacks a requested capability.
            StreamError: The response failed or ended incompletely. For streaming
                calls this may be raised while iterating. SDK errors propagate
                with their original types.

        Connection overrides cannot be used with an injected SDK client.
        Input history is isolated when this method is awaited. Use the returned
        stream as an async context manager to release resources after early exit.
        """
        if protocol not in ("openai", self.profile.protocol):
            raise UnsupportedProviderError("protocol must match ProviderProfile.protocol")
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
        if self.profile.protocol == "ollama" and self.client is None:
            headers = merge_dicts(self.profile.values["extra_headers"], extra_headers)
            if headers:
                sdk_kwargs["headers"] = deepcopy(headers)
        request = self.codec.encode(
            deepcopy(messages),
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
        combined = _MessageAccumulator()
        async with aclosing(result):
            async for chunk in result:
                combined.add(chunk)
        return combined.to_message(require_finished=True)

    async def _stream(
        self, request: dict[str, Any], sdk_kwargs: dict[str, Any]
    ) -> AsyncIterator[MessageChunk]:
        client = self.client
        owned = client is None
        upstream = None
        combined = _MessageAccumulator()
        decoder = self.codec.decoder()
        protocol = self.profile.protocol
        try:
            if client is None:
                if protocol == "anthropic":
                    client = AsyncAnthropic(**sdk_kwargs)
                elif protocol == "gemini":
                    options = {k: v for k, v in sdk_kwargs.items() if k == "base_url"}
                    if "timeout" in sdk_kwargs:
                        options["timeout"] = max(1, round(sdk_kwargs["timeout"] * 1000))
                    if "max_retries" in sdk_kwargs:
                        options["retry_options"] = genai_types.HttpRetryOptions(
                            attempts=sdk_kwargs["max_retries"] + 1
                        )
                    client = genai.Client(
                        **({"api_key": sdk_kwargs["api_key"]} if "api_key" in sdk_kwargs else {}),
                        http_options=genai_types.HttpOptions(**options),
                    )
                elif protocol == "ollama":
                    options = {k: v for k, v in sdk_kwargs.items() if k == "timeout"}
                    headers = sdk_kwargs.get("headers", {})
                    if "api_key" in sdk_kwargs:
                        headers = {"Authorization": f"Bearer {sdk_kwargs['api_key']}", **headers}
                    if headers:
                        options["headers"] = headers
                    client = AsyncOllama(host=sdk_kwargs.get("base_url"), **options)
                else:
                    client = AsyncOpenAI(**sdk_kwargs)
            if protocol == "anthropic":
                upstream = await client.messages.create(**request)
            elif protocol == "gemini":
                upstream = await client.aio.models.generate_content_stream(**request)
            elif protocol == "ollama":
                upstream = await client.chat(**request)
            else:
                endpoint = (
                    client.responses
                    if self.profile.api_type == "responses"
                    else client.chat.completions
                )
                upstream = await endpoint.create(**request)
            async for event in upstream:
                for chunk in decoder.feed(event):
                    combined.add(chunk)
                    yield chunk
            final = decoder.finish()
            combined.add(final)
            yield final
        except StreamError as exc:
            exc.partial_message = {**combined.to_message(), **(exc.partial_message or {})}
            raise
        finally:
            try:
                if upstream is not None:
                    if protocol in ("gemini", "ollama"):
                        await upstream.aclose()
                    else:
                        await upstream.close()
            finally:
                if owned and client is not None:
                    if protocol == "gemini":
                        try:
                            await client.aio.aclose()
                        finally:
                            client.close()
                    else:
                        await client.close()


async def generate(
    *,
    provider: ProviderProfile,
    model: ModelProfile | str,
    messages: Sequence[MessageInput],
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
    client: AsyncOpenAI | AsyncAnthropic | genai.Client | AsyncOllama | None = None,
) -> MessageStream | Message:
    """Generate an assistant message without explicitly constructing MyLLM.

    Args:
        provider: ProviderProfile selecting the API and provider defaults.
        model: A ModelProfile, or a model ID string using default model settings.
            A string does not look up the provider's model catalogue.
        messages: Non-empty sequence of Chat Completions messages, Responses
            items, or a mixture. Native SDK items must first be made dictionaries.
        protocol: Optional consistency check; the default "openai" follows the
            provider protocol. Other values must match ProviderProfile.protocol.
        base_url: Override the provider URL; None inherits, omit leaves it unset.
        api_key: Override credentials; None inherits, omit leaves them unset.
        timeout: Positive timeout in seconds; None inherits the provider value.
        max_retries: Non-negative SDK retry count; None inherits the provider.
        extra_headers: Request headers overriding model/provider dictionary keys.
        extra_body: Extra request fields overriding model/provider keys. Generated
            reasoning parameters have priority over matching fields.
        max_tokens: Positive output-token limit; None inherits, omit leaves unset.
        reasoning: Optional boolean switch; True with effort=none uses medium.
        reasoning_effort: none/low/medium/high/xhigh/max, subject to model support.
        stream: True returns MessageStream; False or None returns Message.
        tools: Local function tools in Responses or Chat Completions format;
            None/omit leaves them unspecified. The caller executes tool calls.
        previous_response_id: Responses-only reference for continuing with new
            messages or tool results rather than resending the old history.
        client: Optional caller-owned AsyncOpenAI, AsyncAnthropic, genai.Client
            or ollama.AsyncClient; connection overrides must be
            omitted and the caller is responsible for closing this client.

    Returns:
        A unified assistant Message, or a MessageStream for stream=True. The
        streaming form is also awaited: ``stream = await generate(...)``.
        Equivalent finish reasons use stop, length, tool_calls or content_filter
        in both modes; other reasons are preserved.

    Raises:
        ConfigurationError: Profiles or request parameters are invalid.
        MessageValidationError: Messages/tools are invalid or cannot be converted.
        UnsupportedProviderError: The requested protocol/provider is unsupported.
        UnsupportedCapabilityError: The model lacks a requested capability.
        StreamError: The response fails or ends incompletely. With stream=True,
            failures may occur during iteration. SDK errors retain their types.

    Example:
        message = await generate(
            provider=ProviderProfile(), model="your-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    if isinstance(model, str):
        model = ModelProfile(model=model)
    llm = MyLLM(provider=provider, model=model, client=client)
    return await llm.generate(
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
