"""Official SDK execution with request-owned stream and history state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, aclosing
from copy import deepcopy
from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from ollama import AsyncClient as AsyncOllama
from openai import AsyncOpenAI, Omit, omit

from my_llm.codec import MyCodec
from my_llm.codec.utils.conversion import dictionaries
from my_llm.codec.utils.validation import validate_history
from my_llm.exceptions import ConfigurationError, MessageValidationError, StreamError
from my_llm.profile import (
    ModelProfile,
    Protocol,
    ProviderProfile,
    ReasoningEffort,
    resolve_profiles,
)
from my_llm.profile.utils.common import PROTOCOLS, merge_dicts
from my_llm.profile.utils.resolution import client_parameters
from my_llm.stream import MessageStream
from my_llm.stream._state import StreamState


class MyLLM:
    """Generate in any standard I/O protocol using a fixed provider/model snapshot.

    ``stdio_protocol`` is independent of the provider's ``protocol``. When unset,
    the first awaited generation guesses it once from the whole history/tools.
    Ambiguous text-only histories prefer openai_chat. Set it explicitly for
    native Anthropic/Ollama or Responses text-only input. Each concurrent request
    owns its stream state and SDK client. Configuration copies are captured here.
    """

    def __init__(
        self,
        *,
        provider: ProviderProfile,
        model: ModelProfile,
        stdio_protocol: Protocol | None = None,
    ) -> None:
        if stdio_protocol is not None and stdio_protocol not in PROTOCOLS:
            raise ConfigurationError(f"unsupported stdio_protocol: {stdio_protocol!r}")
        self.resolved_profile = resolve_profiles(provider, model)
        self.provider = deepcopy(provider)
        self.model = deepcopy(model)
        self.codec = MyCodec()
        self.stdio_protocol = stdio_protocol

    def guess_stdio_protocol(self, messages_or_tools: list[Any]) -> Protocol:
        """Bind the I/O format once. Ambiguous standard messages prefer Chat."""
        if self.stdio_protocol is None:
            guessed = self.codec.match_protocol(messages_or_tools)
            if guessed is None:
                raise MessageValidationError(
                    "cannot recognize a uniform standard protocol; set stdio_protocol explicitly"
                )
            self.stdio_protocol = guessed
        return self.stdio_protocol

    async def generate(
        self,
        *,
        messages: Sequence[Any],
        stream: bool = False,
        tools: Any = omit,
        max_tokens: Any = None,
        reasoning: bool | None = None,
        reasoning_effort: ReasoningEffort = "none",
        extra_headers: Any = None,
        extra_body: Any = None,
        previous_response_id: Any = None,
    ) -> MessageStream | dict | list:
        """Generate a native message, or await and consume a native event stream.

        Provider/model and connection settings belong to construction profiles.
        Messages and tools use one standard protocol per instance. Tools are
        described but never executed. Responses previous_response_id allows
        result messages whose calls are stored on the server. Histories and
        mutable request options are isolated when this method is awaited.
        """
        if not isinstance(stream, bool):
            raise ConfigurationError("stream must be a bool")
        history = dictionaries(messages)
        tool_values = None if tools is None or isinstance(tools, Omit) else dictionaries(tools)
        # Tool declarations can disambiguate otherwise identical text-only histories.
        protocol = self.guess_stdio_protocol([*history, *(tool_values or [])])
        intermediate = self.codec.encode(history, tool_values, input_format=protocol)
        profile = self.resolved_profile
        has_previous = previous_response_id is not None and not isinstance(
            previous_response_id, Omit
        )
        # Only Responses can resolve omitted local calls from a previous server turn.
        validate_history(
            intermediate["messages"],
            allow_external_calls=has_previous and profile.protocol == "openai_responses",
        )
        options = deepcopy(
            dict(
                max_tokens=max_tokens,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                extra_headers=extra_headers,
                extra_body=extra_body,
                previous_response_id=previous_response_id,
            )
        )
        request = self.codec.request(intermediate, profile, **options)
        sdk_kwargs = client_parameters(profile)
        # Ollama accepts headers on the client rather than on each chat request.
        if profile.protocol == "ollama":
            sdk_kwargs["headers"] = deepcopy(
                merge_dicts(profile.get("extra_headers", {}), extra_headers)
            )
        state = StreamState(profile.model)
        result = MessageStream(self._stream(request, sdk_kwargs, state, protocol), state)
        if stream:
            return result
        # Collected and streaming results share one execution and cleanup path.
        async with aclosing(result):
            async for _ in result:
                pass
        return result.result

    def _render(self, events: list[dict], state: StreamState, protocol: Protocol) -> list[dict]:
        output = self.codec.decode(
            chunks=events,
            target_format=protocol,
            sequence_number=state.sequence_number,
            **state.metadata,
        )["chunks"]
        state.sequence_number += len(output)
        return output

    async def _stream(
        self, request: dict, sdk_kwargs: dict, state: StreamState, stdio_protocol: Protocol
    ) -> AsyncIterator[dict]:
        profile = self.resolved_profile
        protocol = profile.protocol
        client = None
        upstream = None
        responses = AsyncExitStack()
        try:
            if protocol == "anthropic":
                client = AsyncAnthropic(**sdk_kwargs)
                upstream = await client.messages.create(**request)
            elif protocol == "gemini":
                options = {k: v for k, v in sdk_kwargs.items() if k == "base_url"}

                async def track_response(response: Any) -> None:
                    responses.push_async_callback(response.aclose)

                # GenAI wraps nested async generators; closing the outer one
                # may leave an HTTPX body suspended until garbage collection.
                # Track responses through its public HTTP options so early exit
                # closes every body (including retries) within this request.
                options["async_client_args"] = {"event_hooks": {"response": [track_response]}}
                # GenAI uses milliseconds; its attempt count includes the initial request.
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
                upstream = await client.aio.models.generate_content_stream(**request)
            elif protocol == "ollama":
                options = {k: v for k, v in sdk_kwargs.items() if k == "timeout"}
                headers = sdk_kwargs.get("headers", {})
                if "api_key" in sdk_kwargs:
                    headers = {"Authorization": f"Bearer {sdk_kwargs['api_key']}", **headers}
                if headers:
                    options["headers"] = headers
                client = AsyncOllama(host=sdk_kwargs.get("base_url"), **options)
                upstream = await client.chat(**request)
            else:
                client = AsyncOpenAI(**sdk_kwargs)
                endpoint = (
                    client.responses if protocol == "openai_responses" else client.chat.completions
                )
                upstream = await endpoint.create(**request)
            async for event in upstream:
                encoded = self.codec.encode(chunks=[event], input_format=protocol, profile=profile)[
                    "chunks"
                ]
                for output in self._render(state.feed(encoded), state, stdio_protocol):
                    yield output
            final = state.finish()
            converted = self.codec.decode(messages=state.messages(), target_format=stdio_protocol)[
                "messages"
            ]
            state.result = converted if stdio_protocol == "openai_responses" else converted[0]
            for output in self._render(final, state, stdio_protocol):
                yield output
        except StreamError as exc:
            converted = self.codec.decode(
                messages=state.messages(partial=True), target_format=stdio_protocol
            )["messages"]
            exc.partial_message = (
                converted if stdio_protocol == "openai_responses" else converted[0]
            )
            raise
        finally:
            try:
                if upstream is not None:
                    if protocol in ("gemini", "ollama"):
                        await upstream.aclose()
                    else:
                        await upstream.close()
            # A failed stream close must not prevent closing the owned client.
            finally:
                if client is not None:
                    # GenAI exposes separate asynchronous and synchronous clients.
                    if protocol == "gemini":
                        try:
                            await responses.aclose()
                        finally:
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
    messages: Sequence[Any],
    stdio_protocol: Protocol | None = None,
    **parameters: Any,
) -> MessageStream | dict | list:
    """One-shot generation with the same standard I/O as MyLLM.generate()."""
    selected = ModelProfile(model=model) if isinstance(model, str) else model
    return await MyLLM(provider=provider, model=selected, stdio_protocol=stdio_protocol).generate(
        messages=messages, **parameters
    )
