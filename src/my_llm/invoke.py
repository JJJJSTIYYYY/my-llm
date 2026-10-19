"""Official SDK execution with request-owned stream and history state."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Hashable, Sequence
from contextlib import AsyncExitStack, aclosing
from copy import deepcopy
from typing import Any

from openai import Omit, omit

from my_llm.codec.converter import MyCodec
from my_llm.codec.shared.conversion import dictionary_views
from my_llm.exceptions import ConfigurationError, MessageValidationError, StreamError
from my_llm.profile.configuration import Protocol, ReasoningEffort
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.resolution import resolve_profiles
from my_llm.stream._state import StreamState
from my_llm.stream.message_stream import MessageStream


class MyLLM:
    """Generate in any standard I/O protocol using a fixed provider/model snapshot.

    ``stdio_protocol`` is independent of the provider's ``protocol``. When unset,
    the first awaited generation binds it only if the whole history/tools match
    a unique protocol. Ambiguous or unrecognized input requires an explicit
    stdio_protocol. Each concurrent request owns its stream state and borrows an
    instance-owned SDK client. Use
    ``async with MyLLM(...)`` or ``await llm.aclose()`` to release clients.
    Configuration copies are captured here; internal mutation is not supported.
    """

    def __init__(
        self,
        *,
        provider: ProviderProfile,
        model: ModelProfile,
        stdio_protocol: Protocol | None = None,
    ) -> None:
        self.codec = MyCodec()
        if stdio_protocol is not None and (
            not isinstance(stdio_protocol, str) or stdio_protocol not in self.codec.codecs
        ):
            raise ConfigurationError(f"unsupported stdio_protocol: {stdio_protocol!r}")
        self.resolved_profile = resolve_profiles(provider, model)
        self.codec.validate_profile(self.resolved_profile)
        self.provider = deepcopy(provider)
        self.model = deepcopy(model)
        self.stdio_protocol = stdio_protocol
        self._clients: dict[Hashable, Any] = {}
        self._client_resources = AsyncExitStack()
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MyLLM is closed")

    def _get_client(self, options: dict) -> Any:
        self._check_open()
        profile = self.resolved_profile
        transport = self.codec.transport(profile.protocol)
        key = transport.client_key(profile, **options)
        if key not in self._clients:
            # SDK construction is synchronous: no task can interleave between
            # lookup and registration, including concurrent first requests.
            client = transport.create_client(profile, **options)
            self._client_resources.push_async_callback(transport.close_client, client)
            self._clients[key] = client
        return self._clients[key]

    async def aclose(self) -> None:
        """Close cached SDK clients once, after finishing/closing response streams."""
        if not self._closed:
            self._closed = True
            self._clients.clear()
            await self._client_resources.aclose()

    async def __aenter__(self) -> MyLLM:
        self._check_open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def guess_stdio_protocol(self, messages_or_tools: list[Any]) -> Protocol:
        """Bind a uniquely inferred I/O format once; otherwise require explicit selection."""
        if self.stdio_protocol is None:
            guessed = self.codec.match_protocol(messages_or_tools)
            if guessed is None:
                raise MessageValidationError(
                    "stdio_protocol is not specified and cannot be inferred uniquely "
                    "from the input; set stdio_protocol explicitly"
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
        self._check_open()
        if not isinstance(stream, bool):
            raise ConfigurationError("stream must be a bool")
        # Preparation does not await: inspect borrowed inputs, then let Encoder
        # capture its one mutable snapshot before any SDK execution can yield.
        history = list(dictionary_views(messages))
        tool_values = (
            None if tools is None or isinstance(tools, Omit) else list(dictionary_views(tools))
        )
        # Tool declarations can disambiguate otherwise identical text-only histories.
        protocol = self.guess_stdio_protocol([*history, *(tool_values or [])])
        self.codec.validate_input(protocol, history, tool_values)
        intermediate = self.codec.encode(history, tool_values, input_format=protocol)
        profile = self.resolved_profile
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
        state = StreamState(profile.model)
        result = MessageStream(self._stream(request, options, state, protocol), state)
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
        self, request: dict, options: dict, state: StreamState, stdio_protocol: Protocol
    ) -> AsyncGenerator[dict, None]:
        profile = self.resolved_profile
        try:
            client = self._get_client(options)
            async with self.codec.upstream(client, request, profile, **options) as upstream:
                async for event in upstream:
                    encoded = self.codec.encode(
                        chunks=[event], input_format=profile.protocol, profile=profile
                    )["chunks"]
                    for output in self._render(state.feed(encoded), state, stdio_protocol):
                        yield output
                final = state.finish()
                converted = self.codec.result(final[-1]["messages"], target_format=stdio_protocol)
                for output in self._render(final, state, stdio_protocol):
                    yield output
            state.result = converted
        except StreamError as exc:
            exc.partial_message = self.codec.result(
                state.messages(partial=True), target_format=stdio_protocol
            )
            raise


async def generate(
    *,
    provider: ProviderProfile,
    model: ModelProfile | str,
    messages: Sequence[Any],
    stdio_protocol: Protocol | None = None,
    **parameters: Any,
) -> MessageStream | dict | list:
    """One-shot generation; a returned stream owns the temporary MyLLM lifetime."""
    selected = ModelProfile(model=model) if isinstance(model, str) else model
    async with AsyncExitStack() as resources:
        llm = await resources.enter_async_context(
            MyLLM(provider=provider, model=selected, stdio_protocol=stdio_protocol)
        )
        result = await llm.generate(messages=messages, **parameters)
        if isinstance(result, MessageStream):
            result.add_close_callback(resources.pop_all().aclose)
        return result
