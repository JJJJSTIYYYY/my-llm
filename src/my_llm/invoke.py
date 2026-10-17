"""Official SDK execution with request-owned stream and history state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from copy import deepcopy
from typing import Any

from openai import Omit, omit

from my_llm.codec.converter import MyCodec
from my_llm.codec.shared.conversion import dictionaries
from my_llm.codec.shared.validation import validate_history
from my_llm.exceptions import ConfigurationError, MessageValidationError, StreamError
from my_llm.profile.configuration import PROTOCOLS, Protocol, ReasoningEffort
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.resolution import resolve_profiles
from my_llm.stream._state import StreamState
from my_llm.stream.message_stream import MessageStream


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
        validate_history(
            intermediate["messages"],
            allow_external_calls=self.codec.decoder(profile.protocol).allows_external_calls(
                previous_response_id
            ),
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
    ) -> AsyncIterator[dict]:
        profile = self.resolved_profile
        try:
            async with self.codec.upstream(request, profile, **options) as upstream:
                async for event in upstream:
                    encoded = self.codec.encode(
                        chunks=[event], input_format=profile.protocol, profile=profile
                    )["chunks"]
                    for output in self._render(state.feed(encoded), state, stdio_protocol):
                        yield output
                final = state.finish()
                converted = self.codec.result(state.messages(), target_format=stdio_protocol)
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
    """One-shot generation with the same standard I/O as MyLLM.generate()."""
    selected = ModelProfile(model=model) if isinstance(model, str) else model
    return await MyLLM(provider=provider, model=selected, stdio_protocol=stdio_protocol).generate(
        messages=messages, **parameters
    )
