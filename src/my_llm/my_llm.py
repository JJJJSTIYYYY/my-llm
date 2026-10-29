"""Instance-owned SDK clients and optional translated history, with per-call streams."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Hashable, Sequence
from contextlib import AsyncExitStack, aclosing
from copy import deepcopy
from dataclasses import replace
from typing import Any
from uuid import uuid4

from my_llm.basic.data import _MISSING
from my_llm.basic.exceptions import ConfigurationError, StreamError
from my_llm.basic.semantic_loss import LossPolicy, SemanticLossHook
from my_llm.codec.converter import MyCodec
from my_llm.middleware.models import Message
from my_llm.middleware.stream import MessageEvent, Usage
from my_llm.profile.configuration import Protocol, ReasoningEffort
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.resolution import ResolvedProfile, resolve_profiles
from my_llm.profile.types import Attribute, omit
from my_llm.runtime.message_stream import MessageStream
from my_llm.runtime.state import StreamState


class MyLLM:
    """Generate messages using a fixed provider, model and input/output protocol.

    Args:
        provider: Connection settings and an explicitly configured upstream protocol.
        model: Model identity and overrides for provider defaults.
        stdio_protocol: Protocol used by both messages and tools, and by results.
            If None, the first generation must uniquely identify a protocol.
            Set "middleware" explicitly for typed Message/FunctionTool input and
            AssistantMessage/MessageEvent output; this format is never inferred.
        semantic_loss_protocol: How supported fields that the target cannot express
            are handled: "raise", "ignore", or "hook".
        input_semantic_loss_hook: Synchronous hook for the complete request draft.
        output_semantic_loss_hook: Synchronous hook for response drafts. Hooks receive
            the draft, grouped losses and context, and modify the draft in place.
        enable_message_cache: Cache translated input batches for this instance.
            Results are cached before semantic-loss adaptation. Hooks own their
            changes and any corresponding cache maintenance.
            Subsequent calls supply only incremental messages, including prior
            assistant results and complete tool rounds. Consecutive equal input
            batches reuse cached history, including retries after an unconsumed
            or failed stream. Defaults to False. Calls must be sequential when enabled.

    Attributes:
        provider: Copy of the supplied provider profile.
        model: Copy of the supplied model profile.
        resolved_profile: Read-only configuration snapshot for this instance.
        stdio_protocol: Read-only input/output protocol, or None before inference.
        codec: Converter available for the registered protocols.
        client_parameters: Prepared connection settings; treat as read-only.

    Raises:
        ConfigurationError: Profiles or the selected protocol are invalid.
        UnsupportedProviderError: The upstream protocol cannot be used.
        ModuleNotFoundError: The upstream protocol's optional SDK is not installed.

    Use ``async with MyLLM(...)`` or ``await llm.aclose()`` to release resources.
    Finish or close response streams before closing the instance. Use an instance
    within one event loop. Concurrent generation calls are supported with message
    caching disabled. With caching enabled, finish or close each stream before the
    next call; use separate instances for concurrent conversations. Create a new
    instance to change its configuration.
    """

    def __init__(
        self,
        *,
        provider: ProviderProfile,
        model: ModelProfile,
        stdio_protocol: Protocol | None = None,
        semantic_loss_protocol: LossPolicy = "raise",
        input_semantic_loss_hook: SemanticLossHook | None = None,
        output_semantic_loss_hook: SemanticLossHook | None = None,
        enable_message_cache: bool = False,
    ) -> None:
        """Create an instance from profiles; see the class for argument contracts."""
        if not isinstance(enable_message_cache, bool):
            raise ConfigurationError("enable_message_cache must be a bool")
        self.codec = MyCodec(
            semantic_loss_protocol=semantic_loss_protocol,
            input_semantic_loss_hook=input_semantic_loss_hook,
            output_semantic_loss_hook=output_semantic_loss_hook,
        )
        self.enable_message_cache = enable_message_cache
        self._message_cache_key = str(uuid4())
        if stdio_protocol is not None and (
            not isinstance(stdio_protocol, str)
            or stdio_protocol != "middleware"
            and stdio_protocol not in self.codec.codecs
        ):
            raise ConfigurationError(f"unsupported stdio_protocol: {stdio_protocol!r}")
        self._resolved_profile = resolve_profiles(provider, model, defaults_for=self.codec.defaults)
        self.codec.validate_profile(self.resolved_profile)
        self.client_parameters = self.codec.client_parameters(self.resolved_profile)
        self.provider = deepcopy(provider)
        self.model = deepcopy(model)
        self._stdio_protocol = stdio_protocol
        self._last_usage: Usage | None = None
        self._clients: dict[Hashable, Any] = {}
        self._client_resources = AsyncExitStack()
        self._closed = False

    @property
    def resolved_profile(self) -> ResolvedProfile:
        """Read-only snapshot binding this instance's upstream protocol and model."""
        return self._resolved_profile

    @property
    def stdio_protocol(self) -> Protocol | None:
        """The fixed input/output protocol, selected explicitly or on first use."""
        return self._stdio_protocol

    def set_semantic_loss_policy(
        self,
        *,
        semantic_loss_protocol: LossPolicy = "raise",
        input_semantic_loss_hook: SemanticLossHook | None = None,
        output_semantic_loss_hook: SemanticLossHook | None = None,
    ) -> None:
        """Replace this instance's semantic-loss policy and direction-specific hooks."""
        self.codec.set_semantic_loss_policy(
            semantic_loss_protocol=semantic_loss_protocol,
            input_semantic_loss_hook=input_semantic_loss_hook,
            output_semantic_loss_hook=output_semantic_loss_hook,
        )

    def get_last_usage(self) -> Usage | None:
        """Return independent counters from the last completed streaming call.

        Unreported fields are None. Non-streaming calls do not expose usage.
        A stream must finish successfully before replacing the previous record.
        """
        return replace(self._last_usage) if self._last_usage is not None else None

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MyLLM is closed")

    def _get_client(self, prepared: dict) -> Any:
        self._check_open()
        profile = self.resolved_profile
        transport = self.codec.transport(profile.protocol)
        key = prepared["local"]["client_key"]
        if key not in self._clients:
            # SDK construction is synchronous: no task can interleave between
            # lookup and registration, including concurrent first requests.
            client = transport.create_client(prepared["client"])
            self._client_resources.push_async_callback(transport.close_client, client)
            self._clients[key] = client
        return self._clients[key]

    async def aclose(self) -> None:
        """Close this instance and release its clients; repeated calls are harmless.

        Finish or close all response streams first. Further generation and consumption
        of streams that have not started raise RuntimeError.
        """
        if not self._closed:
            self._closed = True
            self._clients.clear()
            self.codec.expire_message_cache(self._message_cache_key)
            await self._client_resources.aclose()

    async def __aenter__(self) -> MyLLM:
        """Return this instance, raising RuntimeError if it is already closed."""
        self._check_open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close this instance on leaving the asynchronous context."""
        await self.aclose()

    def guess_stdio_protocol(self, messages_or_tools: list[Any]) -> Protocol:
        """Return the bound protocol, or infer and bind it from a whole input list.

        Args:
            messages_or_tools: Standard messages or tool declarations to inspect.

        Returns:
            The existing protocol, or the uniquely matching registered protocol.

        Raises:
            MessageValidationError: Unbound input matches zero or multiple protocols.
                Empty input cannot identify a protocol. Failed inference leaves the
                instance unbound.
        """
        if self.stdio_protocol is None:
            self._stdio_protocol = self.codec.bind_input(
                {"messages": messages_or_tools, "tools": omit}, None
            )
        return self.stdio_protocol

    async def generate(
        self,
        *,
        messages: Sequence[Any],
        stream: bool = False,
        tools: Any = omit,
        max_tokens: Any = _MISSING,
        reasoning: bool | None = None,
        reasoning_effort: ReasoningEffort = "none",
        extra_headers: Any = None,
        extra_body: Any = None,
        previous_response_id: Any = None,
    ) -> MessageStream | dict | list | Message:
        """Generate a completed message or a closable asynchronous event stream.

        Args:
            messages: Nonempty history in this instance's input/output protocol.
                Items may be dictionaries or supported official SDK models.
                With stdio_protocol="middleware", pass Message models instead.
                With message caching enabled, supply only the new message batch.
                Repeating the last batch retries it without appending it again.
            stream: If True, return a MessageStream to consume with ``async with``.
            tools: Tool declarations in the same protocol. None or omit leaves them
                out; an empty list sends an empty tool list. Tools are not executed.
                Middleware I/O accepts FunctionTool models.
            max_tokens: Positive output limit. If absent, inherit the profile default;
                None is an explicit value and omit removes the configured target key.
            reasoning: Enable reasoning with medium effort when no effort is given.
                False conflicts with a non-"none" reasoning_effort.
            reasoning_effort: One of none, low, medium, high, xhigh or max, subject to
                the model's support and configured mapping.
            extra_headers: Request headers merged by key over profile defaults.
            extra_body: Native generation options merged by key over profile defaults.
                Cannot replace managed fields such as model, messages or tools.
            previous_response_id: Nonempty server-side history ID, supported only for
                an openai_responses upstream. None or omit leaves it out.

        Returns:
            A protocol-native assistant message dictionary, or output items list for
            openai_responses. With stream=True, a MessageStream yielding dictionaries
            in the same protocol; its result becomes available after exhaustion.
            Middleware I/O returns an AssistantMessage; streaming yields MessageEvent
            models and stores the completed AssistantMessage in stream.result.

        Raises:
            RuntimeError: This instance is closed.
            MessageValidationError: Input or tool-call/result associations are invalid,
                or the input/output protocol cannot be inferred uniquely.
            ConfigurationError: A generation option is invalid or conflicts.
            UnsupportedCapabilityError: A generation control is unavailable.
            SemanticLossError: A supported value cannot be represented by the target
                protocol and the active loss policy is "raise".
            StreamError: The response fails or ends without a valid completion.

        SDK errors retain their original types. Input messages and mutable options
        are isolated when this coroutine is awaited. Include the completed result
        in the next history, or in the next incremental batch when caching is enabled.
        """
        self._check_open()

        if (
            previous_response_id is not None
            and previous_response_id is not omit
            and self.enable_message_cache
        ):
            raise ConfigurationError(
                "previous_response_id is not supported when enable_message_cache is True"
            )

        inputs = dict(
            messages=messages,
            stream=stream,
            tools=tools,
            max_tokens=max_tokens,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            extra_headers=extra_headers,
            extra_body=extra_body,
            previous_response_id=previous_response_id,
        )
        protocol = self.codec.bind_input(inputs, self.stdio_protocol)
        self._stdio_protocol = protocol
        profile = self.resolved_profile
        prepared = self.codec.prepare(
            inputs,
            profile,
            input_format=protocol,
            client=self.client_parameters,
            enable_message_cache=self.enable_message_cache,
            message_cache_key=self._message_cache_key,
        )
        state = StreamState(profile.model)
        result = MessageStream(self._stream(prepared, state, protocol), state)
        if prepared["local"]["stream"]:
            return result
        # Collected and streaming results share one execution and cleanup path.
        async with aclosing(result):
            async for _ in result:
                pass
        return result.result

    def _render(
        self, events: list[dict], state: StreamState, protocol: Protocol
    ) -> list[dict | MessageEvent]:
        output = self.codec.decode(
            chunks=events,
            target_format=protocol,
            input_format=self.resolved_profile.protocol,
            origin_source="output",
            semantic_profile=self.resolved_profile,
            sequence_number=state.sequence_number,
            **state.metadata,
        )["chunks"]
        state.sequence_number += len(output)
        return output

    async def _stream(
        self, prepared: dict, state: StreamState, stdio_protocol: Protocol
    ) -> AsyncGenerator[dict | MessageEvent, None]:
        profile = self.resolved_profile
        streaming = prepared["local"]["stream"]
        try:
            client = self._get_client(prepared)
            async with self.codec.upstream(client, prepared["request"], profile) as upstream:
                chunk_offset = 0
                async for event in upstream:
                    encoded = self.codec.encode(
                        chunks=[event],
                        input_format=profile.protocol,
                        profile=profile,
                        chunk_offset=chunk_offset,
                    )["chunks"]
                    chunk_offset += 1
                    emitted = state.feed(encoded)
                    if streaming and emitted:
                        for output in self._render(emitted, state, stdio_protocol):
                            yield output
                final = state.finish()
                converted = self.codec.result(
                    final[-1]["messages"],
                    target_format=stdio_protocol,
                    input_format=profile.protocol,
                    origin_source="output",
                    semantic_profile=profile,
                )
                if streaming:
                    for output in self._render(final, state, stdio_protocol):
                        yield output
            state.result = converted
            self._last_usage = Usage.from_record(state.usage) if streaming else None
        except StreamError as exc:
            exc.partial_message = self.codec.result(
                state.messages(partial=True),
                target_format=stdio_protocol,
                input_format=profile.protocol,
                origin_source="output",
            )
            raise


async def generate(
    *,
    provider: ProviderProfile,
    model: ModelProfile | str,
    messages: Sequence[Any],
    stdio_protocol: Protocol | None = None,
    semantic_loss_protocol: LossPolicy = "raise",
    input_semantic_loss_hook: SemanticLossHook | None = None,
    output_semantic_loss_hook: SemanticLossHook | None = None,
    **parameters: Any,
) -> MessageStream | dict | list | Message:
    """Generate once with an automatically managed temporary MyLLM instance.

    Args:
        provider: Provider connection settings and upstream protocol.
        model: ModelProfile or a nonempty model ID. An ID creates a ModelProfile;
            it does not look up the provider's model catalogue.
        messages: History in the selected input/output protocol.
        stdio_protocol: Input/output protocol, or None for unique inference.
            Typed middleware input requires explicit "middleware".
        semantic_loss_protocol: "raise", "ignore", or "hook" for semantic loss.
        input_semantic_loss_hook: Optional request draft hook for the "hook" policy.
        output_semantic_loss_hook: Optional response draft hook for the "hook" policy.
        **parameters: Generation options accepted by MyLLM.generate, including
            stream, tools, max_tokens, reasoning and extra_body.

    Returns:
        A completed native message/items, or a MessageStream when stream=True.
        Close a returned stream with ``async with`` or ``await stream.aclose()``
        to release its temporary instance, even if iteration never starts.

    Raises:
        ConfigurationError: The profiles or options are invalid.
        MessageValidationError: Input is invalid or its protocol is ambiguous.
        StreamError: The response fails or does not complete successfully.

    The return formats and remaining error behavior match MyLLM.generate.
    """
    selected = (
        ModelProfile(model=Attribute(default_value=model)) if isinstance(model, str) else model
    )
    async with AsyncExitStack() as resources:
        llm = await resources.enter_async_context(
            MyLLM(
                provider=provider,
                model=selected,
                stdio_protocol=stdio_protocol,
                semantic_loss_protocol=semantic_loss_protocol,
                input_semantic_loss_hook=input_semantic_loss_hook,
                output_semantic_loss_hook=output_semantic_loss_hook,
            )
        )
        result = await llm.generate(messages=messages, **parameters)
        if isinstance(result, MessageStream):
            result.add_close_callback(resources.pop_all().aclose)
        return result
