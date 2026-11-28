"""Request-owned SDK clients and streams, with optional instance translated history."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, aclosing
from copy import deepcopy
from dataclasses import fields, replace
from typing import Any
from uuid import uuid4

from my_llm.basic.exceptions import ConfigurationError, MessageValidationError, StreamError
from my_llm.basic.semantic_loss import LossPolicy, SemanticLossHook
from my_llm.codec.converter import MyCodec
from my_llm.middleware.models import (
    FunctionTool,
    InstructionMessage,
    Message,
    TextContent,
    ToolCall,
)
from my_llm.middleware.stream import MessageEvent, Usage
from my_llm.profile.configuration import Protocol, ReasoningEffort
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.resolution import ResolvedProfile, resolve_profiles
from my_llm.profile.types import Attribute, Omit, omit
from my_llm.runtime.message_stream import MessageStream
from my_llm.runtime.state import StreamState
from my_llm.utils.data import _MISSING
from my_llm.utils.json_schema import object_json_schema, parse_json_object

_JSON_TOOL_PROMPT = (
    "Return the requested structured output by calling {tool_name} exactly once. "
    "Use its arguments for the complete answer."
)
_JSON_TEXT_PROMPT = (
    "Return only one JSON object matching this JSON Schema, without commentary "
    "or Markdown fences:\n{schema}"
)


def set__JSON_TOOL_PROMPT(prompt: str) -> None:
    """Replace the global instruction template for tool-based JSON generation.

    Args:
        prompt: A str.format template. Use {tool_name} to insert the synthetic
            tool name; escape literal braces as {{ and }}.

    Affects subsequent generate_json calls on all existing and future MyLLM
    instances. Already prepared requests are unchanged. The template is stored
    without validation; formatting errors surface when generate_json uses it.
    """
    global _JSON_TOOL_PROMPT
    _JSON_TOOL_PROMPT = prompt


def set__JSON_TEXT_PROMPT(prompt: str) -> None:
    """Replace the global instruction template for prompt-based JSON generation.

    Args:
        prompt: A str.format template. Use {schema} to insert the JSON-serialized
            schema with references expanded; escape literal braces as {{ and }}.

    Affects subsequent generate_json calls on all existing and future MyLLM
    instances when function calling is disabled. Already prepared requests are
    unchanged. The template is stored without validation; formatting errors
    surface when generate_json uses it.
    """
    global _JSON_TEXT_PROMPT
    _JSON_TEXT_PROMPT = prompt


class MyLLM:
    """Generate messages using a fixed provider, model and input/output protocol.

    Args:
        provider: Connection settings and default upstream protocol.
        model: Model identity and overrides, including the upstream protocol.
        resolved_profile: Precomputed snapshot for the same provider/model. Skip
            resolution; copy the snapshot and validate the protocol.
        stdio_protocol: Protocol used by both messages and tools, and by results.
            If None, the first generation must uniquely identify a protocol.
            Set "middleware" explicitly for typed Message/FunctionTool input and
            AssistantMessage/MessageEvent output; this format is never inferred.
        semantic_loss_protocol: How supported fields that the target cannot express
            are handled: "raise", "ignore", or "hook".
        input_semantic_loss_hook: Sync or async hook for the complete request draft.
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

    Each call owns its SDK client and response; close response streams explicitly.
    Use an instance within one event loop. Concurrent generation calls are supported with message
    caching disabled. With caching enabled, finish or close each stream before the
    next call; use separate instances for concurrent conversations. Create a new
    instance to change its configuration. Applications own instance reuse and
    session isolation; clear_cache() starts a new translated history.
    """

    def __init__(
        self,
        *,
        provider: ProviderProfile,
        model: ModelProfile,
        resolved_profile: ResolvedProfile | None = None,
        stdio_protocol: Protocol | None = None,
        semantic_loss_protocol: LossPolicy = "raise",
        input_semantic_loss_hook: SemanticLossHook | None = None,
        output_semantic_loss_hook: SemanticLossHook | None = None,
        enable_message_cache: bool = False,
    ) -> None:
        """Create an instance from profiles; see the class for argument contracts."""
        if not isinstance(enable_message_cache, bool):
            raise ConfigurationError("enable_message_cache must be a bool")
        self._codec = MyCodec(
            semantic_loss_protocol=semantic_loss_protocol,
            input_semantic_loss_hook=input_semantic_loss_hook,
            output_semantic_loss_hook=output_semantic_loss_hook,
        )
        self.enable_message_cache = enable_message_cache
        self._message_cache_key = str(uuid4())
        if stdio_protocol is not None and (
            not isinstance(stdio_protocol, str)
            or stdio_protocol != "middleware"
            and stdio_protocol not in self._codec._codecs
        ):
            raise ConfigurationError(f"unsupported stdio_protocol: {stdio_protocol!r}")
        if resolved_profile is None:
            self._resolved_profile = resolve_profiles(
                provider, model, defaults_for=self._codec._defaults
            )
        else:
            if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
                raise ConfigurationError("provider and model must be profile instances")
            if not isinstance(resolved_profile, ResolvedProfile):
                raise ConfigurationError("resolved_profile must be a ResolvedProfile")
            self._resolved_profile = deepcopy(resolved_profile)
        self._codec._validate_profile(self.resolved_profile)
        self._client_parameters = self._codec._client_parameters(self.resolved_profile)
        self.provider = deepcopy(provider)
        if self.provider.provider is None:
            self.provider.provider = Attribute(default_value=self.resolved_profile.protocol)
        self.model = deepcopy(model)
        self._stdio_protocol = stdio_protocol
        self._last_usage: Usage | None = None

    def clear_cache(self) -> None:
        """Clear this instance's translated history and last-input deduplication.

        Finish or close active streams before clearing. Applications own session
        isolation and decide when a reused instance starts a new conversation.
        """
        self._codec._expire_message_cache(self._message_cache_key)

    def compare(self, new_provider: str, new_model: str, new_stdio_protocol: str) -> bool:
        """Compare identifiers used by applications to select a reusable instance.

        Args:
            new_provider: Provider identity, compared with
                provider.provider.default_value, which defaults to the protocol.
            new_model: Model ID, compared with model.model.default_value.
            new_stdio_protocol: Input/output protocol, compared with stdio_protocol.

        Returns:
            True only when all three strings match exactly, including case.
            Before protocol inference, an unbound stdio_protocol matches no string.

        This method does not resolve or compare full configurations, inspect API
        keys, or change instance state. Applications own instance selection and
        session isolation; clear_cache() starts a new translated history.

        Example:
            if llm.compare("vendor", "model-id", "openai_chat"):
                result = await llm.generate(messages=messages)
        """
        return (
            new_provider == self.provider.provider.default_value
            and new_model == self.model.model.default_value
            and new_stdio_protocol == self.stdio_protocol
        )

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
        self._codec._set_semantic_loss_policy(
            semantic_loss_protocol=semantic_loss_protocol,
            input_semantic_loss_hook=input_semantic_loss_hook,
            output_semantic_loss_hook=output_semantic_loss_hook,
        )

    def get_last_usage(self) -> Usage | None:
        """Return independent counters from the last stream or JSON generation.

        generate_json sums reported counters across its initial attempt and retries,
        including failed JSON outputs. Unreported fields remain None. Ordinary
        generate(stream=False) does not expose usage. A regular stream must finish
        successfully before replacing the previous record.
        """
        return replace(self._last_usage) if self._last_usage is not None else None

    async def generate(
        self,
        *,
        messages: Sequence[Any],
        stream: bool = False,
        tools: Any = omit,
        api_key: str | Omit | None = None,
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
            api_key: Per-call authentication value routed through the profile's
                api_key path. None inherits; omit removes the configured connection
                value. Each call owns and closes its SDK client.
            max_tokens: Positive output limit. If absent, inherit the profile default;
                None is an explicit value and omit removes the configured target key.
            reasoning: Enable reasoning with medium effort when no effort is given.
                False conflicts with a non-"none" reasoning_effort.
            reasoning_effort: One of none, low, medium, high, xhigh or max, subject to
                the model's support and configured mapping.
            extra_headers: Request headers merged by key over profile defaults.
            extra_body: Native generation options merged by key over profile defaults.
                Cannot replace managed fields such as model, messages or tools.
            previous_response_id: Nonempty server-side history ID for an
                openai_responses upstream. Other protocols record an unsupported
                semantic loss and omit it. None or omit leaves it out.

        Returns:
            A protocol-native assistant message dictionary, or output items list for
            openai_responses. With stream=True, a MessageStream yielding dictionaries
            in the same protocol; its result becomes available after exhaustion.
            Middleware I/O returns an AssistantMessage; streaming yields MessageEvent
            models and stores the completed AssistantMessage in stream.result.

        Raises:
            MessageValidationError: Input structure or roles are invalid,
                or the input/output protocol cannot be inferred uniquely.
            ConfigurationError: A generation option is invalid or conflicts.
            SemanticLossError: A supported value or requested control is unsupported
                by the target and the active loss policy is "raise".
            StreamError: The response fails or ends without a valid completion.

        SDK errors retain their original types. Input messages and mutable options
        are isolated when this coroutine is awaited. Include the completed result
        in the next history, or in the next incremental batch when caching is enabled.

        Request preparation binds the input protocol, validates its native or typed
        message structure, repairs tool-call/result associations, and translates the
        history into upstream SDK arguments. Missing results become empty results;
        duplicate call IDs are collapsed and duplicate result contents are merged.
        Entire messages containing orphan or ID-less results are discarded,
        including their text; result IDs are never inferred from names/order.
        If no usable history or cached history remains, validation fails without
        creating a new user prompt.
        Responses with previous_response_id may reference calls in server history.
        With message caching enabled, repairs apply only to the incremental batch;
        supply complete tool rounds together. Cached history is not repaired again.

        Both return modes consume an upstream SDK stream. Streaming returns after
        preparation; client creation and network I/O start on iteration. Non-streaming
        consumes and closes that same stream here before returning its final result.
        """

        inputs = dict(
            messages=messages,
            stream=stream,
            tools=tools,
            api_key=api_key,
            max_tokens=max_tokens,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            extra_headers=extra_headers,
            extra_body=extra_body,
            previous_response_id=previous_response_id,
        )
        prepared, protocol = await self._prepare_generation(inputs)
        return await self._consume(prepared, StreamState(self.resolved_profile.model), protocol)

    async def _prepare_generation(
        self,
        inputs: dict,
        *,
        middleware_tools: list[FunctionTool] | None = None,
        request_suffix: list[Message] | None = None,
    ) -> tuple[dict, Protocol]:
        """Bind input and prepare one reusable request without starting execution."""
        if (
            inputs["previous_response_id"] is not None
            and inputs["previous_response_id"] is not omit
            and self.enable_message_cache
        ):
            raise ConfigurationError(
                "previous_response_id is not supported when enable_message_cache is True"
            )

        protocol = self._codec._bind_input(inputs, self.stdio_protocol)
        self._stdio_protocol = protocol
        profile = self.resolved_profile
        prepared = await self._codec._prepare(
            inputs,
            profile,
            input_format=protocol,
            client=self._client_parameters,
            enable_message_cache=self.enable_message_cache,
            message_cache_key=self._message_cache_key,
            middleware_tools=middleware_tools,
            request_suffix=request_suffix,
        )
        return prepared, protocol

    async def _consume(
        self, prepared: dict, state: StreamState, protocol: Protocol
    ) -> MessageStream | dict | list | Message:
        """Execute one request with an owned state and consume it when non-streaming."""
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
        """Render public middleware events and advance the output sequence number."""
        output = self._codec._decode(
            chunks=events,
            target_format=protocol,
            input_format=self.resolved_profile.protocol,
            origin_source="output",
            semantic_profile=self.resolved_profile,
            sequence_number=state.sequence_number,
            stream_metadata=state.metadata,
            stream_indices=state.stream_indices,
            **state.metadata,
        )["chunks"]
        state.sequence_number += len(output)
        return output

    async def _stream(
        self, prepared: dict, state: StreamState, stdio_protocol: Protocol
    ) -> AsyncGenerator[dict | MessageEvent, None]:
        """Execute one prepared request and accumulate its response in shared state.

        Create a request-owned SDK client and keep the response inside the
        transport's cleanup context. Native events become validated middleware
        operations, then StreamState emits text incrementally and holds tool calls
        until successful finalization. Render those events only for streaming calls;
        both modes render the same completed assistant message.

        Publish result and usage after upstream cleanup succeeds. StreamError gains
        a rendered partial message when possible; rendering failures become notes
        on the original error. Transport/SDK failures retain their types. Closing
        this generator exits the response context and closes its SDK client.
        """
        profile = self.resolved_profile
        streaming = prepared["local"]["stream"]
        try:
            transport = self._codec._transport(profile.protocol)
            async with AsyncExitStack() as resources:
                client = transport.create_client(prepared["client"])
                resources.push_async_callback(transport.close_client, client)
                upstream = await resources.enter_async_context(
                    self._codec._upstream(client, prepared["request"], profile)
                )
                chunk_offset = 0
                async for event in upstream:
                    encoded = self._codec._encode_chunks(
                        chunks=[event],
                        input_format=profile.protocol,
                        profile=profile,
                        chunk_offset=chunk_offset,
                    )
                    chunk_offset += 1
                    emitted = state.feed(encoded)
                    if streaming and emitted:
                        for output in self._render(emitted, state, stdio_protocol):
                            yield output
                final = state.finish()
                converted = self._codec._result(
                    final[-1]["messages"],
                    target_format=stdio_protocol,
                    input_format=profile.protocol,
                    origin_source="output",
                    semantic_profile=profile,
                    final_event=final[-1],
                    stream_indices=state.stream_indices,
                    **state.metadata,
                )
                if streaming:
                    for output in self._render(final, state, stdio_protocol):
                        yield output
            state.result = converted
            self._last_usage = Usage.from_record(state.usage) if streaming else None
        except StreamError as exc:
            try:
                exc.partial_message = self._codec._result(
                    state.messages(partial=True),
                    target_format=stdio_protocol,
                    input_format=profile.protocol,
                    origin_source="output",
                )
            except Exception as render_error:
                exc.add_note(f"Failed to render partial_message: {render_error!r}")
            raise

    async def generate_json(
        self,
        *,
        messages: Sequence[Any],
        schema: type | dict[str, Any],
        max_retries: int = 0,
        api_key: str | Omit | None = None,
        max_tokens: Any = _MISSING,
        reasoning: bool | None = None,
        reasoning_effort: ReasoningEffort = "none",
        extra_headers: Any = None,
        extra_body: Any = None,
        previous_response_id: Any = None,
    ) -> dict[str, Any]:
        """Generate a JSON object using a synthetic tool or a JSON-only prompt.

        Args:
            messages: History in this instance's stdio protocol, following generate's
                validation, inference and incremental-cache rules.
            schema: JSON Schema object, or a Pydantic-supported object type (for
                example a BaseModel, dataclass or TypedDict). The root must describe
                an object. The schema guides generation; results are not schema-validated.
            max_retries: Additional attempts after an invalid JSON output, default 0.
                Must be a nonnegative integer. Independent of the Profile's SDK
                connection retry setting; the initial attempt is not a retry.
            api_key: Per-call authentication; None inherits and omit removes it.
            max_tokens: Output limit, with the same inheritance rules as generate.
            reasoning: Enable or disable reasoning as in generate.
            reasoning_effort: Requested reasoning effort as in generate.
            extra_headers: Headers merged over profile defaults.
            extra_body: Native options merged over profile defaults.
            previous_response_id: Responses server history ID; incompatible with
                the instance message cache, as in generate.

        Returns:
            The synthetic tool's argument dictionary when function calling is enabled.
            Otherwise, a JSON object extracted from response text, including fenced
            JSON or JSON surrounded by prose. Only an unambiguous complete object is
            accepted; malformed parents, duplicate keys and non-finite numbers are
            rejected. No tools are executed. Output instructions
            apply only to this request and are excluded from the message cache.
            get_last_usage() exposes the sum of reported usage across all attempts,
            including failed JSON outputs, even when retries are exhausted. Missing
            counters remain None; known counters are summed without inventing usage.

        Raises:
            TypeError: schema is neither a dictionary nor a supported Python type.
            ValueError: The schema cannot be inlined or its root is not an object,
                the expected tool call is
                missing or ambiguous, or response text contains no valid JSON object
                after max_retries additional attempts.
            ConfigurationError, MessageValidationError, SemanticLossError, StreamError:
                As for generate. SDK errors retain their original types.

        This method always collects the complete response and closes its resources.
        Capability selection uses resolved_profile.supports_function_calling; upstream
        SDK/configuration errors do not trigger JSON retries or switch generation mode.
        Each attempt owns its response and client. Schema/input errors occur before
        retry execution and leave the previous usage record unchanged.
        """
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ConfigurationError("max_retries must be a non-negative integer")
        tool = FunctionTool(
            name="structured_output",
            description="Return the requested structured output.",
            parameters=object_json_schema(schema),
        )
        use_tool = self.resolved_profile.get("supports_function_calling", False)
        instruction = (
            _JSON_TOOL_PROMPT.format(tool_name=tool.name)
            if use_tool
            else _JSON_TEXT_PROMPT.format(schema=json.dumps(tool.parameters, ensure_ascii=False))
        )
        prepared, _ = await self._prepare_generation(
            dict(
                messages=messages,
                stream=False,
                tools=omit,
                api_key=api_key,
                max_tokens=max_tokens,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                extra_headers=extra_headers,
                extra_body=extra_body,
                previous_response_id=previous_response_id,
            ),
            middleware_tools=[tool] if use_tool else None,
            request_suffix=[InstructionMessage("system", [TextContent(instruction)])],
        )
        total_usage: Usage | None = None
        for attempt in range(max_retries + 1):
            state = StreamState(self.resolved_profile.model)
            try:
                try:
                    result = await self._consume(prepared, state, "middleware")
                except MessageValidationError as exc:
                    # Invalid generated tool arguments fail during response assembly.
                    error = exc
                else:
                    try:
                        if use_tool:
                            calls = [
                                part
                                for part in result.content
                                if isinstance(part, ToolCall) and part.name == tool.name
                            ]
                            if len(calls) != 1:
                                raise ValueError(
                                    f"expected exactly one {tool.name} tool call, got {len(calls)}"
                                )
                            return calls[0].arguments
                        return parse_json_object(
                            "".join(
                                part.text
                                for part in result.content
                                if isinstance(part, TextContent)
                            )
                        )
                    except ValueError as exc:
                        error = exc
            finally:
                usage = Usage.from_record(state.usage)
                if usage is not None:
                    if total_usage is None:
                        total_usage = usage
                    else:
                        for field in fields(Usage):
                            value = getattr(usage, field.name)
                            if value is not None:
                                previous = getattr(total_usage, field.name)
                                setattr(total_usage, field.name, (previous or 0) + value)
                self._last_usage = total_usage
            if attempt == max_retries:
                raise error
