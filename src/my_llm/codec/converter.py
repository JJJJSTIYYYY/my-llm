"""Reusable, profile-independent conversion among registered protocols."""

from collections.abc import Callable, MutableMapping
from typing import Any

from my_llm.basic.data import dictionary_views
from my_llm.basic.exceptions import (
    ConfigurationError,
    MessageValidationError,
    UnsupportedProviderError,
)
from my_llm.basic.protocols import (
    BUILTIN_CODECS,
    builtin_codec,
    register_codec,
    registered_codec_factories,
    require_sdk,
    sdk_available,
)
from my_llm.basic.semantic_loss import (
    BaseSemanticLossAdaptor,
    LossPolicy,
    SemanticLossContext,
    SemanticLossHook,
)
from my_llm.cache.message_cache import MessageCache
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.output import output_chunks, output_messages
from my_llm.codec.shared.stream import is_batch
from my_llm.middleware.models import (
    FunctionTool,
    Message,
    messages_from_validated_wire,
    tools_from_wire,
)
from my_llm.middleware.protocol import ProtocolMiddleWare, generation_input, wire_values
from my_llm.middleware.stream import MessageEvent
from my_llm.middleware.validation import validate_events, validate_messages, validate_operations
from my_llm.profile.configuration import Protocol, ProtocolDefaults
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit


class _CodecInstances(MutableMapping[str, BaseCodec]):
    """Instance-owned factory snapshot; unused protocols allocate no components."""

    def __init__(self, adaptor: BaseSemanticLossAdaptor):
        """Initialize a lazy registry whose codecs share this converter's adaptor."""
        self.factories = registered_codec_factories()
        self.instances: dict[str, BaseCodec] = {}
        self.adaptor = adaptor

    def __getitem__(self, protocol: str) -> BaseCodec:
        """Return an owned codec, constructing it lazily from its registered factory."""
        if protocol not in self.instances:
            codec = self.factories[protocol]()
            codec.semantic_loss_adaptor = self.adaptor
            self.instances[protocol] = codec
        return self.instances[protocol]

    def __setitem__(self, protocol: str, codec: BaseCodec) -> None:
        """Attach this converter's adaptor and store an explicitly supplied codec."""
        codec.semantic_loss_adaptor = self.adaptor
        self.instances[protocol] = codec
        self.factories.setdefault(protocol, type(codec))

    def __delitem__(self, protocol: str) -> None:
        """Remove an owned codec instance and its local factory entry."""
        del self.factories[protocol]
        self.instances.pop(protocol, None)

    def __iter__(self):
        """Iterate protocol names available through registered factories."""
        return iter(self.factories)

    def __len__(self):
        """Count available protocol factories without instantiating codecs."""
        return len(self.factories)

    def __contains__(self, protocol):
        """Check protocol availability without instantiating its codec."""
        return protocol in self.factories


class MyCodec:
    """Convert messages, tools and complete event lists between standard protocols.

    Construction selects a semantic-loss policy and snapshots registered factories.
    Protocol components are constructed lazily within this converter. Register
    custom protocols before creating instances.
    Calls do not retain profiles, inferred formats or stream fragments.
    Use convert for direct conversion; profile=None selects standard field behavior.
    Only recognized fields are translated; conversion is not guaranteed lossless.
    """

    __slots__ = ("codecs", "semantic_loss_adaptor", "_message_cache")

    def __init__(
        self,
        semantic_loss_protocol: LossPolicy = "raise",
        input_semantic_loss_hook: SemanticLossHook | None = None,
        output_semantic_loss_hook: SemanticLossHook | None = None,
    ) -> None:
        """Create instance-owned protocol components, loss policy and translated-history cache."""
        self.semantic_loss_adaptor = BaseSemanticLossAdaptor(
            semantic_loss_protocol, input_semantic_loss_hook, output_semantic_loss_hook
        )
        self.codecs = _CodecInstances(self.semantic_loss_adaptor)
        self._message_cache = MessageCache()

    def expire_message_cache(self, key: str) -> None:
        """Release the translated input history associated with one MyLLM instance."""
        self._message_cache.expire(key)

    def set_semantic_loss_policy(
        self,
        semantic_loss_protocol: LossPolicy = "raise",
        input_semantic_loss_hook: SemanticLossHook | None = None,
        output_semantic_loss_hook: SemanticLossHook | None = None,
    ) -> None:
        """Replace the adaptor shared by all current and future codecs of this converter."""
        adaptor = BaseSemanticLossAdaptor(
            semantic_loss_protocol, input_semantic_loss_hook, output_semantic_loss_hook
        )
        self.semantic_loss_adaptor = adaptor
        self.codecs.adaptor = adaptor
        for codec in self.codecs.instances.values():
            codec.semantic_loss_adaptor = adaptor

    @staticmethod
    def register(factory: Callable[[], BaseCodec]) -> None:
        """Register a factory producing independent codecs and components."""
        if not callable(factory):
            raise ConfigurationError("protocol implementation must be a codec factory")
        codec = factory()
        other = factory()
        if not isinstance(codec, BaseCodec) or not isinstance(other, BaseCodec):
            raise ConfigurationError("codec factory must return a BaseCodec")
        protocol = codec.protocol_or_api_style
        if not isinstance(protocol, str) or not protocol.strip():
            raise ConfigurationError("protocol name must be a non-empty string")
        if protocol == "middleware":
            raise ConfigurationError("middleware is reserved for direct intermediate I/O")
        if not isinstance(codec.defaults, ProtocolDefaults):
            raise ConfigurationError("protocol defaults must be ProtocolDefaults")
        for component, expected in (
            (codec.encoder, BaseEncoder),
            (codec.decoder, BaseDecoder),
            (codec.transport, BaseTransport),
        ):
            if not isinstance(component, expected):
                raise ConfigurationError(f"protocol component must be a {expected.__name__}")
        if codec.encoder.protocol != protocol or codec.decoder.protocol != protocol:
            raise ConfigurationError("codec, encoder and decoder must use the same protocol name")
        if codec is other or any(
            getattr(codec, name) is getattr(other, name)
            for name in ("encoder", "decoder", "transport")
        ):
            raise ConfigurationError("codec factory must create independent protocol components")
        register_codec(factory, protocol=protocol)

    def _codec(self, protocol: str) -> BaseCodec:
        """Resolve a supported native codec, rejecting middleware or unknown protocol names."""
        try:
            return self.codecs[protocol]
        except (KeyError, TypeError) as exc:
            require_sdk(protocol)
            raise UnsupportedProviderError(f"unsupported protocol: {protocol!r}") from exc

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        """Return the unique protocol matching the whole list, or None."""
        matches = [
            protocol
            for protocol, codec in self.codecs.items()
            if codec.match_protocol(messages_or_tools) is not None
        ]
        return matches[0] if len(matches) == 1 else None

    def decoder(self, protocol: str) -> BaseDecoder:
        """Return this instance's intermediate-to-native converter."""
        return self._codec(protocol).decoder

    def transport(self, protocol: str) -> BaseTransport:
        """Return the SDK execution component composed by this protocol's codec."""
        return self._codec(protocol).transport

    def defaults(self, protocol: str) -> ProtocolDefaults:
        """Borrow this instance's default declarations for profile resolution."""
        return self._codec(protocol).defaults

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate protocol-specific settings in a ResolvedProfile.

        Raises ConfigurationError for incompatible options, or
        UnsupportedProviderError for an unregistered protocol. A missing optional
        SDK for a built-in protocol raises ModuleNotFoundError.
        """
        self._codec(profile.protocol).validate_profile(profile)

    def validate_input(self, protocol: str, messages: list, tools: list | None = None) -> None:
        """Validate messages and optional tools in the specified standard protocol.

        Raises MessageValidationError for invalid input. No request is sent.
        """
        if protocol == "middleware":
            generation_input(messages, tools)
        else:
            self._codec(protocol).validate_input(messages, tools)

    def client_parameters(self, profile: ResolvedProfile) -> dict:
        """Return prepared SDK connection arguments for a resolved profile.

        Does not create a client or send a request. Treat the returned settings as
        read-only when reusing them across prepare calls.
        """
        return self.decoder(profile.protocol).client_parameters(profile)

    def bind_input(self, inputs: dict, protocol: Protocol | None) -> Protocol:
        """Return an explicit protocol or infer a unique protocol from inputs.

        Args:
            inputs: Dictionary containing messages and tools. Inference replaces these
                entries with inspected values; pass a separate dictionary if needed.
            protocol: Explicit protocol to return unchanged, or None for inference.

        Raises:
            MessageValidationError: Inference is ambiguous or has no match.

        This method does not remember the selected protocol between calls.

        An explicit protocol skips inference; its message schemas are validated
        during prepare. Middleware models require the explicit middleware protocol.
        Native inference inspects the whole message/tool batch and retains the
        inspected views in inputs so a one-shot sequence is not consumed twice.
        """
        if protocol == "middleware":
            return protocol
        if (
            isinstance(inputs["messages"], (list, tuple))
            and inputs["messages"]
            and isinstance(inputs["messages"][0], Message)
        ) or (
            isinstance(inputs["tools"], (list, tuple))
            and inputs["tools"]
            and isinstance(inputs["tools"][0], FunctionTool)
        ):
            raise MessageValidationError(
                'intermediate inputs require explicit stdio_protocol="middleware"'
            )
        if protocol is not None:
            return protocol
        history = list(dictionary_views(inputs["messages"]))
        tools = inputs["tools"]
        values = None if tools is None or isinstance(tools, Omit) else list(dictionary_views(tools))
        protocol = self.match_protocol([*history, *(values or [])])
        if protocol is None:
            raise MessageValidationError(
                "stdio_protocol is not specified and cannot be inferred uniquely "
                "from the input; set stdio_protocol explicitly"
            )
        # Retain the inspected views; a one-shot sequence must not be read again.
        inputs.update(messages=history, tools=values)
        return protocol

    def prepare(
        self,
        inputs: dict,
        profile: ResolvedProfile,
        *,
        input_format: Protocol,
        client: dict,
        enable_message_cache: bool = False,
        message_cache_key: str | None = None,
    ) -> dict:
        """Prepare one complete generation call without performing network I/O.

        Args:
            inputs: MyLLM.generate options as a dictionary, including messages, tools
                and stream. Values use input_format.
            profile: Resolved upstream provider/model settings.
            input_format: Registered protocol, or "middleware" for typed history/tools
                that bypass the source encoder.
            client: Connection settings from client_parameters(profile).
            enable_message_cache: Append translated incremental input to cached history.
            message_cache_key: Unique instance key required when caching is enabled.

        Returns:
            Dictionary with client (SDK constructor arguments), request (SDK call
            arguments), and local (stream flag and client_key) entries.

        Raises:
            ConfigurationError: Generation options or target request are invalid.
            MessageValidationError: Messages, tools or their associations are invalid.
            SemanticLossError: Content or controls are unsupported by the target and
                the active loss policy is "raise".

        Native input goes through its source encoder; typed middleware input goes
        through generation_input instead. Both paths converge on the upstream
        decoder, which checks canonical structure and call/result associations,
        translates messages/tools, and validates final SDK arguments.

        Message caching stages only the current batch's native translation, after
        association validation. It joins prior translated history during assembly
        and publishes it through BaseCodec.translate before semantic-loss adaptation.
        No SDK client is created and no network request is sent here.
        """
        if enable_message_cache and message_cache_key is None:
            raise ConfigurationError("message caching requires an instance key")
        if enable_message_cache:
            inputs = {**inputs, "enable_message_cache": True}
        if input_format == "middleware":
            intermediate = {
                **{key: value for key, value in inputs.items() if key not in ("messages", "tools")},
                "profile": profile,
                "middleware": generation_input(inputs["messages"], inputs["tools"]),
            }
        else:
            intermediate = self._codec(input_format).encoder.request(inputs, profile)
        target = self._codec(profile.protocol)
        batch = None
        if enable_message_cache:
            batch = self._message_cache.batch(
                message_cache_key, fingerprint=self._message_cache.fingerprint(inputs["messages"])
            )
            intermediate["message_transform"] = batch.append
        return target.translate(
            lambda: target.decoder.prepare(intermediate, client),
            SemanticLossContext(input_format, profile.protocol, "input", "request"),
            draft_key="request",
            enable_message_cache=enable_message_cache,
            message_cache_batch=batch,
        )

    def upstream(self, client: Any, request: dict, profile: ResolvedProfile):
        """Return an async context manager yielding upstream SDK events.

        Args:
            client: Client created by the selected protocol's transport.
            request: Prepared native SDK call arguments.
            profile: Resolved profile selecting the upstream protocol.

        Exiting the context closes response resources; the caller owns the client.
        """
        return self.transport(profile.protocol).upstream(client, request)

    def result(
        self,
        messages: list,
        *,
        target_format: Protocol,
        origin_source="output",
        input_format: Protocol | None = None,
        semantic_profile: ResolvedProfile | None = None,
    ):
        """Render intermediate messages as one completed target-protocol result.

        Returns a message dictionary, or an output items list for openai_responses.
        For target_format="middleware", returns a typed AssistantMessage and requires
        input_format identifying the native source for the loss-policy context.
        Use decode when a list of converted messages is required instead.
        """
        if target_format == "middleware":
            if input_format is None:
                raise ConfigurationError("middleware results require the source input_format")
            source = self._codec(input_format)

            def render():
                """Validate canonical result structure and apply output capability filtering."""
                values = wire_values(messages)
                validate_messages(values)
                return messages_from_validated_wire(output_messages(values, semantic_profile))[0]

            return source.translate(
                render,
                SemanticLossContext(input_format, target_format, origin_source, "result"),
            )
        target = self._codec(target_format)
        return target.translate(
            lambda: target.decoder.result(
                target.decoder.messages(messages, semantic_profile=semantic_profile)
            ),
            SemanticLossContext(input_format, target_format, origin_source, "result"),
        )

    def encode(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        input_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> ProtocolMiddleWare:
        """Convert selected native values to the fixed, typed middleware models.

        Args:
            messages: Standard message list, or None to leave it out of the result.
            tools: Standard tool declarations, or None to leave them out.
            input_format: Registered source protocol name.
            profile: Optional source-profile overrides; None uses standard fields.
            chunks: Standard event list, or None to leave it out.
            **parameters: Additional conversion options supported by the source codec.

        Returns:
            A ProtocolMiddleWare with typed messages, tools and stream collections.
            Unspecified collections are None; supplied empty collections stay empty.

        Typed messages/tools can also be passed to MyLLM.generate with explicit
        stdio_protocol="middleware".
        Invalid recognized values raise MessageValidationError;
        unregistered protocols raise UnsupportedProviderError.
        A missing optional SDK for a built-in protocol raises ModuleNotFoundError.
        """
        codec = self._codec(input_format)
        return ProtocolMiddleWare(
            messages=codec.encode_messages(messages, profile, **parameters)
            if messages is not None
            else None,
            tools=codec.encode_tools(tools, profile, **parameters) if tools is not None else None,
            stream=codec.encode_chunks(chunks, profile, **parameters)
            if chunks is not None
            else None,
        )

    def decode(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        target_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Render selected intermediate values in a standard target protocol.

        Args:
            messages: Intermediate messages, or None to leave them out.
            tools: Intermediate tool declarations, or None to leave them out.
            target_format: Registered target protocol name.
                "middleware" returns typed values without invoking a decoder;
                supply input_format in parameters to identify the native source.
            profile: Optional target-profile overrides; None uses standard fields.
            chunks: Intermediate events, or a complete encoded offline event batch.
            **parameters: Target conversion options, such as model or sequence_number.

        Returns:
            A dictionary containing lists under supplied messages, tools and chunks
            keys. Unsupported fields are not added to the target format.

        Use convert for direct standard-to-standard conversion. Live asynchronous
        consumption is available through MyLLM.generate(stream=True).
        """
        chunks = wire_values(chunks) if chunks is not None else None
        if chunks and any(is_batch(chunk) for chunk in chunks):
            if not all(is_batch(chunk) for chunk in chunks):
                raise MessageValidationError("cannot mix middleware batches and public events")
            validate_operations(chunks)
            # Complete offline streams use the same request-owned accumulation
            # as live calls, regardless of their source and target protocols.
            from my_llm.runtime.state import StreamState

            state = StreamState(parameters.get("model", profile.model if profile else ""))
            converted_chunks = state.feed(chunks)
            converted_chunks.extend(state.finish())
            chunks = converted_chunks
            parameters = {**state.metadata, **parameters}
        origin_source = parameters.pop("origin_source", "input")
        input_format = parameters.pop("input_format", parameters.get("source_protocol"))

        if target_format == "middleware":
            if input_format is None:
                raise ConfigurationError("middleware output requires the source input_format")
            semantic_profile = parameters.pop("semantic_profile", None)

            def render_middleware():
                """Validate and project selected middleware outputs with source capability rules."""
                result = {}
                if messages is not None:
                    values = wire_values(messages)
                    validate_messages(values)
                    result["messages"] = messages_from_validated_wire(
                        output_messages(values, semantic_profile)
                    )
                if tools is not None:
                    result["tools"] = tools_from_wire(wire_values(tools))
                if chunks is not None:
                    validate_events(chunks)
                    result["chunks"] = [
                        MessageEvent.from_validated_wire(event)
                        for event in output_chunks(chunks, semantic_profile)
                    ]
                return result

            return self._codec(input_format).translate(
                render_middleware,
                SemanticLossContext(input_format, target_format, origin_source, "middleware"),
                draft_key="chunks"
                if chunks is not None and messages is None and tools is None
                else None,
            )

        codec = self._codec(target_format)

        def render():
            """Delegate each requested output collection to its target-protocol decoder."""
            result = {}
            for kind, values in (("messages", messages), ("tools", tools), ("chunks", chunks)):
                if values is not None:
                    result[kind] = getattr(codec, f"decode_{kind}")(values, profile, **parameters)
            return result

        return codec.translate(
            render,
            SemanticLossContext(input_format, target_format, origin_source, "decoder"),
            draft_key="chunks"
            if chunks is not None and messages is None and tools is None
            else None,
        )

    def convert(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        input_format: Protocol,
        target_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Convert standard messages, tools or a complete event list in one call.

        Args:
            messages: Source-protocol history, or None.
            tools: Source-protocol tool declarations, or None.
            input_format: Registered source protocol name.
            target_format: Registered target protocol name.
            profile: Optional target-profile overrides.
            chunks: Complete source-protocol event list for offline stream conversion.
            **parameters: Options forwarded to target conversion.

        Returns:
            Dictionary of converted lists, containing only the supplied messages,
            tools and chunks keys. Source values are not modified. Unknown fields
            are excluded; cross-protocol conversion is not guaranteed lossless.

        Use MessageStream for live events rather than converting successive fragments
        with this method.
        """
        intermediate = self.encode(messages, tools, input_format=input_format, chunks=chunks)
        return self.decode(
            messages=intermediate.messages,
            tools=intermediate.tools,
            chunks=intermediate["chunks"] if intermediate.stream is not None else None,
            target_format=target_format,
            profile=profile,
            input_format=input_format,
            **parameters,
        )

    def request(
        self, intermediate: ProtocolMiddleWare | dict, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        """Build a native SDK request from encoded messages and optional tools.

        Args:
            intermediate: Dictionary with messages and optional tools from encode.
            profile: Resolved upstream settings.
            **parameters: Generation options accepted by the target decoder.

        Returns:
            Validated SDK keyword arguments without creating clients or sending I/O.

        Use prepare to supply a complete unencoded generation call.
        """
        if isinstance(intermediate, ProtocolMiddleWare):
            messages, tools = intermediate.messages or [], intermediate.tools
        else:
            messages, tools = intermediate["messages"], intermediate.get("tools")
        source = messages[0].get("protocol") if messages else None
        target = self._codec(profile.protocol)
        return target.translate(
            lambda: target.request(messages, tools, profile, **parameters),
            SemanticLossContext(source, profile.protocol, "input", "request"),
        )


for _protocol in BUILTIN_CODECS:
    if sdk_available(_protocol):
        MyCodec.register(builtin_codec(_protocol))
del _protocol
