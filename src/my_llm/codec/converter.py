"""Reusable, profile-independent conversion among registered protocols."""

from typing import Any

from my_llm._optional import BUILTIN_CODECS, builtin_codec, require_sdk, sdk_available
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.conversion import dictionary_views
from my_llm.codec.shared.stream import is_batch
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedProviderError
from my_llm.profile.configuration import Protocol, ProtocolDefaults
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit
from my_llm.registry import register_codec, registered_codecs


class MyCodec:
    """Convert messages, tools and complete event lists between standard protocols.

    Construction takes no arguments and captures the currently registered codecs
    in the codecs mapping. Register custom protocols before creating instances.
    Calls do not retain profiles, inferred formats or stream fragments.
    Use convert for direct conversion; profile=None selects standard field behavior.
    Only recognized fields are translated; conversion is not guaranteed lossless.
    """

    __slots__ = ("codecs",)

    def __init__(self) -> None:
        """Capture the currently registered protocol implementations."""
        self.codecs = registered_codecs()

    @staticmethod
    def register(codec: BaseCodec) -> None:
        """Register a protocol for subsequently constructed MyCodec/MyLLM instances.

        Args:
            codec: BaseCodec with ProtocolDefaults and matching encoder/decoder protocol
                names. The codec and its components must declare empty __slots__.

        Raises:
            ConfigurationError: Components are invalid or the protocol name is already
                registered. Existing instances retain their registration snapshot.
        """
        if not isinstance(codec, BaseCodec):
            raise ConfigurationError("protocol implementation must be a BaseCodec")
        protocol = codec.protocol_or_api_style
        if not isinstance(protocol, str) or not protocol.strip():
            raise ConfigurationError("protocol name must be a non-empty string")
        if not isinstance(codec.defaults, ProtocolDefaults):
            raise ConfigurationError("protocol defaults must be ProtocolDefaults")
        components = (
            (codec.encoder, BaseEncoder),
            (codec.decoder, BaseDecoder),
            (codec.transport, BaseTransport),
        )
        for component, expected in components:
            if not isinstance(component, expected):
                raise ConfigurationError(f"protocol component must be a {expected.__name__}")
        if codec.encoder.protocol != protocol or codec.decoder.protocol != protocol:
            raise ConfigurationError("codec, encoder and decoder must use the same protocol name")
        if any(
            hasattr(component, "__dict__")
            or any(base.__dict__.get("__slots__", ()) for base in type(component).__mro__)
            for component in (codec, *(component for component, _ in components))
        ):
            raise ConfigurationError("protocol components must declare empty __slots__")
        register_codec(codec)

    def _codec(self, protocol: str) -> BaseCodec:
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
        """Return the stateless intermediate-to-native converter."""
        return self._codec(protocol).decoder

    def transport(self, protocol: str) -> BaseTransport:
        """Return the SDK execution component composed by this protocol's codec."""
        return self._codec(protocol).transport

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
        """
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
        self, inputs: dict, profile: ResolvedProfile, *, input_format: Protocol, client: dict
    ) -> dict:
        """Prepare one complete generation call without performing network I/O.

        Args:
            inputs: MyLLM.generate options as a dictionary, including messages, tools
                and stream. Values use input_format.
            profile: Resolved upstream provider/model settings.
            input_format: Registered protocol used by the supplied history and tools.
            client: Connection settings from client_parameters(profile).

        Returns:
            Dictionary with client (SDK constructor arguments), request (SDK call
            arguments), and local (stream flag and client_key) entries.

        Raises:
            ConfigurationError: Generation options or target request are invalid.
            MessageValidationError: Messages, tools or their associations are invalid.
            UnsupportedCapabilityError: Required content cannot be represented.
        """
        source = self._codec(input_format)
        intermediate = source.encoder.request(inputs, profile)
        return self.decoder(profile.protocol).prepare(intermediate, client)

    def upstream(self, client: Any, request: dict, profile: ResolvedProfile):
        """Return an async context manager yielding upstream SDK events.

        Args:
            client: Client created by the selected protocol's transport.
            request: Prepared native SDK call arguments.
            profile: Resolved profile selecting the upstream protocol.

        Exiting the context closes response resources; the caller owns the client.
        """
        return self.transport(profile.protocol).upstream(client, request)

    def result(self, messages: list, *, target_format: Protocol):
        """Render intermediate messages as one completed target-protocol result.

        Returns a message dictionary, or an output items list for openai_responses.
        Use decode when a list of converted messages is required instead.
        """
        converted = self.decode(messages=messages, target_format=target_format)["messages"]
        return self.decoder(target_format).result(converted)

    def encode(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        input_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Convert selected standard-protocol values to intermediate dictionaries.

        Args:
            messages: Standard message list, or None to leave it out of the result.
            tools: Standard tool declarations, or None to leave them out.
            input_format: Registered source protocol name.
            profile: Optional source-profile overrides; None uses standard fields.
            chunks: Standard event list, or None to leave it out.
            **parameters: Additional conversion options supported by the source codec.

        Returns:
            A dictionary containing only supplied messages, tools and chunks entries;
            each value is a list. Empty supplied lists remain present.

        Intermediate values are for conversion APIs, not application history.
        Invalid recognized values raise MessageValidationError or
        UnsupportedCapabilityError; unregistered protocols raise UnsupportedProviderError.
        A missing optional SDK for a built-in protocol raises ModuleNotFoundError.
        """
        codec = self._codec(input_format)
        result = {}
        for kind, values in (("messages", messages), ("tools", tools), ("chunks", chunks)):
            if values is not None:
                result[kind] = getattr(codec, f"encode_{kind}")(values, profile, **parameters)
        return result

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
            profile: Optional target-profile overrides; None uses standard fields.
            chunks: Intermediate events, or a complete encoded offline event batch.
            **parameters: Target conversion options, such as model or sequence_number.

        Returns:
            A dictionary containing lists under supplied messages, tools and chunks
            keys. Unsupported fields are not added to the target format.

        Use convert for direct standard-to-standard conversion. Live asynchronous
        consumption is available through MyLLM.generate(stream=True).
        """
        codec = self._codec(target_format)
        if chunks and all(is_batch(chunk) for chunk in chunks):
            # Complete offline streams use the same request-owned accumulation
            # as live calls, regardless of their source and target protocols.
            from my_llm.stream._state import StreamState

            state = StreamState(parameters.get("model", profile.model if profile else ""))
            converted_chunks = state.feed(chunks)
            converted_chunks.extend(state.finish())
            chunks = converted_chunks
            parameters = {**state.metadata, **parameters}
        result = {}
        for kind, values in (("messages", messages), ("tools", tools), ("chunks", chunks)):
            if values is not None:
                result[kind] = getattr(codec, f"decode_{kind}")(values, profile, **parameters)
        return result

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
            **intermediate, target_format=target_format, profile=profile, **parameters
        )

    def request(self, intermediate: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Build a native SDK request from encoded messages and optional tools.

        Args:
            intermediate: Dictionary with messages and optional tools from encode.
            profile: Resolved upstream settings.
            **parameters: Generation options accepted by the target decoder.

        Returns:
            Validated SDK keyword arguments without creating clients or sending I/O.

        Use prepare to supply a complete unencoded generation call.
        """
        return self._codec(profile.protocol).request(
            intermediate["messages"], intermediate.get("tools"), profile, **parameters
        )


for _protocol in BUILTIN_CODECS:
    if sdk_available(_protocol):
        MyCodec.register(builtin_codec(_protocol)())
del _protocol
