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
    """Convert recognized standard fields to intermediate dictionaries and back.

    No profile, format guess or stream fragments are retained between calls.
    ``profile=None`` means the standard protocol without provider overrides.
    """

    __slots__ = ("codecs",)

    def __init__(self) -> None:
        self.codecs = registered_codecs()

    @staticmethod
    def register(codec: BaseCodec) -> None:
        """Register a stateless protocol before constructing MyCodec or MyLLM.

        The Codec owns its default configuration and all three implementations.
        Existing converter instances keep their registration snapshot. Names
        cannot replace an already registered protocol.
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
        self._codec(profile.protocol).validate_profile(profile)

    def validate_input(self, protocol: str, messages: list, tools: list | None = None) -> None:
        """Validate declared input through its codec's transport."""
        self._codec(protocol).validate_input(messages, tools)

    def client_parameters(self, profile: ResolvedProfile) -> dict:
        return self.decoder(profile.protocol).client_parameters(profile)

    def bind_input(self, inputs: dict, protocol: Protocol | None) -> Protocol:
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
        source = self._codec(input_format)
        intermediate = source.encoder.request(inputs, profile)
        return self.decoder(profile.protocol).prepare(intermediate, client)

    def upstream(self, client: Any, request: dict, profile: ResolvedProfile):
        return self.transport(profile.protocol).upstream(client, request)

    def result(self, messages: list, *, target_format: Protocol):
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
        """Convert standard input directly to a target protocol."""
        intermediate = self.encode(messages, tools, input_format=input_format, chunks=chunks)
        return self.decode(
            **intermediate, target_format=target_format, profile=profile, **parameters
        )

    def request(self, intermediate: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        return self._codec(profile.protocol).request(
            intermediate["messages"], intermediate.get("tools"), profile, **parameters
        )


for _protocol in BUILTIN_CODECS:
    if sdk_available(_protocol):
        MyCodec.register(builtin_codec(_protocol)())
del _protocol
