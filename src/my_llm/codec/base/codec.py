"""Symmetric stateless protocol conversion facade."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.base.transport import BaseTransport
from my_llm.profile.configuration import Protocol, ProtocolDefaults
from my_llm.profile.resolution import ResolvedProfile


class BaseCodec(ABC):
    """Abstract protocol extension with encoder, decoder, transport and defaults.

    Implement the abstract properties and declare empty __slots__, then register
    an instance with MyCodec.register. Public conversion methods accept lists and
    an optional ResolvedProfile; None selects standard protocol behavior.
    Encoding returns intermediate dictionaries; decoding returns native values.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def protocol_or_api_style(self) -> Protocol:
        """Native identity used only for routing and provenance."""
        ...

    @property
    @abstractmethod
    def defaults(self) -> ProtocolDefaults:
        """Standard profile settings owned and declared by this protocol."""
        ...

    @property
    @abstractmethod
    def encoder(self) -> BaseEncoder:
        """The native-to-intermediate implementation; required for every codec."""
        ...

    @property
    @abstractmethod
    def decoder(self) -> BaseDecoder:
        """The intermediate-to-native implementation; required for every codec."""
        ...

    @property
    @abstractmethod
    def transport(self) -> BaseTransport:
        """Official SDK lifecycle and execution, independent of conversion."""
        ...

    @final
    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate construction-time protocol settings through Decoder."""
        self.decoder.validate_profile(profile)

    @final
    def validate_input(self, messages: list, tools: list | None = None) -> None:
        """Validate the bound input protocol through Encoder."""
        self.encoder.validate_input(messages, tools)

    @final
    def request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        """Return validated SDK call arguments from intermediate messages and tools.

        profile selects upstream settings; parameters supplies generation controls.
        This method performs no network I/O.
        """
        self.validate_profile(profile)
        return self.decoder.request(messages, tools, profile, **parameters)

    @final
    def encode_messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return intermediate messages from native values using optional profile overrides."""
        return self.encoder.messages(messages, profile, **parameters)

    @final
    def decode_messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return native message dictionaries from intermediate messages."""
        return self.decoder.messages(messages, profile, **parameters)

    @final
    def encode_tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return intermediate tool declarations from native tool values."""
        return self.encoder.tools(tools, profile, **parameters)

    @final
    def decode_tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return native tool declarations from intermediate tools."""
        return self.decoder.tools(tools, profile, **parameters)

    @final
    def encode_chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return intermediate events from the supplied native event list."""
        return self.encoder.chunks(chunks, profile, **parameters)

    @final
    def decode_chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return native event dictionaries from the supplied intermediate events."""
        return self.decoder.chunks(chunks, profile, **parameters)

    @final
    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        """Return this protocol when the entire input conforms, otherwise None."""
        return self.protocol_or_api_style if self.encoder.matches(messages_or_tools) else None
