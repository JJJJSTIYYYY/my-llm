"""Symmetric stateless protocol conversion facade."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.profile.configuration import Protocol
from my_llm.profile.resolution import ResolvedProfile


class BaseCodec(ABC):
    """A protocol and its stateless encoder/decoder; profiles are call arguments."""

    __slots__ = ()

    @property
    @abstractmethod
    def protocol_or_api_style(self) -> Protocol:
        """Native identity used only for routing and provenance."""
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

    @final
    def encode_messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        return self.encoder.messages(messages, profile, **parameters)

    @final
    def decode_messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        return self.decoder.messages(messages, profile, **parameters)

    @final
    def encode_tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        return self.encoder.tools(tools, profile, **parameters)

    @final
    def decode_tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        return self.decoder.tools(tools, profile, **parameters)

    @final
    def encode_chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        return self.encoder.chunks(chunks, profile, **parameters)

    @final
    def decode_chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        return self.decoder.chunks(chunks, profile, **parameters)

    @final
    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        """Return this protocol when the entire input conforms, otherwise None."""
        return self.protocol_or_api_style if self.encoder.matches(messages_or_tools) else None
