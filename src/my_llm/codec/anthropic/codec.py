"""Stateless anthropic protocol facade."""

from typing import Any

from my_llm.codec.anthropic.decoding import AnthropicDecoder
from my_llm.codec.anthropic.encoding import AnthropicEncoder
from my_llm.codec.base import BaseCodec
from my_llm.codec.utils.matching import candidates


class AnthropicCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "anthropic"
    encoder = AnthropicEncoder()
    decoder = AnthropicDecoder()

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        return (
            self.protocol_or_api_style
            if self.protocol_or_api_style in candidates(messages_or_tools)
            else None
        )
