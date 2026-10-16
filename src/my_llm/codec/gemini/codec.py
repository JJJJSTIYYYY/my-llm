"""Stateless gemini protocol facade."""

from typing import Any

from my_llm.codec.base import BaseCodec
from my_llm.codec.gemini.decoding import GeminiDecoder
from my_llm.codec.gemini.encoding import GeminiEncoder
from my_llm.codec.utils.matching import candidates


class GeminiCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "gemini"
    encoder = GeminiEncoder()
    decoder = GeminiDecoder()

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        return (
            self.protocol_or_api_style
            if self.protocol_or_api_style in candidates(messages_or_tools)
            else None
        )
