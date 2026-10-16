"""Stateless ollama protocol facade."""

from typing import Any

from my_llm.codec.base import BaseCodec
from my_llm.codec.ollama.decoding import OllamaDecoder
from my_llm.codec.ollama.encoding import OllamaEncoder
from my_llm.codec.utils.matching import candidates


class OllamaCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "ollama"
    encoder = OllamaEncoder()
    decoder = OllamaDecoder()

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        return (
            self.protocol_or_api_style
            if self.protocol_or_api_style in candidates(messages_or_tools)
            else None
        )
