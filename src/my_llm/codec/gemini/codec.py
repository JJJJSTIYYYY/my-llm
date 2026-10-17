"""Stateless gemini protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.gemini.decoding.decoder import GeminiDecoder
from my_llm.codec.gemini.encoding.encoder import GeminiEncoder
from my_llm.codec.gemini.transport import GeminiTransport


class GeminiCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "gemini"
    encoder = GeminiEncoder()
    decoder = GeminiDecoder()
    transport = GeminiTransport()
