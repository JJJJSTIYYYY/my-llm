"""Stateless ollama protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.ollama.decoding.decoder import OllamaDecoder
from my_llm.codec.ollama.encoding.encoder import OllamaEncoder


class OllamaCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "ollama"
    encoder = OllamaEncoder()
    decoder = OllamaDecoder()
