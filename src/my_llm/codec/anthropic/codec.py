"""Stateless anthropic protocol facade."""

from my_llm.codec.anthropic.decoding.decoder import AnthropicDecoder
from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
from my_llm.codec.base.codec import BaseCodec


class AnthropicCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "anthropic"
    encoder = AnthropicEncoder()
    decoder = AnthropicDecoder()
