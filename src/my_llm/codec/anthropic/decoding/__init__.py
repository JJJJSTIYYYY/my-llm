"""Stateless anthropic decoding."""

from my_llm.codec.anthropic.decoding import chunks, messages, tools
from my_llm.codec.base import BaseDecoder


class AnthropicDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)


__all__ = ["AnthropicDecoder"]
