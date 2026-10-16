"""Stateless gemini decoding."""

from my_llm.codec.base import BaseDecoder
from my_llm.codec.gemini.decoding import chunks, messages, tools


class GeminiDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)


__all__ = ["GeminiDecoder"]
