"""Stateless ollama decoding."""

from my_llm.codec.base import BaseDecoder
from my_llm.codec.ollama.decoding import chunks, messages, tools


class OllamaDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)


__all__ = ["OllamaDecoder"]
