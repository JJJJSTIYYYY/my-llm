"""Stateless ollama encoding."""

from my_llm.codec.base import BaseEncoder
from my_llm.codec.ollama.encoding import chunks, messages, tools


class OllamaEncoder(BaseEncoder):
    __slots__ = ()
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)


__all__ = ["OllamaEncoder"]
