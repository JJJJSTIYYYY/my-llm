"""Stateless gemini encoding."""

from my_llm.codec.base import BaseEncoder
from my_llm.codec.gemini.encoding import chunks, messages, tools


class GeminiEncoder(BaseEncoder):
    __slots__ = ()
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)


__all__ = ["GeminiEncoder"]
