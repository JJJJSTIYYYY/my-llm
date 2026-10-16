"""Stateless anthropic encoding."""

from my_llm.codec.anthropic.encoding import chunks, messages, tools
from my_llm.codec.base import BaseEncoder


class AnthropicEncoder(BaseEncoder):
    __slots__ = ()
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)


__all__ = ["AnthropicEncoder"]
