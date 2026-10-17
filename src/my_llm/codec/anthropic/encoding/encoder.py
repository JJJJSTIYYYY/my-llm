"""Stateless anthropic encoding."""

from my_llm.codec.anthropic.encoding import matching
from my_llm.codec.anthropic.encoding.messages import conversion as messages
from my_llm.codec.anthropic.encoding.stream import chunks, usage
from my_llm.codec.anthropic.encoding.tools import definitions as tools
from my_llm.codec.base.encoder import BaseEncoder


class AnthropicEncoder(BaseEncoder):
    __slots__ = ()
    matches = staticmethod(matching.matches)
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)
    usage = staticmethod(usage.encode)
