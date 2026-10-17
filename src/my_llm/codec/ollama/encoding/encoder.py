"""Stateless ollama encoding."""

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.ollama.encoding import matching
from my_llm.codec.ollama.encoding.messages import conversion as messages
from my_llm.codec.ollama.encoding.stream import chunks, usage
from my_llm.codec.ollama.encoding.tools import definitions as tools


class OllamaEncoder(BaseEncoder):
    __slots__ = ()
    matches = staticmethod(matching.matches)
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)
    usage = staticmethod(usage.encode)
