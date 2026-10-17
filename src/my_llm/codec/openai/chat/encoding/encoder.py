"""Stateless openai/chat encoding."""

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.chat.encoding import matching
from my_llm.codec.openai.chat.encoding.messages import conversion as messages
from my_llm.codec.openai.chat.encoding.stream import chunks, usage
from my_llm.codec.openai.chat.encoding.tools import definitions as tools


class ChatCompletionsEncoder(BaseEncoder):
    __slots__ = ()
    matches = staticmethod(matching.matches)
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)
    usage = staticmethod(usage.encode)
