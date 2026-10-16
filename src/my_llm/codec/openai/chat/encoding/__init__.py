"""Stateless openai/chat encoding."""

from my_llm.codec.base import BaseEncoder
from my_llm.codec.openai.chat.encoding import chunks, messages, tools


class ChatCompletionsEncoder(BaseEncoder):
    __slots__ = ()
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)


__all__ = ["ChatCompletionsEncoder"]
