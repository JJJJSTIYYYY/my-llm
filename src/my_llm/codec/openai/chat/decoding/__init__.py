"""Stateless openai/chat decoding."""

from my_llm.codec.base import BaseDecoder
from my_llm.codec.openai.chat.decoding import chunks, messages, tools


class ChatCompletionsDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)


__all__ = ["ChatCompletionsDecoder"]
