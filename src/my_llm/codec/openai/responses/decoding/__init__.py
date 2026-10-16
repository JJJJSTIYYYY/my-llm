"""Stateless openai/responses decoding."""

from my_llm.codec.base import BaseDecoder
from my_llm.codec.openai.responses.decoding import chunks, messages, tools


class ResponsesDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)


__all__ = ["ResponsesDecoder"]
