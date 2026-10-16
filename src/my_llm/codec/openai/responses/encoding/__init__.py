"""Stateless openai/responses encoding."""

from my_llm.codec.base import BaseEncoder
from my_llm.codec.openai.responses.encoding import chunks, messages, tools


class ResponsesEncoder(BaseEncoder):
    __slots__ = ()
    messages = staticmethod(messages.encode)
    tools = staticmethod(tools.encode)
    chunks = staticmethod(chunks.encode)


__all__ = ["ResponsesEncoder"]
