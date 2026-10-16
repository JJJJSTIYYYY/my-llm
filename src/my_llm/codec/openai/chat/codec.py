"""Stateless openai_chat protocol facade."""

from typing import Any

from my_llm.codec.base import BaseCodec
from my_llm.codec.openai.chat.decoding import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding import ChatCompletionsEncoder
from my_llm.codec.utils.matching import candidates


class ChatCompletionsCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_chat"
    encoder = ChatCompletionsEncoder()
    decoder = ChatCompletionsDecoder()

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        return (
            self.protocol_or_api_style
            if self.protocol_or_api_style in candidates(messages_or_tools)
            else None
        )
