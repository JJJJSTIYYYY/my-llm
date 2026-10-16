"""Stateless openai_responses protocol facade."""

from typing import Any

from my_llm.codec.base import BaseCodec
from my_llm.codec.openai.responses.decoding import ResponsesDecoder
from my_llm.codec.openai.responses.encoding import ResponsesEncoder
from my_llm.codec.utils.matching import candidates


class ResponsesCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_responses"
    encoder = ResponsesEncoder()
    decoder = ResponsesDecoder()

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        return (
            self.protocol_or_api_style
            if self.protocol_or_api_style in candidates(messages_or_tools)
            else None
        )
