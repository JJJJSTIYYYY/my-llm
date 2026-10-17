"""Stateless openai_chat protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder


class ChatCompletionsCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_chat"
    encoder = ChatCompletionsEncoder()
    decoder = ChatCompletionsDecoder()
