"""Stateless openai_responses protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.codec.openai.responses.transport import ResponsesTransport


class ResponsesCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_responses"
    encoder = ResponsesEncoder()
    decoder = ResponsesDecoder()
    transport = ResponsesTransport()
