"""Stateless gemini protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.gemini.decoding.decoder import GeminiDecoder
from my_llm.codec.gemini.encoding.encoder import GeminiEncoder
from my_llm.codec.gemini.transport import GeminiTransport
from my_llm.profile.configuration import ProtocolDefaults


class GeminiCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "gemini"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=True,
        supports_reasoning_effort=False,
        max_tokens_path="$.config.max_output_tokens",
        content_path="$.candidates[0].content.parts[0].text",
        reasoning_content_path="$.candidates[0].content.parts[0].text",
        reasoning_enable_parameter={},
        reasoning_effort_parameter={},
    )
    encoder = GeminiEncoder()
    decoder = GeminiDecoder()
    transport = GeminiTransport()
