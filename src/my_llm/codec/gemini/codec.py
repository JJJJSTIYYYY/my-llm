"""Stateless gemini protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.gemini.decoding.decoder import GeminiDecoder
from my_llm.codec.gemini.encoding.encoder import GeminiEncoder
from my_llm.codec.gemini.transport import GeminiTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class GeminiCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "gemini"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(
            name="replay_reasoning_content_when_tool_calls", default_value=True
        ),
        supports_reasoning_effort=Config(name="supports_reasoning_effort", default_value=False),
        max_tokens=Parameter(
            name="max_tokens", path="$.config.max_output_tokens", default_value=omit, phase="decode"
        ),
        content=Parameter(
            name="content",
            path="$.candidates[0].content.parts[0].text",
            default_value=omit,
            phase="encode",
        ),
        reasoning_content=Parameter(
            name="reasoning_content",
            path="$.candidates[0].content.parts[0].text",
            default_value=omit,
            phase="encode",
        ),
        reasoning_enable=Parameter(
            name="reasoning_enable",
            path="$.config.thinking_config",
            default_value={"include_thoughts": True, "thinking_budget": -1},
            phase="decode",
        ),
        reasoning_effort=Parameter(
            name="reasoning_effort",
            path="$.config.thinking_config.thinking_level",
            default_value=omit,
            phase="decode",
        ),
    )
    encoder = GeminiEncoder()
    decoder = GeminiDecoder()
    transport = GeminiTransport()
