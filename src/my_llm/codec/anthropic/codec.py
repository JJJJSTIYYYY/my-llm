"""Stateless anthropic protocol facade."""

from my_llm.codec.anthropic.decoding.decoder import AnthropicDecoder
from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
from my_llm.codec.anthropic.transport import AnthropicTransport
from my_llm.codec.base.codec import BaseCodec
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class AnthropicCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "anthropic"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(
            name="replay_reasoning_content_when_tool_calls", default_value=True
        ),
        supports_reasoning_effort=Config(name="supports_reasoning_effort", default_value=False),
        max_tokens=Parameter(
            name="max_tokens", path="$.max_tokens", default_value=omit, phase="decode"
        ),
        content=Parameter(name="content", path="$.delta.text", default_value=omit, phase="encode"),
        reasoning_content=Parameter(
            name="reasoning_content", path="$.delta.thinking", default_value=omit, phase="encode"
        ),
        reasoning_enable=Parameter(
            name="reasoning_enable",
            path="$.thinking",
            default_value={"type": "enabled", "budget_tokens": 1024},
            phase="decode",
        ),
        reasoning_effort=Parameter(
            name="reasoning_effort",
            path="$.output_config.effort",
            default_value=omit,
            phase="decode",
        ),
    )
    encoder = AnthropicEncoder()
    decoder = AnthropicDecoder()
    transport = AnthropicTransport()
