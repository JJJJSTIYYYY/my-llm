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
        replay_reasoning_content_when_tool_calls=Config(default_value=True),
        max_tokens=Parameter(path="$.max_tokens", default_value=omit),
        content=Parameter(path="$.delta.text", default_value=omit),
        reasoning_content=Parameter(path="$.delta.thinking", default_value=omit),
        reasoning_enable=Parameter(
            path="$.thinking",
            default_value={"type": "enabled", "budget_tokens": 1024},
        ),
        reasoning_effort=Parameter(
            supports=False,
            path="$.output_config.effort",
            default_value=omit,
        ),
    )
    encoder = AnthropicEncoder()
    decoder = AnthropicDecoder()
    transport = AnthropicTransport()
