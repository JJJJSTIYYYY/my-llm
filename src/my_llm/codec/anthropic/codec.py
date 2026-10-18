"""Stateless anthropic protocol facade."""

from my_llm.codec.anthropic.decoding.decoder import AnthropicDecoder
from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
from my_llm.codec.anthropic.transport import AnthropicTransport
from my_llm.codec.base.codec import BaseCodec
from my_llm.profile.configuration import ProtocolDefaults


class AnthropicCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "anthropic"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=True,
        supports_reasoning_effort=False,
        max_tokens_path="$.max_tokens",
        content_path="$.delta.text",
        reasoning_content_path="$.delta.thinking",
        reasoning_enable_parameter={},
        reasoning_effort_parameter={},
    )
    encoder = AnthropicEncoder()
    decoder = AnthropicDecoder()
    transport = AnthropicTransport()
