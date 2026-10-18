"""Stateless ollama protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.ollama.decoding.decoder import OllamaDecoder
from my_llm.codec.ollama.encoding.encoder import OllamaEncoder
from my_llm.codec.ollama.transport import OllamaTransport
from my_llm.profile.configuration import ProtocolDefaults


class OllamaCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "ollama"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=True,
        supports_reasoning_effort=False,
        max_tokens_path="$.options.num_predict",
        content_path="$.message.content",
        reasoning_content_path="$.message.thinking",
        reasoning_enable_parameter={},
        reasoning_effort_parameter={},
    )
    encoder = OllamaEncoder()
    decoder = OllamaDecoder()
    transport = OllamaTransport()
