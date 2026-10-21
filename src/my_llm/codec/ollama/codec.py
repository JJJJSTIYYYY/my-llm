"""Stateless ollama protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.ollama.decoding.decoder import OllamaDecoder
from my_llm.codec.ollama.encoding.encoder import OllamaEncoder
from my_llm.codec.ollama.transport import OllamaTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class OllamaCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "ollama"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(default_value=True),
        api_key=Parameter(
            path="$.extra_headers.Authorization",
            default_value=omit,
        ),
        max_tokens=Parameter(path="$.options.num_predict", default_value=omit),
        content=Parameter(path="$.message.content", default_value=omit),
        reasoning_content=Parameter(path="$.message.thinking", default_value=omit),
        reasoning_enable=Parameter(path="$.think", default_value=True),
        reasoning_effort=Parameter(supports=False, path="$.think", default_value=omit),
    )
    encoder = OllamaEncoder()
    decoder = OllamaDecoder()
    transport = OllamaTransport()
