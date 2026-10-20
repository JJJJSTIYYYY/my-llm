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
        replay_reasoning_content_when_tool_calls=Config(
            name="replay_reasoning_content_when_tool_calls", default_value=True
        ),
        supports_reasoning_effort=Config(name="supports_reasoning_effort", default_value=False),
        max_tokens=Parameter(
            name="max_tokens", path="$.options.num_predict", default_value=omit, phase="decode"
        ),
        content=Parameter(
            name="content", path="$.message.content", default_value=omit, phase="encode"
        ),
        reasoning_content=Parameter(
            name="reasoning_content", path="$.message.thinking", default_value=omit, phase="encode"
        ),
        reasoning_enable=Parameter(
            name="reasoning_enable", path="$.think", default_value=True, phase="decode"
        ),
        reasoning_effort=Parameter(
            name="reasoning_effort", path="$.think", default_value=omit, phase="decode"
        ),
    )
    encoder = OllamaEncoder()
    decoder = OllamaDecoder()
    transport = OllamaTransport()
