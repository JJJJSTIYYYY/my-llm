"""Stateless openai_responses protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.codec.openai.responses.transport import ResponsesTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class ResponsesCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_responses"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(
            name="replay_reasoning_content_when_tool_calls", default_value=True
        ),
        supports_reasoning_effort=Config(name="supports_reasoning_effort", default_value=True),
        max_tokens=Parameter(
            name="max_tokens", path="$.max_output_tokens", default_value=omit, phase="decode"
        ),
        content=Parameter(name="content", path="$.delta", default_value=omit, phase="encode"),
        reasoning_content=Parameter(
            name="reasoning_content", path="$.delta", default_value=omit, phase="encode"
        ),
        reasoning_enable=Parameter(
            name="reasoning_enable",
            path="$.reasoning",
            default_value={"effort": "medium", "summary": "detailed"},
            phase="decode",
        ),
        reasoning_effort=Parameter(
            name="reasoning_effort", path="$.reasoning.effort", default_value=omit, phase="decode"
        ),
    )
    encoder = ResponsesEncoder()
    decoder = ResponsesDecoder()
    transport = ResponsesTransport()
