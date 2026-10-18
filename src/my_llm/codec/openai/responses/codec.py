"""Stateless openai_responses protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.codec.openai.responses.transport import ResponsesTransport
from my_llm.profile.configuration import ProtocolDefaults


class ResponsesCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_responses"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=True,
        supports_reasoning_effort=True,
        max_tokens_path="$.max_output_tokens",
        content_path="response.output_text.delta",
        reasoning_content_path="response.reasoning_summary_text.delta",
        reasoning_enable_parameter={"reasoning": {"effort": "medium", "summary": "detailed"}},
        reasoning_effort_parameter={"reasoning": {"effort": "{effort}", "summary": "detailed"}},
    )
    encoder = ResponsesEncoder()
    decoder = ResponsesDecoder()
    transport = ResponsesTransport()
