"""Stateless openai_chat protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class ChatCompletionsCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_chat"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(
            name="replay_reasoning_content_when_tool_calls", default_value=False
        ),
        supports_reasoning_effort=Config(name="supports_reasoning_effort", default_value=False),
        max_tokens=Parameter(
            name="max_tokens", path="$.max_completion_tokens", default_value=omit, phase="decode"
        ),
        content=Parameter(
            name="content", path="$.choices[0].delta.content", default_value=omit, phase="encode"
        ),
        reasoning_content=Parameter(
            name="reasoning_content",
            path="$.choices[0].delta.reasoning_content",
            default_value=omit,
            phase="encode",
        ),
        reasoning_enable=Parameter(
            name="reasoning_enable",
            path="$.reasoning_effort",
            default_value="medium",
            phase="decode",
        ),
        reasoning_effort=Parameter(
            name="reasoning_effort", path="$.reasoning_effort", default_value=omit, phase="decode"
        ),
    )
    encoder = ChatCompletionsEncoder()
    decoder = ChatCompletionsDecoder()
    transport = ChatCompletionsTransport()
