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
        replay_reasoning_content_when_tool_calls=Config(default_value=True),
        max_tokens=Parameter(path="$.max_completion_tokens", default_value=omit),
        content=Parameter(path="$.choices[0].delta.content", default_value=omit),
        reasoning_content=Parameter(
            path="$.choices[0].delta.reasoning_content",
            default_value=omit,
        ),
        reasoning_enable=Parameter(
            path="$.reasoning_effort",
            default_value="medium",
        ),
        reasoning_effort=Parameter(supports=False, path="$.reasoning_effort", default_value=omit),
    )
    encoder = ChatCompletionsEncoder()
    decoder = ChatCompletionsDecoder()
    transport = ChatCompletionsTransport()
