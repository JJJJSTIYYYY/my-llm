"""Stateless openai_chat protocol facade."""

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from my_llm.profile.configuration import ProtocolDefaults


class ChatCompletionsCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_chat"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=False,
        supports_reasoning_effort=False,
        max_tokens_path="$.max_completion_tokens",
        content_path="$.choices[0].delta.content",
        reasoning_content_path="$.choices[0].delta.reasoning_content",
        reasoning_enable_parameter={"reasoning_effort": "medium"},
        reasoning_effort_parameter={"reasoning_effort": "{effort}"},
    )
    encoder = ChatCompletionsEncoder()
    decoder = ChatCompletionsDecoder()
    transport = ChatCompletionsTransport()
