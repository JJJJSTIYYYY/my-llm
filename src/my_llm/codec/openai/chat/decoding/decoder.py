"""Stateless openai/chat decoding."""

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.chat.decoding.messages import conversion as messages
from my_llm.codec.openai.chat.decoding.request import assembly as request
from my_llm.codec.openai.chat.decoding.request import settings, transport
from my_llm.codec.openai.chat.decoding.stream import chunks, usage
from my_llm.codec.openai.chat.decoding.tools import definitions as tools


class ChatCompletionsDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)
    usage = staticmethod(usage.decode)
    request = staticmethod(request.build)
    upstream = staticmethod(transport.open_stream)
    profile_defaults = staticmethod(settings.defaults)
    usage_modes = settings.USAGE_MODES
