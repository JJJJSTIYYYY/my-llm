"""Stateless gemini decoding."""

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.gemini.decoding.messages import conversion as messages
from my_llm.codec.gemini.decoding.request import assembly as request
from my_llm.codec.gemini.decoding.request import settings, transport
from my_llm.codec.gemini.decoding.stream import chunks, usage
from my_llm.codec.gemini.decoding.tools import definitions as tools


class GeminiDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)
    usage = staticmethod(usage.decode)
    request = staticmethod(request.build)
    upstream = staticmethod(transport.open_stream)
    profile_defaults = staticmethod(settings.defaults)
    usage_modes = settings.USAGE_MODES
