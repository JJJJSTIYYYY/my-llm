"""Stateless openai/responses decoding."""

from openai import Omit

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.responses.decoding.messages import conversion as messages
from my_llm.codec.openai.responses.decoding.request import assembly as request
from my_llm.codec.openai.responses.decoding.request import settings, transport
from my_llm.codec.openai.responses.decoding.stream import chunks, usage
from my_llm.codec.openai.responses.decoding.tools import definitions as tools


class ResponsesDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)
    usage = staticmethod(usage.decode)
    request = staticmethod(request.build)
    upstream = staticmethod(transport.open_stream)
    profile_defaults = staticmethod(settings.defaults)
    usage_modes = settings.USAGE_MODES

    def result(self, messages: list) -> list:
        return messages

    def allows_external_calls(self, previous_response_id) -> bool:
        return previous_response_id is not None and not isinstance(previous_response_id, Omit)
