"""Stateless ollama decoding."""

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.ollama.decoding.messages import conversion as messages
from my_llm.codec.ollama.decoding.request import assembly as request
from my_llm.codec.ollama.decoding.request import settings, transport
from my_llm.codec.ollama.decoding.stream import chunks, usage
from my_llm.codec.ollama.decoding.tools import definitions as tools
from my_llm.exceptions import UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile


class OllamaDecoder(BaseDecoder):
    __slots__ = ()
    messages = staticmethod(messages.decode)
    tools = staticmethod(tools.decode)
    chunks = staticmethod(chunks.decode)
    usage = staticmethod(usage.decode)
    request = staticmethod(request.build)
    upstream = staticmethod(transport.open_stream)
    profile_defaults = staticmethod(settings.defaults)
    usage_modes = settings.USAGE_MODES

    def validate_profile(self, profile: ResolvedProfile) -> None:
        super().validate_profile(profile)
        retries = profile.get("max_retries")
        if isinstance(retries, int) and not isinstance(retries, bool) and retries > 0:
            raise UnsupportedCapabilityError("Ollama SDK does not support a positive max_retries")
