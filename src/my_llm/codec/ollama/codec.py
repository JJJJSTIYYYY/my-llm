"""Ollama native chat requests and streamed message conversion."""

from typing import Any

from openai import Omit

from my_llm.codec.ollama.decoding.stream import OllamaDecoder
from my_llm.codec.ollama.encoding.messages import encode_messages
from my_llm.codec.ollama.encoding.parameters import request_parameters
from my_llm.codec.ollama.encoding.tools import encode_tools
from my_llm.profile.utils.resolution import ResolvedProfile


class OllamaCodec:
    """Encode only local functions for AsyncClient.chat(stream=True)."""

    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile

    def encode(self, messages: Any, *, tools: Any, **parameters: Any) -> dict[str, Any]:
        request = request_parameters(self.profile, **parameters)
        history = encode_messages(messages, self.profile)
        encoded_tools = encode_tools(tools, self.profile)
        request.update(model=self.profile.model, messages=history, stream=True)
        if not isinstance(encoded_tools, Omit):
            request["tools"] = encoded_tools
        return request

    def decoder(self) -> "OllamaDecoder":
        """Return a fresh decoder for one native chat stream."""
        return OllamaDecoder(self.profile)
