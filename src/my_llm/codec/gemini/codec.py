"""Google Gen AI native content and streamed candidate conversion."""

from typing import Any

from google.genai import types
from openai import Omit

from my_llm.codec.gemini.decoding.stream import GeminiDecoder
from my_llm.codec.gemini.encoding.messages import encode_messages
from my_llm.codec.gemini.encoding.parameters import request_parameters
from my_llm.codec.gemini.encoding.tools import encode_tools
from my_llm.profile.utils.resolution import ResolvedProfile


class GeminiCodec:
    """Encode generate_content_stream requests without automatic tool execution."""

    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile

    def encode(self, messages: Any, *, tools: Any, **parameters: Any) -> dict[str, Any]:
        config = request_parameters(self.profile, **parameters)
        history, system = encode_messages(messages)
        encoded_tools = encode_tools(tools, self.profile)
        if system:
            config["system_instruction"] = "\n\n".join(system)
        if not isinstance(encoded_tools, Omit):
            config["tools"] = encoded_tools
        return {
            "model": self.profile.model,
            "contents": history,
            "config": types.GenerateContentConfig(**config),
        }

    def decoder(self) -> "GeminiDecoder":
        """Return a new decoder for one GenerateContentResponse stream."""
        return GeminiDecoder(self.profile)
