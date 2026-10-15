"""Anthropic Messages request and stream conversion."""

from typing import Any

from anthropic import omit

from my_llm.codec.anthropic.decoding.stream import AnthropicDecoder
from my_llm.codec.anthropic.encoding.messages import encode_messages
from my_llm.codec.anthropic.encoding.parameters import request_parameters
from my_llm.codec.anthropic.encoding.tools import encode_tools
from my_llm.profile.utils.resolution import ResolvedProfile


class AnthropicCodec:
    """Encode native Messages requests and create independent stream decoders."""

    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile

    def encode(self, messages: Any, *, tools: Any, **parameters: Any) -> dict[str, Any]:
        request = request_parameters(self.profile, **parameters)
        history, system = encode_messages(messages)
        request.update(
            model=self.profile.model,
            messages=history,
            stream=True,
            system="\n\n".join(system) if system else omit,
            tools=encode_tools(tools, self.profile),
        )
        return request

    def decoder(self) -> "AnthropicDecoder":
        """Return a fresh decoder for a raw AsyncAnthropic event stream."""
        return AnthropicDecoder(self.profile)
