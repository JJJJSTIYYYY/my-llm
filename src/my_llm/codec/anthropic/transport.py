"""Reusable Anthropic SDK client and request-owned Messages streams."""

from typing import Any

from anthropic import AsyncAnthropic

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.transport import owned_stream
from my_llm.profile.resolution import ResolvedProfile, client_parameters


class AnthropicTransport(BaseTransport):
    """Official SDK lifecycle and native request execution for this protocol."""

    __slots__ = ()

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncAnthropic:
        return AsyncAnthropic(**client_parameters(profile))

    def upstream(
        self, client: AsyncAnthropic, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.messages.create(**request))
