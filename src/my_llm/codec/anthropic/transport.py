"""Official SDK invocation and request-owned stream cleanup."""

from anthropic import AsyncAnthropic

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.transport import owned_stream


class AnthropicTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> AsyncAnthropic:
        return AsyncAnthropic(**parameters)

    def upstream(self, client: AsyncAnthropic, request: dict):
        return owned_stream(lambda: client.messages.create(**request))
