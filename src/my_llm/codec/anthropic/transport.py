"""Official SDK invocation and request-owned stream cleanup."""

from anthropic import AsyncAnthropic

from my_llm.codec.base.transport import BaseTransport, owned_stream


class AnthropicTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> AsyncAnthropic:
        """Create the Anthropic SDK client from prepared connection settings."""
        return AsyncAnthropic(**parameters)

    def upstream(self, client: AsyncAnthropic, request: dict):
        """Open a Anthropic response stream and close it when its context exits."""
        return owned_stream(lambda: client.messages.create(**request))
