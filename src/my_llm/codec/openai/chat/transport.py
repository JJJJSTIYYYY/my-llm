"""Official SDK invocation and request-owned stream cleanup."""

from openai import AsyncOpenAI

from my_llm.codec.base.transport import BaseTransport, owned_stream


class ChatCompletionsTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> AsyncOpenAI:
        """Create the Chat Completions SDK client from prepared connection settings."""
        return AsyncOpenAI(**parameters)

    def upstream(self, client: AsyncOpenAI, request: dict):
        """Open a Chat Completions response stream and close it when its context exits."""
        return owned_stream(lambda: client.chat.completions.create(**request))
