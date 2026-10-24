"""Official SDK invocation and request-owned stream cleanup."""

from openai import AsyncOpenAI

from my_llm.codec.base.transport import BaseTransport, owned_stream


class ChatCompletionsTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> AsyncOpenAI:
        return AsyncOpenAI(**parameters)

    def upstream(self, client: AsyncOpenAI, request: dict):
        return owned_stream(lambda: client.chat.completions.create(**request))
