"""Official SDK invocation and request-owned stream cleanup."""

from ollama import AsyncClient as AsyncOllama

from my_llm.codec.base.transport import BaseTransport, owned_stream


class OllamaTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> AsyncOllama:
        """Create the Ollama SDK client from prepared connection settings."""
        return AsyncOllama(**parameters)

    def upstream(self, client: AsyncOllama, request: dict):
        """Open a Ollama response stream and close it when its context exits."""
        return owned_stream(lambda: client.chat(**request), close_stream="aclose")
