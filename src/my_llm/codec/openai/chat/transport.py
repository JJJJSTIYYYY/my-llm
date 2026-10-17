"""Reusable OpenAI SDK client and request-owned Chat streams."""

from typing import Any

from openai import AsyncOpenAI

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.transport import owned_stream
from my_llm.profile.resolution import ResolvedProfile, client_parameters


class ChatCompletionsTransport(BaseTransport):
    """Official SDK lifecycle and native request execution for this protocol."""

    __slots__ = ()

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**client_parameters(profile))

    def upstream(
        self, client: AsyncOpenAI, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.chat.completions.create(**request))
