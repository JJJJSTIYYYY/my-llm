"""Stateless official SDK lifecycle and request execution contract."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Hashable
from contextlib import AbstractAsyncContextManager
from typing import Any

from my_llm.profile.resolution import ResolvedProfile


class BaseTransport(ABC):
    """Execute native requests without interpreting intermediate values.

    BaseCodec composes this transport with its encoder and decoder. MyLLM owns
    cached SDK clients; each upstream context owns only its response resources.
    Profiles and clients are call arguments, never retained on this object.
    """

    __slots__ = ()

    @abstractmethod
    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> Any:
        """Create an official SDK client for the instance-owned cache."""
        ...

    async def close_client(self, client: Any) -> None:
        """Close a cached SDK client through its public async close interface."""
        await client.close()

    def client_key(self, profile: ResolvedProfile, **parameters: Any) -> Hashable:
        """Identify request options bound to the SDK client, if any."""
        return None

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate SDK connection settings before creating any client."""

    @abstractmethod
    def upstream(
        self, client: Any, request: dict, profile: ResolvedProfile, **parameters: Any
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        """Execute a native request and close its response, borrowing the client."""
        ...
