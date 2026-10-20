"""Official SDK construction and response resource ownership only."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any


class BaseTransport(ABC):
    """Borrow clients and execute final SDK kwargs; retain no request state."""

    __slots__ = ()

    @abstractmethod
    def create_client(self, parameters: dict) -> Any:
        """Construct a client from Decoder's already prepared SDK arguments."""
        ...

    async def close_client(self, client: Any) -> None:
        await client.close()

    @abstractmethod
    def upstream(
        self, client: Any, request: dict
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        """Execute final SDK kwargs and close this request's response resources."""
        ...
