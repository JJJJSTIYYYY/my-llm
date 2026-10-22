"""Official SDK construction and response resource ownership only."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any


class BaseTransport(ABC):
    """Abstract SDK client and response lifecycle interface for custom protocols.

    Implement create_client and upstream, overriding close_client if required by
    the SDK. Declare empty __slots__; instances retain no clients or request state.
    Callers own created clients and must close them after response contexts exit.
    """

    __slots__ = ()

    @abstractmethod
    def create_client(self, parameters: dict) -> Any:
        """Return a new SDK client from prepared constructor keyword arguments.

        parameters comes from the protocol's client preparation API. The caller owns
        the returned client and must eventually pass it to close_client.
        """
        ...

    async def close_client(self, client: Any) -> None:
        """Release an SDK client by awaiting client.close().

        Override when the SDK uses a different asynchronous closing interface.
        """
        await client.close()

    @abstractmethod
    def upstream(
        self, client: Any, request: dict
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        """Return an async context manager yielding native SDK response events.

        client is a borrowed SDK client and request contains final SDK keyword
        arguments. Exiting the context releases response resources, including on early
        exit or failure, and leaves client ownership with the caller.
        """
        ...
