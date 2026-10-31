"""Official SDK construction and response resource ownership only."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any


class BaseTransport(ABC):
    """Abstract SDK client and response lifecycle interface for custom protocols.

    Implement create_client and upstream, overriding close_client if required by
    the SDK. Each codec owns its transport; callers own created clients and must
    close them after response contexts exit.
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


@asynccontextmanager
async def owned_stream(
    create: Callable[[], Awaitable[Any]],
    *,
    close_stream: str = "close",
) -> AsyncIterator[Any]:
    """Own one SDK response stream for the lifetime of an asynchronous context.

    Await create lazily when the context is entered, register the SDK close
    method named by close_stream, and yield the native iterator. Normal exit,
    early exit, errors and cancellation run its cleanup. The SDK client used
    by create remains borrowed and is closed by MyLLM instead.
    """
    async with AsyncExitStack() as resources:
        upstream = await create()
        resources.push_async_callback(getattr(upstream, close_stream))
        yield upstream
