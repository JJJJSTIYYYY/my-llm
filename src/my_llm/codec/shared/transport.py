"""Resource ownership shared by native SDK stream implementations."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any


@asynccontextmanager
async def owned_stream(
    client: Any,
    create: Callable[[], Awaitable[Any]],
    *,
    close_stream: str = "close",
) -> AsyncIterator[Any]:
    """Close the body before its client, including failed creation or cancellation."""
    async with AsyncExitStack() as resources:
        resources.push_async_callback(client.close)
        upstream = await create()
        resources.push_async_callback(getattr(upstream, close_stream))
        yield upstream
