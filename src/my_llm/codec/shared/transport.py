"""Resource ownership shared by native SDK stream implementations."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any


@asynccontextmanager
async def owned_stream(
    create: Callable[[], Awaitable[Any]],
    *,
    close_stream: str = "close",
) -> AsyncIterator[Any]:
    """Close one response stream without closing its borrowed SDK client."""
    async with AsyncExitStack() as resources:
        upstream = await create()
        resources.push_async_callback(getattr(upstream, close_stream))
        yield upstream
