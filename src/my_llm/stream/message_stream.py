"""Lifecycle wrapper for standard-protocol asynchronous response events."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import Any

from my_llm.stream._state import StreamState


class MessageStream(AsyncIterator[dict[str, Any]]):
    """A closable stream of standard event dictionaries in ``stdio_protocol``.

    Text/thinking are incremental. Function calls are emitted only when complete.
    After exhaustion, ``result`` contains the native message (Responses: items).
    Use ``async with stream`` to release SDK resources after early exit.
    """

    def __init__(self, iterator: AsyncGenerator[dict[str, Any], None], state: StreamState) -> None:
        self._iterator = iterator
        self._state = state
        self._closed = False
        self._close_callbacks = AsyncExitStack()

    def add_close_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Release an async owner once when this stream closes, including before iteration."""
        if self._closed:
            raise RuntimeError("MessageStream is closed")
        self._close_callbacks.push_async_callback(callback)

    @property
    def result(self) -> Any:
        """Return the completed standard message, or None before exhaustion."""
        return deepcopy(self._state.result)

    def __aiter__(self) -> MessageStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._iterator)
        # Exhaustion and cancellation must release resources as well as errors.
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Close the response and any one-shot owner, preserving reusable clients."""
        if not self._closed:
            self._closed = True
            try:
                await self._iterator.aclose()
            finally:
                await self._close_callbacks.aclose()

    async def __aenter__(self) -> MessageStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
