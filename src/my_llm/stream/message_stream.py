"""Lifecycle wrapper for standard-protocol asynchronous response events."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import Any

from my_llm.stream._state import StreamState


class MessageStream(AsyncIterator[dict[str, Any]]):
    """A closable asynchronous iterator of protocol-native event dictionaries.

    Obtain streams from ``await llm.generate(..., stream=True)``. Text and visible
    reasoning arrive incrementally; tool calls are exposed with complete arguments.
    Use ``async with stream`` so early exit, errors and cancellation release the
    response. Iteration is single-use; do not consume the same stream concurrently.

    Attributes:
        result: Independent copy of the completed native message, or Responses
            output items. None until successful exhaustion, including after an
            early close. Save this result as conversation history.
    """

    def __init__(self, iterator: AsyncGenerator[dict[str, Any], None], state: StreamState) -> None:
        self._iterator = iterator
        self._state = state
        self._closed = False
        self._close_callbacks = AsyncExitStack()

    def add_close_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register an asynchronous no-argument cleanup callback.

        Callbacks run once in reverse registration order when the stream closes,
        even if iteration never starts. Registering after close raises RuntimeError.
        """
        if self._closed:
            raise RuntimeError("MessageStream is closed")
        self._close_callbacks.push_async_callback(callback)

    @property
    def result(self) -> Any:
        """Return a deep copy of the completed message/items, or None before completion."""
        return deepcopy(self._state.result)

    def __aiter__(self) -> MessageStream:
        """Return this single-use asynchronous iterator."""
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Yield the next event; close on exhaustion, errors or cancellation."""
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._iterator)
        # Exhaustion and cancellation must release resources as well as errors.
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Close the response and run registered cleanup callbacks once.

        Safe before iteration and on repeated calls. Closing early does not produce a
        completed result. A reusable MyLLM instance remains available for other calls.
        """
        if not self._closed:
            self._closed = True
            try:
                await self._iterator.aclose()
            finally:
                await self._close_callbacks.aclose()

    async def __aenter__(self) -> MessageStream:
        """Return this stream for managed asynchronous consumption."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close the stream when leaving its asynchronous context."""
        await self.aclose()
