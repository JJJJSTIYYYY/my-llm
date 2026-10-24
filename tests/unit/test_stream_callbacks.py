"""Stream owners attach cleanup through the public lifecycle interface."""

import asyncio

import pytest

from my_llm.runtime.message_stream import MessageStream
from my_llm.runtime.state import StreamState


@pytest.mark.parametrize("consume", [False, True])
async def test_close_callbacks_run_once_even_without_starting_the_stream(consume):
    events = []

    async def values():
        events.append("started")
        yield {"event": "value"}

    async def close_owner():
        events.append("closed")

    stream = MessageStream(values(), StreamState("demo"))
    stream.add_close_callback(close_owner)
    if consume:
        async for _ in stream:
            pass
    await stream.aclose()
    await stream.aclose()
    assert events == (["started", "closed"] if consume else ["closed"])
    with pytest.raises(RuntimeError, match="MessageStream is closed"):
        stream.add_close_callback(close_owner)


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_all_registered_owners_close_when_a_callback_fails(error):
    events = []

    async def values():
        yield {}

    async def close_owner():
        events.append("owner")

    async def fail():
        events.append("failure")
        raise error("cleanup failed")

    stream = MessageStream(values(), StreamState("demo"))
    stream.add_close_callback(close_owner)
    stream.add_close_callback(fail)
    with pytest.raises(error, match="cleanup failed"):
        await stream.aclose()
    await stream.aclose()
    assert events == ["failure", "owner"]


async def test_owners_close_when_the_response_iterator_cannot_close():
    events = []

    async def values():
        try:
            yield {}
        finally:
            raise RuntimeError("response cleanup failed")

    async def close_owner():
        events.append("closed")

    stream = MessageStream(values(), StreamState("demo"))
    stream.add_close_callback(close_owner)
    await anext(stream)
    with pytest.raises(RuntimeError, match="response cleanup failed"):
        await stream.aclose()
    await stream.aclose()
    assert events == ["closed"]


async def test_iteration_cancellation_releases_the_owner_once():
    started = asyncio.Event()
    events = []

    async def values():
        started.set()
        await asyncio.Future()
        yield {}

    async def close_owner():
        events.append("closed")

    stream = MessageStream(values(), StreamState("demo"))
    stream.add_close_callback(close_owner)
    task = asyncio.create_task(anext(stream))
    async with asyncio.timeout(5):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    assert events == ["closed"]
