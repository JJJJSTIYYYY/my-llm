"""Each generation owns its SDK client and response stream."""

import asyncio

import pytest

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile, generate
from tests.helpers.protocols import chat_event
from tests.helpers.transport import FakeClient


@pytest.fixture
def sdk_factory(monkeypatch):
    clients = []

    def factory(**kwargs):
        client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    return clients


def make_llm():
    return MyLLM(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(model=Attribute(default_value="demo")),
    )


async def test_each_generation_creates_and_closes_its_own_client(sdk_factory):
    llm = make_llm()
    for streaming in (False, True, False):
        result = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=streaming)
        if streaming:
            async with result:
                async for _ in result:
                    pass
            result = result.result
        assert result["content"] == "ok"
        assert all(client.closed and client.close_count == 1 for client in sdk_factory)
    assert len(sdk_factory) == 3
    assert all(len(client.requests) == 1 for client in sdk_factory)
    assert all(client.close_count == 1 for client in sdk_factory)


async def test_client_creation_failure_does_not_prevent_later_generation(monkeypatch):
    calls = 0
    clients = []

    def factory(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("client construction failed")
        client = FakeClient([chat_event(finish="stop")])
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    llm = make_llm()
    with pytest.raises(RuntimeError, match="construction failed"):
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
    for _ in range(2):
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
    assert calls == 3
    assert all(client.close_count == 1 for client in clients)


async def test_request_creation_failure_closes_client(monkeypatch, sdk_factory):
    async def fail(self, **kwargs):
        raise RuntimeError("request creation failed")

    monkeypatch.setattr(FakeClient, "request", fail)
    llm = make_llm()
    with pytest.raises(RuntimeError, match="request creation failed"):
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
    assert sdk_factory[0].close_count == 1


@pytest.mark.parametrize("exit_mode", ["complete", "early", "failure", "cancel"])
async def test_one_shot_stream_owns_and_closes_temporary_client(
    sdk_factory, monkeypatch, exit_mode
):
    stream = await generate(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model="demo",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert sdk_factory == []
    while True:
        event = await anext(stream)
        if any(choice["delta"].get("content") for choice in event.get("choices", [])):
            break
    client = sdk_factory[0]
    if exit_mode == "failure":
        client.upstream.failure = RuntimeError("stream failed")
        with pytest.raises(RuntimeError, match="stream failed"):
            async for _ in stream:
                pass
    elif exit_mode == "cancel":
        started = asyncio.Event()

        async def wait(self):
            started.set()
            await asyncio.Future()

        monkeypatch.setattr(type(client.upstream), "__anext__", wait)
        task = asyncio.create_task(anext(stream))
        async with asyncio.timeout(5):
            await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif exit_mode == "complete":
        async for _ in stream:
            pass
        assert stream.result["content"] == "ok"
    else:
        await stream.aclose()
        assert stream.result is None
    await stream.aclose()
    assert client.close_count == 1 and client.upstream.closed


async def test_unstarted_one_shot_stream_does_not_create_client(sdk_factory):
    stream = await generate(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model="demo",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    await stream.aclose()
    assert sdk_factory == []
