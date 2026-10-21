"""SDK clients live with MyLLM; response streams live with individual requests."""

import asyncio

import pytest

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile, generate
from my_llm.codec.openai.chat.codec import ChatCompletionsCodec
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
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


async def test_llm_uses_the_transport_composed_by_its_codec():
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    calls = []

    class Transport(ChatCompletionsTransport):
        __slots__ = ()

        def create_client(self, profile, **parameters):
            calls.append("create")
            return client

        async def close_client(self, value):
            calls.append("close")
            await super().close_client(value)

    class Codec(ChatCompletionsCodec):
        __slots__ = ()
        transport = Transport()

    async with make_llm() as llm:
        codec = Codec()
        llm.codec.codecs["openai_chat"] = codec
        assert llm.codec.transport("openai_chat") is codec.transport
        for _ in range(2):
            result = await llm.generate(messages=[{"role": "user", "content": "hi"}])
            assert result["content"] == "ok"
        assert calls == ["create"] and not client.closed
    assert calls == ["create", "close"] and client.close_count == 1


async def test_client_is_lazy_reused_and_closed_once(sdk_factory):
    llm = make_llm()
    async with llm:
        assert sdk_factory == []
        for streaming in (False, True, False):
            result = await llm.generate(
                messages=[{"role": "user", "content": "hi"}], stream=streaming
            )
            if streaming:
                async with result:
                    async for _ in result:
                        pass
                result = result.result
            assert result["content"] == "ok"
        assert len(sdk_factory) == 1
        client = sdk_factory[0]
        assert len(client.requests) == 3
        assert not client.closed and all(stream.closed for stream in client.streams)
    await llm.aclose()
    assert client.close_count == 1


async def test_instances_do_not_share_sdk_clients(sdk_factory):
    async with make_llm() as first, make_llm() as second:
        await first.generate(messages=[{"role": "user", "content": "first"}])
        await second.generate(messages=[{"role": "user", "content": "second"}])
        assert len(sdk_factory) == 2
        await first.aclose()
        assert sdk_factory[0].closed and not sdk_factory[1].closed
        await second.generate(messages=[{"role": "user", "content": "again"}])
    assert all(client.close_count == 1 for client in sdk_factory)


async def test_closed_instance_rejects_new_and_prepared_streams(sdk_factory):
    llm = make_llm()
    stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
    await llm.aclose()
    with pytest.raises(RuntimeError, match="MyLLM is closed"):
        await anext(stream)
    with pytest.raises(RuntimeError, match="MyLLM is closed"):
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError, match="MyLLM is closed"):
        async with llm:
            pass
    assert sdk_factory == []


async def test_failed_client_construction_can_be_retried(monkeypatch):
    calls = 0
    client = FakeClient([chat_event(finish="stop")])

    def factory(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("failed to create SDK client")
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    async with make_llm() as llm:
        with pytest.raises(RuntimeError, match="failed to create"):
            await llm.generate(messages=[{"role": "user", "content": "hi"}])
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
        assert calls == 2
    assert client.close_count == 1


async def test_failed_request_creation_preserves_client(monkeypatch, sdk_factory):
    async with make_llm() as llm:
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
        client = sdk_factory[0]
        request = client.chat.completions.create

        async def fail(**kwargs):
            raise RuntimeError("failed to create response")

        monkeypatch.setattr(client.chat.completions, "create", fail)
        with pytest.raises(RuntimeError, match="failed to create response"):
            await llm.generate(messages=[{"role": "user", "content": "hi"}])
        assert not client.closed
        monkeypatch.setattr(client.chat.completions, "create", request)
        assert (await llm.generate(messages=[{"role": "user", "content": "hi"}]))["content"] == "ok"
        assert len(sdk_factory) == 1


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
