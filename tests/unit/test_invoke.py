import asyncio

import pytest
from conftest import FakeClient, chat_event

from my_llm import (
    ConfigurationError,
    IncompleteStreamError,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    UnsupportedProviderError,
    generate,
)


async def test_owned_client_parameters_and_cleanup(monkeypatch):
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    parameters = []

    def factory(**kwargs):
        parameters.append(kwargs)
        return client

    monkeypatch.setattr("my_llm.invoke.AsyncOpenAI", factory)
    result = await generate(
        provider=ProviderProfile(
            api_key="profile", base_url="https://example.test", timeout=30, max_retries=2
        ),
        model="demo",
        messages=[{"role": "user", "content": "x"}],
        api_key="override",
        max_retries=0,
    )
    assert result["content"] == "ok"
    assert parameters == [
        {"api_key": "override", "base_url": "https://example.test", "timeout": 30, "max_retries": 0}
    ]
    assert client.closed and client.upstream.closed


async def test_borrowed_client_not_closed():
    client = FakeClient([chat_event(finish="stop")])
    await generate(
        provider=ProviderProfile(),
        model="demo",
        messages=[{"role": "user", "content": "x"}],
        client=client,
    )
    assert client.upstream.closed and not client.closed


async def test_early_exit_closes_stream():
    client = FakeClient(
        [chat_event(content="one"), chat_event(content="two"), chat_event(finish="stop")]
    )
    stream = await generate(
        provider=ProviderProfile(),
        model="demo",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        client=client,
    )
    assert not client.requests  # Request/SDK client creation is lazy.
    async with stream:
        async for chunk in stream:
            assert chunk.content == "one"
            break
    assert client.upstream.closed
    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_truncated_stream_has_partial_message_and_cleanup():
    client = FakeClient([chat_event(content="partial")])
    with pytest.raises(IncompleteStreamError) as error:
        await generate(
            provider=ProviderProfile(),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
            client=client,
        )
    assert error.value.partial_message == {"role": "assistant", "content": "partial"}
    assert client.upstream.closed


async def test_transport_failure_closes_stream():
    client = FakeClient([chat_event(content="partial")], failure=RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        await generate(
            provider=ProviderProfile(),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
            client=client,
        )
    assert client.upstream.closed


async def test_cancellation_closes_upstream():
    client = FakeClient([])
    started = asyncio.Event()

    async def waiting_next(self):
        started.set()
        await asyncio.Future()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(type(client.upstream), "__anext__", waiting_next)
    try:
        task = asyncio.create_task(
            generate(
                provider=ProviderProfile(),
                model="demo",
                messages=[{"role": "user", "content": "x"}],
                client=client,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.upstream.closed
    finally:
        monkeypatch.undo()


async def test_reserved_protocol_and_invalid_stream_fail_before_request():
    client = FakeClient([])
    invoke = MyLLM(provider=ProviderProfile(), model=ModelProfile(model="demo"), client=client)
    with pytest.raises(UnsupportedProviderError):
        await invoke.generate(messages=[{"role": "user", "content": "x"}], protocol="anthropic")
    with pytest.raises(ConfigurationError):
        await invoke.generate(messages=[{"role": "user", "content": "x"}], stream="yes")
    with pytest.raises(ConfigurationError, match="injected"):
        await invoke.generate(messages=[{"role": "user", "content": "x"}], timeout=2)
    assert not client.requests
