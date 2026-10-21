import asyncio

import pytest

from my_llm import (
    ConfigurationError,
    IncompleteStreamError,
    MessageValidationError,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    generate,
)
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import chat_event
from tests.helpers.transport import FakeClient


def patch_client(monkeypatch, events, failure=None):
    client = FakeClient(events, failure=failure)
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    monkeypatch.setattr("my_llm.codec.openai.responses.transport.AsyncOpenAI", factory)
    return client, captured


async def test_owned_client_configuration_and_cleanup(monkeypatch):
    client, captured = patch_client(
        monkeypatch, [chat_event(content="ok"), chat_event(finish="stop")]
    )
    result = await generate(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(
            protocol="openai_chat",
            api_key=parameter("api_key", "profile"),
            timeout=config(30),
            max_retries=config(0),
        ),
        model="demo",
        messages=[{"role": "user", "content": "x"}],
    )
    assert result == {"role": "assistant", "content": "ok"}
    assert captured == [{"api_key": "profile", "timeout": 30, "max_retries": 0}]
    assert client.closed and client.upstream.closed


async def test_unstarted_and_early_closed_streams_release_resources(monkeypatch):
    client, captured = patch_client(
        monkeypatch,
        [chat_event(content="one"), chat_event(content="two"), chat_event(finish="stop")],
    )
    llm = MyLLM(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(protocol="openai_chat"),
        model=ModelProfile(model="demo"),
    )
    stream = await llm.generate(messages=[{"role": "user", "content": "x"}], stream=True)
    await stream.aclose()
    assert captured == []
    stream = await llm.generate(messages=[{"role": "user", "content": "x"}], stream=True)
    async with stream:
        async for event in stream:
            if event["choices"][0]["delta"].get("content"):
                break
    assert not client.closed and client.upstream.closed
    await llm.aclose()
    assert client.closed
    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_missing_terminal_event_preserves_partial_message(monkeypatch):
    client, _ = patch_client(monkeypatch, [chat_event(content="partial")])
    with pytest.raises(IncompleteStreamError) as error:
        await generate(
            stdio_protocol="openai_chat",
            provider=ProviderProfile(protocol="openai_chat"),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
        )
    assert error.value.partial_message == {"role": "assistant", "content": "partial"}
    assert client.closed and client.upstream.closed


async def test_transport_error_closes_everything(monkeypatch):
    client, _ = patch_client(monkeypatch, [], RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        await generate(
            stdio_protocol="openai_chat",
            provider=ProviderProfile(protocol="openai_chat"),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
        )
    assert client.closed and client.upstream.closed


async def test_cancellation_closes_everything(monkeypatch):
    client, _ = patch_client(monkeypatch, [])
    started = asyncio.Event()

    async def wait(self):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(type(client.upstream), "__anext__", wait)
    async with asyncio.TaskGroup() as tasks:
        task = tasks.create_task(
            generate(
                stdio_protocol="openai_chat",
                provider=ProviderProfile(protocol="openai_chat"),
                model="demo",
                messages=[{"role": "user", "content": "x"}],
            )
        )
        await started.wait()
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed and client.upstream.closed


async def test_stdio_protocol_binds_once_and_rejects_other_formats(monkeypatch):
    patch_client(monkeypatch, [chat_event(finish="stop")])
    llm = MyLLM(provider=ProviderProfile(protocol="openai_chat"), model=ModelProfile(model="demo"))
    await llm.generate(messages=[{"role": "developer", "content": [{"type": "text", "text": "x"}]}])
    assert llm.stdio_protocol == "openai_chat"
    with pytest.raises(MessageValidationError, match="bound"):
        await llm.generate(messages=[{"role": "user", "parts": [{"text": "x"}]}])


async def test_configuration_errors_happen_before_sdk_creation(monkeypatch):
    _, captured = patch_client(monkeypatch, [])
    llm = MyLLM(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(protocol="openai_chat"),
        model=ModelProfile(model="demo"),
    )
    with pytest.raises(ConfigurationError):
        await llm.generate(messages=[{"role": "user", "content": "x"}], stream="yes")
    for option in ("provider", "model", "client", "api_key", "timeout", "protocol"):
        with pytest.raises(TypeError):
            await llm.generate(messages=[{"role": "user", "content": "x"}], **{option: object()})
    assert captured == []


async def test_provider_error_is_not_masked_by_partial_tool_json(monkeypatch):
    from my_llm import ResponseFailedError

    events = [
        chat_event(
            content="partial",
            calls=[{"index": 0, "id": "call_1", "function": {"name": "tool", "arguments": "{"}}],
        ),
        {"error": {"message": "provider failure"}},
    ]
    client, _ = patch_client(monkeypatch, events)
    with pytest.raises(ResponseFailedError) as failure:
        await generate(
            stdio_protocol="openai_chat",
            provider=ProviderProfile(protocol="openai_chat"),
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert failure.value.partial_message["content"] == "partial"
    assert failure.value.details == {"message": "provider failure"}
    assert client.closed and client.upstream.closed


async def test_bound_stdio_protocol_does_not_detect_again(monkeypatch):
    patch_client(monkeypatch, [chat_event(finish="stop")])
    llm = MyLLM(
        provider=ProviderProfile(protocol="openai_chat"),
        model=ModelProfile(model="demo"),
        stdio_protocol="openai_chat",
    )

    def fail(*args):
        raise AssertionError("bound protocol must not be guessed again")

    monkeypatch.setattr(type(llm.codec), "match_protocol", fail)
    await llm.generate(messages=[{"role": "user", "content": "hi"}])


async def test_provider_paths_apply_before_standard_output_decoding(monkeypatch):
    from my_llm import omit

    event = chat_event(content="standard", reasoning="ignored")
    event["choices"][0]["delta"]["provider_text"] = "configured"
    patch_client(monkeypatch, [event, chat_event(finish="stop")])
    llm = MyLLM(
        provider=ProviderProfile(
            protocol="openai_chat",
            content=parameter(
                "content", protocol="openai_chat", path="$.choices[0].delta.provider_text"
            ),
            reasoning_content=omit,
        ),
        model=ModelProfile(model="demo"),
        stdio_protocol="gemini",
    )
    result = await llm.generate(messages=[{"role": "user", "parts": [{"text": "hi"}]}])
    assert result == {"role": "model", "parts": [{"text": "configured"}]}


async def test_target_finish_failure_does_not_publish_a_completed_result(monkeypatch):
    from my_llm import ResponseFailedError
    from tests.helpers.protocols import stream_events

    events = stream_events("anthropic")
    events[-2]["delta"]["stop_reason"] = "pause_turn"
    client = FakeClient(events)
    monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", lambda **kwargs: client)
    llm = MyLLM(
        provider=ProviderProfile(protocol="anthropic"),
        model=ModelProfile(
            model="demo", max_tokens=parameter("max_tokens", 2048, protocol="anthropic")
        ),
        stdio_protocol="openai_chat",
    )
    stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
    with pytest.raises(ResponseFailedError, match="finish reason"):
        async with stream:
            async for _ in stream:
                pass
    assert stream.result is None
    assert not client.closed and client.upstream.closed
    await llm.aclose()
    assert client.closed and client.upstream.closed


async def test_early_exit_during_final_tool_frames_has_no_completed_result(monkeypatch):
    from tests.helpers.protocols import stream_events

    client, _ = patch_client(monkeypatch, stream_events("openai_chat", tool=True))
    llm = MyLLM(
        provider=ProviderProfile(protocol="openai_chat"),
        model=ModelProfile(model="demo"),
        stdio_protocol="openai_chat",
    )
    stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
    async with stream:
        async for event in stream:
            if any(
                choice.get("delta", {}).get("tool_calls") for choice in event.get("choices", [])
            ):
                break
    assert stream.result is None
    assert not client.closed and client.upstream.closed
    await llm.aclose()
    assert client.closed and client.upstream.closed
