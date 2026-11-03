"""Async downloads finish before SDK requests and survive cached retries."""

import asyncio
import base64
import json
from dataclasses import replace
from inspect import isawaitable

import httpx
import pytest

from my_llm import Config, MyLLM
from my_llm.codec.ollama.prebuilt import tool_images
from tests.helpers.protocols import stream_events
from tests.helpers.tool_images import DATA, URL, consume, request_history, responses_history
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import PNG, profiles


def downloads(monkeypatch, respond):
    clients, requests = [], []

    async def handler(request):
        requests.append(request)
        result = respond(request)
        return await result if isawaitable(result) else result

    def client(**options):
        value = httpx.AsyncClient(
            **options, transport=httpx.MockTransport(handler), trust_env=False
        )
        clients.append(value)
        return value

    monkeypatch.setattr(tool_images, "AsyncClient", client)
    return clients, requests


def llm_options(*, vision=True, cached=True):
    provider, model = profiles("ollama")
    return dict(
        provider=provider,
        model=replace(model, supports_vision=Config(default_value=vision)),
        stdio_protocol="openai_responses",
        semantic_loss_protocol="hook",
        input_semantic_loss_hook=tool_images.move_tool_images_to_user_message,
        enable_message_cache=cached,
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("redirect", [False, True])
async def test_downloaded_images_are_cached_once_and_ready_before_stream_return(
    monkeypatch, stream, redirect
):
    def response(request):
        if redirect and request.url.path == "/a.png":
            return httpx.Response(302, headers={"location": "/actual.png"})
        # Actual bytes determine image format, independently of the HTTP MIME hint.
        return httpx.Response(
            200, content=base64.b64decode(PNG), headers={"content-type": "text/plain"}
        )

    clients, requests = downloads(monkeypatch, response)
    batch = responses_history()
    output = next(item for item in batch if item.get("type") == "function_call_output")
    output["output"].insert(2, {"type": "input_image", "image_url": URL})
    with LocalSDK("ollama", wire_frames("ollama", stream_events("ollama"), 41)) as sdk:
        async with MyLLM(**llm_options()) as llm:
            result = await llm.generate(messages=batch, stream=stream)
            assert len(requests) == (2 if redirect else 1)
            assert all(client.is_closed for client in clients)
            if stream:
                assert not sdk.requests
                await consume(result)
            first = json.loads(sdk.requests[-1].content)
            message = first["messages"][-1]
            assert message["images"] == [PNG, PNG, DATA]
            assert "[Unsupported Image]" not in message["content"]
            await llm.generate(messages=batch)
            assert len(requests) == (2 if redirect else 1)
            assert len(clients) == 1
            retry = json.loads(sdk.requests[-1].content)
            assert retry["messages"] == first["messages"]
            await llm.generate(messages=[{"role": "user", "content": "next"}])
            assert json.loads(sdk.requests[-1].content)["messages"][-2] == message


@pytest.mark.parametrize("failure", ["http", "timeout", "non_image"])
async def test_download_failures_become_cached_unsupported_markers(monkeypatch, failure):
    def response(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("download timed out", request=request)
        return httpx.Response(404 if failure == "http" else 200, content=b"not an image")

    clients, requests = downloads(monkeypatch, response)
    with LocalSDK("ollama", wire_frames("ollama", stream_events("ollama"), 47)) as sdk:
        async with MyLLM(**llm_options()) as llm:
            for _ in range(2):
                await llm.generate(messages=responses_history())
                message = json.loads(sdk.requests[-1].content)["messages"][-1]
                assert message == {
                    "role": "user",
                    "content": "[Unsupported Image]",
                    "images": [DATA],
                }
        assert len(requests) == len(clients) == 1
        assert clients[0].is_closed


async def test_url_downloads_overlap_and_preserve_original_image_order(monkeypatch):
    barrier = asyncio.Barrier(2)

    async def response(request):
        await barrier.wait()
        return httpx.Response(200, content=base64.b64decode(PNG))

    clients, requests = downloads(monkeypatch, response)
    batch = responses_history()
    output = next(item for item in batch if item.get("type") == "function_call_output")
    output["output"].insert(2, {"type": "input_image", "image_url": URL + "?second"})
    with LocalSDK("ollama", wire_frames("ollama", stream_events("ollama"), 43)) as sdk:
        async with MyLLM(**llm_options()) as llm:
            async with asyncio.timeout(5):
                await llm.generate(messages=batch)
        assert len(requests) == 2
        assert json.loads(sdk.requests[-1].content)["messages"][-1]["images"] == [PNG, PNG, DATA]
        assert clients[0].is_closed


async def test_cancellation_closes_downloads_and_allows_clean_retry(monkeypatch):
    started, blocked = asyncio.Event(), asyncio.Event()
    count = 0

    async def response(request):
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            await blocked.wait()
        return httpx.Response(200, content=base64.b64decode(PNG))

    clients, requests = downloads(monkeypatch, response)
    with LocalSDK("ollama", wire_frames("ollama", stream_events("ollama"), 41)) as sdk:
        async with MyLLM(**llm_options()) as llm:
            task = asyncio.create_task(llm.generate(messages=responses_history(), stream=True))
            async with asyncio.timeout(5):
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert clients[0].is_closed
            assert not sdk.requests
            assert not any(
                "images" in message
                for message in llm.codec._message_cache.get(llm._message_cache_key)
            )
            await llm.generate(messages=responses_history())
            assert len(requests) == 2
            message = json.loads(sdk.requests[-1].content)["messages"][-1]
            assert message["images"] == [PNG, DATA]
        assert all(client.is_closed for client in clients)


async def test_models_without_vision_never_download(monkeypatch):
    def unexpected(**options):
        pytest.fail("a model without vision must not trigger image downloads")

    monkeypatch.setattr(tool_images, "AsyncClient", unexpected)
    with LocalSDK("ollama", wire_frames("ollama", stream_events("ollama"), 37)) as sdk:
        async with MyLLM(**llm_options(vision=False)) as llm:
            await llm.generate(messages=responses_history())
        request = json.loads(sdk.requests[-1].content)
        assert str(request).count("[Unsupported Image]") == 2
        assert not any(message.get("images") for message in request_history("ollama", request))


async def test_downloads_without_message_cache_are_request_local(monkeypatch):
    clients, requests = downloads(
        monkeypatch, lambda request: httpx.Response(200, content=base64.b64decode(PNG))
    )
    with LocalSDK("ollama", wire_frames("ollama", stream_events("ollama"), 37)) as sdk:
        async with MyLLM(**llm_options(cached=False)) as llm:
            for _ in range(2):
                await llm.generate(messages=responses_history())
            assert not llm.codec._message_cache._histories
        assert len(clients) == len(requests) == 2
        assert all(client.is_closed for client in clients)
        assert all(
            json.loads(request.content)["messages"][-1]["images"] == [PNG, DATA]
            for request in sdk.requests
        )
