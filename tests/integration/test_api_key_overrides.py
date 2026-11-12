"""Per-call authentication reaches SDK headers and stays isolated between calls."""

import asyncio
import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from my_llm import Attribute, Config, InputParameter, ModelProfile, MyLLM, ProviderProfile
from tests.helpers.protocols import PROTOCOLS, stream_events
from tests.helpers.transport import sse


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("path", ["$.api_key", "$.extra_headers.Authorization"])
@pytest.mark.parametrize("stream", [False, True])
async def test_api_key_overrides_reach_sdk_headers_and_reuse_clients(
    monkeypatch, protocol, path, stream
):
    requests, clients, constructors = [], [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            text=sse(stream_events(protocol), anthropic=protocol == "anthropic"),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        constructors.append(dict(kwargs))
        sdk = AsyncAnthropic if protocol == "anthropic" else AsyncOpenAI
        client = sdk(
            **{"api_key": "sdk-fallback", **kwargs},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    target = {
        "openai_chat": "my_llm.codec.openai.chat.transport.AsyncOpenAI",
        "openai_responses": "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        "anthropic": "my_llm.codec.anthropic.transport.AsyncAnthropic",
    }[protocol]
    monkeypatch.setattr(target, factory)
    provider = ProviderProfile(
        protocol=Config(default_value=protocol),
        api_key=InputParameter(path=path, default_value="profile-key"),
        base_url=Config(default_value="https://local.test"),
    )
    async with MyLLM(
        provider=provider,
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
    ) as llm:
        profile_hash = llm.profile_hash
        for options in (
            {},
            {"api_key": "call-key"},
            {"api_key": None},
            {"api_key": "call-key"},
            {"api_key": "profile-key"},
        ):
            result = await llm.generate(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2048,
                stream=stream,
                **options,
            )
            if stream:
                async with result:
                    async for _ in result:
                        pass
                result = result.result
            assert result["role"] == "assistant"
        assert len(clients) == 2
        assert all(not client.is_closed() for client in clients)
        assert llm.resolved_profile.get("api_key") == "profile-key"
        assert llm.profile_hash == profile_hash
        assert provider.api_key.default_value == "profile-key"
        assert llm.client_parameters == constructors[0]
    assert all(client.is_closed() for client in clients)
    keys = ["profile-key", "call-key", "profile-key", "call-key", "profile-key"]
    header = "x-api-key" if protocol == "anthropic" and path == "$.api_key" else "Authorization"
    prefix = "Bearer " if protocol != "anthropic" and path == "$.api_key" else ""
    assert [request.headers[header] for request in requests] == [prefix + key for key in keys]
    for request in requests:
        body = json.loads(request.content)
        assert "api_key" not in body and "default_headers" not in body
        assert "profile-key" not in request.content.decode()
        assert "call-key" not in request.content.decode()
    if path != "$.api_key":
        assert constructors[1]["default_headers"] == {"Authorization": "call-key"}
        assert "api_key" not in constructors[1]


async def test_concurrent_prepared_streams_keep_their_own_key(monkeypatch):
    barrier = asyncio.Barrier(3)
    observed, clients = [], []

    async def handler(request):
        observed.append(
            (
                json.loads(request.content)["messages"][0]["content"],
                request.headers["Authorization"],
            )
        )
        await barrier.wait()
        return httpx.Response(
            200,
            text=sse(stream_events("openai_chat")),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
    ) as llm:
        streams = [
            await llm.generate(
                messages=[{"role": "user", "content": key}], api_key=key, stream=True
            )
            for key in ("first-key", "second-key", "first-key")
        ]
        assert not clients

        async def consume(stream):
            async with stream:
                async for _ in stream:
                    pass
            assert stream.result["role"] == "assistant"

        async with asyncio.timeout(5):
            await asyncio.gather(*(consume(stream) for stream in reversed(streams)))
        assert len(clients) == 2
        assert sorted(observed) == [
            ("first-key", "Bearer first-key"),
            ("first-key", "Bearer first-key"),
            ("second-key", "Bearer second-key"),
        ]
        assert all(not client.is_closed() for client in clients)
    assert all(client.is_closed() for client in clients)
