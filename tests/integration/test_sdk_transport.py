"""Official OpenAI SDK serialization and SSE parsing, without public client injection."""

import json
from dataclasses import dataclass

import httpx
import pytest
from openai import AsyncOpenAI

from my_llm import ModelProfile, MyLLM, ProviderProfile
from my_llm.profile.types import Parameter
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history, stream_events, tools
from tests.helpers.schema import validate_request
from tests.helpers.transport import sse


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize(
    "stdio", ["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"]
)
@pytest.mark.parametrize("stream", [False, True])
async def test_official_openai_transport_all_stdio_formats(monkeypatch, protocol, stdio, stream):
    requests, clients = [], []

    def handler(request):
        requests.append(request)
        validate_request(protocol, json.loads(request.content))
        return httpx.Response(
            200,
            text=sse(stream_events(protocol, tool=True), done=protocol == "openai_chat"),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    monkeypatch.setattr("my_llm.codec.openai.responses.transport.AsyncOpenAI", factory)
    llm = MyLLM(
        provider=ProviderProfile(
            protocol=protocol,
            api_key=parameter("api_key", "local-test"),
            base_url=config("base_url", "https://local.test/v1"),
            usage_mode=config(
                "usage_mode", "native" if protocol == "openai_responses" else "stream_options"
            ),
        ),
        model=ModelProfile(
            model="demo", supports_function_calling=config("supports_function_calling", True)
        ),
        stdio_protocol=stdio,
    )
    result = await llm.generate(messages=history(stdio), tools=tools(stdio), stream=stream)
    if stream:
        async with result:
            events = [event async for event in result]
        assert events and all(isinstance(e, dict) for e in events)
        result = result.result
    assert isinstance(result, list if stdio == "openai_responses" else dict)
    body = json.loads(requests[0].content)
    assert body["model"] == "demo" and body["stream"] is True
    assert "Sunny" in str(body["input" if protocol == "openai_responses" else "messages"])
    assert requests[0].url.path.endswith(
        "/responses" if protocol == "openai_responses" else "/chat/completions"
    )
    assert requests[0].headers["Authorization"] == "Bearer local-test"
    assert not clients[0].is_closed()
    await llm.aclose()
    assert clients[0].is_closed()


@pytest.mark.parametrize(
    "protocol,path,token_key",
    [
        ("openai_chat", "$.max_tokens", "max_tokens"),
        ("openai_chat", "$.extra_body.max_tokens", "max_tokens"),
        ("openai_chat", "$.extra_body.max_completion_tokens", "max_completion_tokens"),
        ("openai_responses", "$.max_output_tokens", "max_output_tokens"),
        ("openai_responses", "$.extra_body.max_tokens", "max_tokens"),
        ("openai_responses", "$.extra_body.max_completion_tokens", "max_completion_tokens"),
    ],
)
async def test_custom_token_path_is_serialized_by_official_sdk(
    monkeypatch, protocol, path, token_key
):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=sse(stream_events(protocol), done=protocol == "openai_chat"),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    monkeypatch.setattr("my_llm.codec.openai.responses.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=ProviderProfile(
            protocol=protocol,
            api_key=parameter("api_key", "local-test"),
            max_tokens=parameter("max_tokens", 100, protocol=protocol, path=path),
        ),
        model=ModelProfile(model="demo"),
        stdio_protocol="openai_chat",
    ) as llm:
        await llm.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=50)
    assert bodies[0][token_key] == 50
    assert {key for key in bodies[0] if key.startswith("max_")} == {token_key}
    assert "extra_body" not in bodies[0]


async def test_sdk_http_errors_retain_their_type_and_preserve_client(monkeypatch):
    from openai import BadRequestError

    clients = []

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(400, json={"error": {"message": "bad"}})
                )
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    monkeypatch.setattr("my_llm.codec.openai.responses.transport.AsyncOpenAI", factory)
    llm = MyLLM(
        stdio_protocol="openai_chat",
        provider=ProviderProfile(
            api_key=parameter("api_key", "local"), max_retries=config("max_retries", 0)
        ),
        model=ModelProfile(model="demo"),
    )
    with pytest.raises(BadRequestError):
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
    assert not clients[0].is_closed()
    await llm.aclose()
    assert clients[0].is_closed()


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
async def test_inherited_parameter_defaults_and_vendor_effort_reach_official_sdk(
    monkeypatch, protocol
):
    @dataclass(frozen=True, kw_only=True)
    class VendorModel(ModelProfile):
        vendor_limit: Parameter[int | None] = Parameter(
            name="Vendor limit",
            path="$.extra_body.vendor_limit",
            default_value=None,
            phase="decode",
        )

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=sse(stream_events(protocol), done=protocol == "openai_chat"),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    monkeypatch.setattr("my_llm.codec.openai.responses.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=ProviderProfile(protocol=protocol, api_key=parameter("api_key", "local-test")),
        model=VendorModel(
            model="demo",
            supports_reasoning_effort=config("supports_reasoning_effort", True),
            reasoning_effort_mapping=config("reasoning_effort_mapping", {"low": "vendor_effort"}),
        ),
        stdio_protocol="openai_chat",
    ) as llm:
        await llm.generate(messages=[{"role": "user", "content": "hello"}], reasoning_effort="low")

    assert "vendor_limit" in bodies[0] and bodies[0]["vendor_limit"] is None
    effort = (
        bodies[0]["reasoning_effort"]
        if protocol == "openai_chat"
        else bodies[0]["reasoning"]["effort"]
    )
    assert effort == "vendor_effort"
