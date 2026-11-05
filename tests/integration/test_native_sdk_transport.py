"""Native SDK request serialization and stream parsing against local transports."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history, stream_events, tools
from tests.helpers.schema import validate_request
from tests.helpers.transport import sse


@pytest.mark.parametrize("protocol", ["anthropic"])
@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_sdk_transport_all_stdio_formats(monkeypatch, protocol, stdio, stream):
    requests, http_clients = [], []

    def handler(request):
        requests.append(request)
        validate_request(protocol, json.loads(request.content))
        events = stream_events(protocol, tool=True)
        return httpx.Response(
            200,
            text=sse(events, anthropic=protocol == "anthropic").replace("data: [DONE]\n\n", ""),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http_clients.append(http)
        return AsyncAnthropic(**kwargs, http_client=http)

    monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", factory)
    llm = MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value=protocol),
            api_key=parameter(
                "api_key",
                "local-test",
                protocol=protocol,
            ),
            base_url=config("https://local.test"),
            usage_mode=config("native"),
        ),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
            supports_function_calling=config(True),
        ),
        stdio_protocol=stdio,
    )
    options = {
        "anthropic": {"tool_choice": {"type": "auto"}, "metadata": {"user_id": "local-test"}}
    }[protocol]
    result = await llm.generate(
        messages=history(stdio), tools=tools(stdio), stream=stream, extra_body=options
    )
    if stream:
        async with result:
            events = [e async for e in result]
        assert events
        result = result.result
    assert isinstance(result, list if stdio == "openai_responses" else dict)
    body = json.loads(requests[0].content)
    assert "api_key" not in body and "default_headers" not in body
    assert "Sunny" in str(body)
    assert requests[0].headers["x-api-key"] == "local-test"
    assert body["max_tokens"] == 2048
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert body["tool_choice"] == options["tool_choice"]
    assert body["metadata"] == options["metadata"]
    assert requests[0].url.path == "/v1/messages"
    assert all(not client.is_closed for client in http_clients)
    await llm.aclose()
    assert all(client.is_closed for client in http_clients)


@pytest.mark.parametrize("protocol", ["anthropic"])
@pytest.mark.parametrize("header", ["Authorization", "X-Api-Key"])
@pytest.mark.parametrize("provider_override", [False, True])
async def test_header_authentication_without_sdk_api_key(
    monkeypatch, protocol, header, provider_override
):
    requests, client_options = [], []
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    def handler(request):
        requests.append(request)
        events = stream_events(protocol)
        return httpx.Response(
            200,
            text=sse(events, anthropic=True).replace("data: [DONE]\n\n", ""),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        client_options.append(kwargs)
        transport = httpx.MockTransport(handler)
        return AsyncAnthropic(**kwargs, http_client=httpx.AsyncClient(transport=transport))

    monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", factory)
    async with MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value=protocol),
            base_url=config("https://local.test"),
            api_key=parameter(
                "api_key", "Bearer local-header-test", path=f"$.extra_headers.{header}"
            ),
            extra_headers=config({header: "provider-override"} if provider_override else {}),
        ),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        ),
        stdio_protocol="openai_chat",
    ) as llm:
        result = await llm.generate(messages=[{"role": "user", "content": "hello"}])
        await llm.generate(
            messages=[{"role": "user", "content": "hello"}],
            extra_headers={header: "request-override"},
        )
        await llm.generate(messages=[{"role": "user", "content": "hello"}])

    assert result["role"] == "assistant"
    assert all("api_key" not in options for options in client_options)
    expected = "provider-override" if provider_override else "Bearer local-header-test"
    assert [request.headers[header] for request in requests] == [
        expected,
        "request-override",
        expected,
    ]
    assert len(client_options) == 1
    assert all("local-header-test" not in request.content.decode() for request in requests)
    if header == "Authorization":
        assert all("x-api-key" not in request.headers for request in requests)
