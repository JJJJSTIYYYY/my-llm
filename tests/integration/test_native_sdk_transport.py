"""Native SDK request serialization and stream parsing against local transports."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from google import genai
from ollama import AsyncClient

from my_llm import ModelProfile, MyLLM, ProviderProfile
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history, stream_events, tools
from tests.helpers.schema import validate_request
from tests.helpers.transport import sse

GenAIClient = genai.Client


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
@pytest.mark.parametrize(
    "stdio", ["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"]
)
@pytest.mark.parametrize("stream", [False, True])
async def test_native_sdk_transport_all_stdio_formats(monkeypatch, protocol, stdio, stream):
    requests, http_clients = [], []

    def handler(request):
        requests.append(request)
        validate_request(protocol, json.loads(request.content))
        events = stream_events(protocol, tool=True)
        if protocol == "gemini":
            aliases = {
                "response_id": "responseId",
                "function_call": "functionCall",
                "thought_signature": "thoughtSignature",
                "finish_reason": "finishReason",
                "usage_metadata": "usageMetadata",
                "total_token_count": "totalTokenCount",
            }

            def wire(value):
                if isinstance(value, list):
                    return [wire(v) for v in value]
                if isinstance(value, dict):
                    return {aliases.get(k, k): wire(v) for k, v in value.items()}
                return value

            events = wire(events)
        if protocol == "ollama":
            body = "\n".join(json.dumps(e) for e in events) + "\n"
            return httpx.Response(200, text=body, headers={"Content-Type": "application/x-ndjson"})
        return httpx.Response(
            200,
            text=sse(events, anthropic=protocol == "anthropic").replace("data: [DONE]\n\n", ""),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        if protocol == "anthropic":
            http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            http_clients.append(http)
            return AsyncAnthropic(**kwargs, http_client=http)
        options = kwargs.pop("http_options")
        options = options.model_copy(
            update={
                "async_client_args": {
                    **(options.async_client_args or {}),
                    "transport": httpx.MockTransport(handler),
                }
            }
        )
        client = GenAIClient(**kwargs, http_options=options)
        http_clients.append(client._api_client._async_httpx_client)
        return client

    if protocol == "ollama":

        def ollama_factory(**kwargs):
            client = AsyncClient(**kwargs, transport=httpx.MockTransport(handler))
            http_clients.append(client._client)
            return client

        monkeypatch.setattr("my_llm.codec.ollama.transport.AsyncOllama", ollama_factory)
    elif protocol == "anthropic":
        monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", factory)
    else:
        monkeypatch.setattr("my_llm.codec.gemini.transport.genai.Client", factory)
    llm = MyLLM(
        provider=ProviderProfile(
            protocol=protocol,
            api_key=parameter(
                "api_key",
                "Bearer local-test" if protocol == "ollama" else "local-test",
                protocol=protocol,
            ),
            base_url=config("base_url", "https://local.test"),
            usage_mode=config("usage_mode", "native"),
        ),
        model=ModelProfile(
            model="demo",
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
            supports_function_calling=config("supports_function_calling", True),
        ),
        stdio_protocol=stdio,
    )
    options = {
        "anthropic": {"tool_choice": {"type": "auto"}, "metadata": {"user_id": "local-test"}},
        "gemini": {
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            "safety_settings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}
            ],
        },
        "ollama": {},
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
    if protocol == "anthropic":
        assert requests[0].headers["x-api-key"] == "local-test"
        assert body["max_tokens"] == 2048
        assert body["tools"][0]["input_schema"]["type"] == "object"
        assert body["tool_choice"] == options["tool_choice"]
        assert body["metadata"] == options["metadata"]
        assert requests[0].url.path == "/v1/messages"
    elif protocol == "gemini":
        assert requests[0].headers["x-goog-api-key"] == "local-test"
        assert body["generationConfig"]["maxOutputTokens"] == 2048
        assert "streamGenerateContent" in str(requests[0].url)
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
        assert body["safetySettings"] == options["safety_settings"]
        calls = [
            part
            for message in body["contents"]
            for part in message["parts"]
            if "functionCall" in part
        ]
        assert calls
        if stdio != "gemini":
            assert calls[0]["thoughtSignature"] == "skip_thought_signature_validator"
    else:
        assert requests[0].headers["Authorization"] == "Bearer local-test"
        assert body["options"]["num_predict"] == 2048
        assert requests[0].url.path == "/api/chat"
    assert all(not client.is_closed for client in http_clients)
    await llm.aclose()
    assert all(client.is_closed for client in http_clients)


@pytest.mark.parametrize("protocol", ["anthropic", "ollama"])
@pytest.mark.parametrize("header", ["Authorization", "X-Api-Key"])
@pytest.mark.parametrize("provider_override", [False, True])
async def test_header_authentication_without_sdk_api_key(
    monkeypatch, protocol, header, provider_override
):
    requests, client_options = [], []
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def handler(request):
        requests.append(request)
        events = stream_events(protocol)
        if protocol == "ollama":
            return httpx.Response(
                200,
                text="\n".join(json.dumps(event) for event in events) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
        return httpx.Response(
            200,
            text=sse(events, anthropic=True).replace("data: [DONE]\n\n", ""),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        client_options.append(kwargs)
        transport = httpx.MockTransport(handler)
        if protocol == "anthropic":
            return AsyncAnthropic(**kwargs, http_client=httpx.AsyncClient(transport=transport))
        return AsyncClient(**kwargs, transport=transport)

    target = (
        "my_llm.codec.anthropic.transport.AsyncAnthropic"
        if protocol == "anthropic"
        else "my_llm.codec.ollama.transport.AsyncOllama"
    )
    monkeypatch.setattr(target, factory)
    async with MyLLM(
        provider=ProviderProfile(
            protocol=protocol,
            base_url=config("base_url", "https://local.test"),
            api_key=parameter(
                "api_key", "Bearer local-header-test", path=f"$.extra_headers.{header}"
            ),
            extra_headers=config(
                "extra_headers", {header: "provider-override"} if provider_override else {}
            ),
        ),
        model=ModelProfile(
            model="demo", max_tokens=parameter("max_tokens", 2048, protocol=protocol)
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
        expected, "request-override", expected
    ]
    assert len(client_options) == (2 if protocol == "ollama" else 1)
    assert all("local-header-test" not in request.content.decode() for request in requests)
    if header == "Authorization":
        assert all("x-api-key" not in request.headers for request in requests)
