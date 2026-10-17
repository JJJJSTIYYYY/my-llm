"""Native SDK request serialization and stream parsing against local transports."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from google import genai
from ollama import AsyncClient

from my_llm import ModelProfile, MyLLM, ProviderProfile
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
            api_key="local-test",
            base_url="https://local.test",
            usage_mode="native",
        ),
        model=ModelProfile(model="demo", max_tokens=2048, supports_function_calling=True),
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
    assert "Sunny" in str(body)
    if protocol == "anthropic":
        assert body["max_tokens"] == 2048
        assert body["tools"][0]["input_schema"]["type"] == "object"
        assert body["tool_choice"] == options["tool_choice"]
        assert body["metadata"] == options["metadata"]
        assert requests[0].url.path == "/v1/messages"
    elif protocol == "gemini":
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
        assert body["options"]["num_predict"] == 2048
        assert requests[0].url.path == "/api/chat"
    assert all(not client.is_closed for client in http_clients)
    await llm.aclose()
    assert all(client.is_closed for client in http_clients)
