"""Official OpenAI SDK serialization and SSE parsing, without public client injection."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from my_llm import ModelProfile, MyLLM, ProviderProfile
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
            api_key="local-test",
            base_url="https://local.test/v1",
            usage_mode="native" if protocol == "openai_responses" else "stream_options",
        ),
        model=ModelProfile(model="demo", supports_function_calling=True),
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
    assert not clients[0].is_closed()
    await llm.aclose()
    assert clients[0].is_closed()


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
        provider=ProviderProfile(api_key="local", max_retries=0), model=ModelProfile(model="demo")
    )
    with pytest.raises(BadRequestError):
        await llm.generate(messages=[{"role": "user", "content": "hi"}])
    assert not clients[0].is_closed()
    await llm.aclose()
    assert clients[0].is_closed()
