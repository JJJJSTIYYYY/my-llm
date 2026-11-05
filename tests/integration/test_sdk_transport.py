"""Official OpenAI SDK serialization and SSE parsing, without public client injection."""

import json
from copy import deepcopy
from dataclasses import dataclass, field

import httpx
import pytest
from openai import AsyncOpenAI

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile
from my_llm.profile.types import InputParameter
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history, stream_events, tools
from tests.helpers.schema import validate_request
from tests.helpers.transport import sse


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic"])
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
            protocol=Config(default_value=protocol),
            api_key=parameter("api_key", "local-test"),
            base_url=config("https://local.test/v1"),
            usage_mode=config("native" if protocol == "openai_responses" else "stream_options"),
        ),
        model=ModelProfile(
            model=Attribute(default_value="demo"), supports_function_calling=config(True)
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


@pytest.mark.parametrize("usage_mode", ["none", "stream_options"])
@pytest.mark.parametrize("body_source", ["provider", "call"])
@pytest.mark.parametrize(
    "stream_options",
    [None, {"include_usage": False}, {"include_usage": False, "include_obfuscation": False}],
)
async def test_chat_generated_usage_overrides_sdk_body_overlay(
    monkeypatch, usage_mode, body_source, stream_options
):
    bodies = []
    extra_body = {"stream_options": deepcopy(stream_options), "temperature": 0.25}
    original_body = deepcopy(extra_body)

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=sse(stream_events("openai_chat")),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value="openai_chat"),
            api_key=parameter("api_key", "local-test"),
            usage_mode=config(usage_mode),
            extra_body=config(extra_body) if body_source == "provider" else None,
        ),
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
    ) as llm:
        await llm.generate(
            messages=[{"role": "user", "content": "hi"}],
            **({"extra_body": extra_body} if body_source == "call" else {}),
        )

    expected = (
        {**(stream_options or {}), "include_usage": True}
        if usage_mode == "stream_options"
        else stream_options
    )
    assert bodies[0]["stream_options"] == expected
    assert bodies[0]["temperature"] == 0.25
    assert extra_body == original_body


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
            protocol=Config(default_value=protocol),
            api_key=parameter("api_key", "local-test"),
            max_tokens=parameter("max_tokens", 100, protocol=protocol, path=path),
        ),
        model=ModelProfile(model=Attribute(default_value="demo")),
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
            protocol=Config(default_value="openai_chat"),
            api_key=parameter("api_key", "local"),
            max_retries=config(0),
        ),
        model=ModelProfile(model=Attribute(default_value="demo")),
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
    @dataclass(kw_only=True)
    class VendorModel(ModelProfile):
        vendor_limit: InputParameter[int | None] = field(
            default_factory=lambda: InputParameter(
                path="$.extra_body.vendor_limit",
                default_value=None,
            ),
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
        provider=ProviderProfile(
            protocol=Config(default_value=protocol), api_key=parameter("api_key", "local-test")
        ),
        model=VendorModel(
            model=Attribute(default_value="demo"),
            reasoning_effort=parameter("reasoning_effort", supports=True, protocol=protocol),
            reasoning_effort_mapping=config({"low": "vendor_effort"}),
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


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", ["content", "reasoning_content"])
async def test_output_event_type_through_official_sdk_sse(monkeypatch, protocol, stream, name):
    from my_llm import Message, OutputParameter, ReasoningContent, TextContent
    from tests.helpers.protocols import chat_event, response_completed

    selected_type = "response.vendor.delta"
    native_type = (
        "response.output_text.delta"
        if name == "content"
        else "response.reasoning_summary_text.delta"
    )
    events = [
        {"type": native_type, "vendor": {"value": "ignored"}},
        {"type": selected_type, "vendor": {"value": "custom "}},
        {"type": selected_type, "vendor": {"value": "value"}},
    ]
    if protocol == "openai_chat":
        events.append(chat_event(finish="stop"))
    else:
        item = (
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "custom value", "annotations": []},
                ],
            }
            if name == "content"
            else {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "custom value"},
                ],
            }
        )
        events.append(response_completed([item]))

    def handler(request):
        return httpx.Response(
            200,
            text=sse(events, done=protocol == "openai_chat"),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    module = "chat" if protocol == "openai_chat" else "responses"
    monkeypatch.setattr(f"my_llm.codec.openai.{module}.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value=protocol),
            api_key=InputParameter(path="$.api_key", default_value="local-test"),
        ),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            **{
                name: OutputParameter(path="$.vendor.value", event_type=selected_type),
            },
        ),
        stdio_protocol="middleware",
    ) as llm:
        result = await llm.generate(messages=[Message("user", [TextContent("hi")])], stream=stream)
        if stream:
            async with result:
                public = [event async for event in result]
            assert [event.value for event in public if event.type == "delta"] == [
                "custom ",
                "value",
            ]
            result = result.result
        block = next(
            block
            for block in result.content
            if isinstance(
                block,
                TextContent if name == "content" else ReasoningContent,
            )
        )
        assert (block.text if name == "content" else block.reasoning) == "custom value"
