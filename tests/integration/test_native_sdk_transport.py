"""Exercise official native SDK serialization and streaming with local transports."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from ollama import AsyncClient

from my_llm import (
    IncompleteStreamError,
    MessageChunk,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    generate,
)

PROTOCOLS = ("anthropic", "gemini", "ollama")


def native_events(protocol, *, calls=False):
    if protocol == "anthropic":
        blocks = [
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "text", "text": ""},
        ]
        if calls:
            blocks += [
                {"type": "tool_use", "id": f"call_{i}", "name": "weather", "input": {}}
                for i in range(2)
            ]
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "demo",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2, "output_tokens": 0, "cache_read_input_tokens": 1},
                },
            }
        ]
        for index, block in enumerate(blocks):
            events.append({"type": "content_block_start", "index": index, "content_block": block})
            if block["type"] == "thinking":
                deltas = [{"type": "thinking_delta", "thinking": s} for s in ["思", "考"]]
                deltas += [{"type": "signature_delta", "signature": "signed"}]
            elif block["type"] == "text":
                deltas = [{"type": "text_delta", "text": s} for s in ["你", "好"]]
            else:
                deltas = [
                    {"type": "input_json_delta", "partial_json": s} for s in ['{"city":', '"上海"}']
                ]
            events += [{"type": "content_block_delta", "index": index, "delta": d} for d in deltas]
            events.append({"type": "content_block_stop", "index": index})
        events += [
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if calls else "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
        return events
    if protocol == "gemini":
        parts = [
            {"text": "思", "thought": True},
            {"text": "考", "thought": True},
            {"text": "你"},
            {"text": "好", "thoughtSignature": "c2lnbmVk"},
        ]
        if calls:
            parts += [
                {"functionCall": {"name": "weather", "args": {"city": "上海"}}} for _ in range(2)
            ]
        events = [
            {"candidates": [{"index": 0, "content": {"role": "model", "parts": [p]}}]}
            for p in parts
        ]
        events += [
            {
                "candidates": [{"index": 0, "finishReason": "STOP"}],
                "responseId": "gen_1",
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 5,
                    "thoughtsTokenCount": 1,
                },
            }
        ]
        return events
    events = [
        {
            "model": "demo",
            "message": {"role": "assistant", "content": "你", "thinking": "思"},
            "done": False,
        },
        {
            "model": "demo",
            "message": {"role": "assistant", "content": "好", "thinking": "考"},
            "done": False,
        },
    ]
    if calls:
        events += [
            {
                "model": "demo",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "weather", "arguments": {"city": "上海"}}}
                        for _ in range(2)
                    ],
                },
                "done": False,
            }
        ]
    return events + [
        {
            "model": "demo",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 2,
            "eval_count": 3,
        }
    ]


@pytest.fixture
def native_sdk_factory():
    def make(protocol, events, *, status=200):
        requests = []

        def handler(request):
            requests.append(request)
            if status != 200:
                return httpx.Response(
                    status,
                    json={
                        "error": {
                            "code": status,
                            "message": "failure",
                            "type": "authentication_error",
                        }
                    },
                )
            if protocol == "ollama":
                text = "\n".join(json.dumps(e) for e in events) + "\n"
                return httpx.Response(
                    200, text=text, headers={"content-type": "application/x-ndjson"}
                )
            text = "".join(
                (f"event: {e['type']}\n" if protocol == "anthropic" else "")
                + f"data: {json.dumps(e)}\n\n"
                for e in events
            )
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(transport=transport)
        if protocol == "anthropic":
            client = AsyncAnthropic(
                api_key="local", base_url="https://native.test", max_retries=0, http_client=http
            )
        elif protocol == "gemini":
            client = genai.Client(
                api_key="local",
                http_options=types.HttpOptions(
                    base_url="https://native.test",
                    httpx_async_client=http,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        else:
            # Ollama constructs its own SDK-compatible HTTP client using these official kwargs.
            client = AsyncClient(host="https://native.test", transport=transport)
        return client, requests

    return make


async def close_client(protocol, client):
    if protocol == "gemini":
        await client.aio.aclose()
        client.close()
    else:
        await client.close()


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [None, False, True])
async def test_native_sdk_content_reasoning_usage_and_omitted_fields(
    native_sdk_factory, protocol, stream
):
    client, requests = native_sdk_factory(protocol, native_events(protocol))
    provider = ProviderProfile(
        protocol=protocol,
        usage_mode="native",
        extra_body={
            "stream_options": {"include_usage": True},
            "reasoning_effort": "nonsense",
            "not_a_native_parameter": True,
        },
    )
    try:
        result = await generate(
            provider=provider,
            model=ModelProfile(model="demo", max_tokens=2048),
            messages=[{"role": "user", "content": "hello", "name": "ignored"}],
            reasoning=True,
            stream=stream,
            client=client,
        )
        if stream:
            combined = MessageChunk()
            async with result:
                async for chunk in result:
                    combined += chunk
            result = combined.to_message(require_finished=True)
        assert result["content"] == "你好"
        assert result["reasoning_content"] == "思考"
        assert result["finish_reason"] == "stop"
        body = json.loads(requests[0].content)
        assert "stream_options" not in body and "reasoning_effort" not in body
        assert "not_a_native_parameter" not in body
        if protocol == "anthropic":
            assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
            assert body["max_tokens"] == 2048
            assert body["messages"] == [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ]
            assert result["usage"]["input_tokens"] == 2 and result["usage"]["output_tokens"] == 3
            assert requests[0].url.path == "/v1/messages"
        elif protocol == "gemini":
            assert body["generationConfig"]["thinkingConfig"] == {
                "include_thoughts": True,
                "thinking_budget": -1,
            }
            assert body["generationConfig"]["maxOutputTokens"] == 2048
            assert result["usage"]["total_token_count"] == 5
            assert requests[0].url.path == "/v1beta/models/demo:streamGenerateContent"
        else:
            assert body["think"] is True and body["options"]["num_predict"] == 2048
            assert result["usage"]["prompt_eval_count"] == 2
            assert requests[0].url.path == "/api/chat"
    finally:
        await close_client(protocol, client)


@pytest.mark.parametrize("stream", [None, False, True])
@pytest.mark.parametrize(
    "protocol,reason,calls,expected",
    [
        ("anthropic", "end_turn", False, "stop"),
        ("anthropic", "stop_sequence", False, "stop"),
        ("anthropic", "max_tokens", False, "length"),
        ("anthropic", "model_context_window_exceeded", False, "length"),
        ("anthropic", "tool_use", True, "tool_calls"),
        ("anthropic", "refusal", False, "content_filter"),
        ("anthropic", "pause_turn", False, "pause_turn"),
        ("gemini", "STOP", False, "stop"),
        ("gemini", "STOP", True, "tool_calls"),
        ("gemini", "MAX_TOKENS", False, "length"),
        ("ollama", "stop", False, "stop"),
        ("ollama", "stop", True, "tool_calls"),
        ("ollama", "length", False, "length"),
    ],
)
async def test_native_sdk_normalized_finish_reasons(
    native_sdk_factory, protocol, reason, calls, expected, stream
):
    events = native_events(protocol, calls=calls)
    if protocol == "anthropic":
        events[-2]["delta"]["stop_reason"] = reason
    elif protocol == "gemini":
        events[-1]["candidates"][0]["finishReason"] = reason
    else:
        events[-1]["done_reason"] = reason
    client, _ = native_sdk_factory(protocol, events)
    llm = MyLLM(
        provider=ProviderProfile(protocol=protocol),
        model=ModelProfile(model="demo", max_tokens=2048),
        client=client,
    )
    try:
        result = await llm.generate(messages=[{"role": "user", "content": "hello"}], stream=stream)
        if stream:
            combined = MessageChunk()
            async with result:
                async for chunk in result:
                    assert chunk.finish_reason in (None, expected)
                    combined += chunk
            result = combined.to_message(require_finished=True)
        assert result["finish_reason"] == expected
        assert bool(result.get("tool_calls")) is calls
        assert result["content"] == "你好"
    finally:
        await close_client(protocol, client)


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_sdk_multiple_tools_and_signed_history_roundtrip(
    native_sdk_factory, protocol, function_tool
):
    events = native_events(protocol, calls=True)
    client, requests = native_sdk_factory(protocol, events)
    provider = ProviderProfile(protocol=protocol)
    model = ModelProfile(model="demo", max_tokens=2048, supports_function_calling=True)
    history = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "weather?"}]
    try:
        result = await generate(
            provider=provider, model=model, messages=history, tools=[function_tool], client=client
        )
        calls = result["tool_calls"]
        assert result["finish_reason"] == "tool_calls"
        assert len(calls) == 2 and len({c["id"] for c in calls}) == 2
        assert all(json.loads(c["function"]["arguments"]) == {"city": "上海"} for c in calls)
        assert "usage" not in result
        history += [
            result,
            *[{"role": "tool", "tool_call_id": c["id"], "content": "sunny"} for c in calls],
        ]
        events[:] = native_events(protocol)
        final = await generate(
            provider=provider, model=model, messages=history, tools=[function_tool], client=client
        )
        assert final["content"] == "你好"
        body = json.loads(requests[-1].content)
        if protocol == "anthropic":
            assert body["system"] == "Be helpful"
            assert body["messages"][1]["content"][0] == {
                "type": "thinking",
                "thinking": "思考",
                "signature": "signed",
            }
            assert [b["tool_use_id"] for b in body["messages"][-1]["content"]] == [
                c["id"] for c in calls
            ]
            assert body["tools"][0]["input_schema"] == function_tool["function"]["parameters"]
        elif protocol == "gemini":
            parts = body["contents"][1]["parts"]
            assert any(p.get("thoughtSignature") == "c2lnbmVk" for p in parts)
            assert [p["functionResponse"]["id"] for p in body["contents"][-1]["parts"]] == [
                c["id"] for c in calls
            ]
            assert all(
                p["functionResponse"]["name"] == "weather" for p in body["contents"][-1]["parts"]
            )
            assert list(body["tools"][0]) == ["functionDeclarations"]
        else:
            assert body["messages"][-1]["tool_name"] == "weather"
            assert body["messages"][2]["thinking"] == "思考"
            assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"city": "上海"}
        assert len(requests) == 2  # No provider tools or automatic execution.
    finally:
        await close_client(protocol, client)


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_truncated_stream_retains_partial_message(native_sdk_factory, protocol):
    events = native_events(protocol)[:-1]
    client, _ = native_sdk_factory(protocol, events)
    try:
        with pytest.raises(IncompleteStreamError) as exc:
            await generate(
                provider=ProviderProfile(protocol=protocol),
                model=ModelProfile(model="demo", max_tokens=2048),
                messages=[{"role": "user", "content": "hello"}],
                client=client,
            )
        assert exc.value.partial_message["content"] == "你好"
        assert exc.value.partial_message["reasoning_content"] == "思考"
    finally:
        await close_client(protocol, client)


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_stream_early_close_and_parallel_state(native_sdk_factory, protocol):
    import asyncio

    client, requests = native_sdk_factory(protocol, native_events(protocol))
    llm = MyLLM(
        provider=ProviderProfile(protocol=protocol),
        model=ModelProfile(model="demo", max_tokens=2048),
        client=client,
    )
    try:
        result = await llm.generate(messages=[{"role": "user", "content": "x"}], stream=True)
        assert not requests
        async with result:
            await anext(result)
        with pytest.raises(StopAsyncIteration):
            await anext(result)
        results = await asyncio.gather(
            *[llm.generate(messages=[{"role": "user", "content": "x"}]) for _ in range(3)]
        )
        assert all(r["content"] == "你好" for r in results)
    finally:
        await close_client(protocol, client)


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_owned_client_mapping_and_cleanup(native_sdk_factory, monkeypatch, protocol):
    client, requests = native_sdk_factory(protocol, native_events(protocol))
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return client

    monkeypatch.setattr(
        {
            "anthropic": "my_llm.invoke.AsyncAnthropic",
            "gemini": "my_llm.invoke.genai.Client",
            "ollama": "my_llm.invoke.AsyncOllama",
        }[protocol],
        factory,
    )
    result = await generate(
        provider=ProviderProfile(
            protocol=protocol,
            base_url="https://native.test",
            api_key="local",
            timeout=12.5,
            max_retries=0,
            extra_headers={"x-profile": "yes"},
        ),
        model=ModelProfile(model="demo", max_tokens=2048),
        messages=[{"role": "user", "content": "x"}],
    )
    assert result["content"] == "你好" and len(requests) == 1
    kwargs = captured[0]
    if protocol == "anthropic":
        assert kwargs == {
            "api_key": "local",
            "base_url": "https://native.test",
            "timeout": 12.5,
            "max_retries": 0,
        }
        assert client.is_closed()
    elif protocol == "gemini":
        assert kwargs["api_key"] == "local"
        assert kwargs["http_options"].timeout == 12500
        assert kwargs["http_options"].retry_options.attempts == 1
        client.close()
    else:
        assert kwargs == {
            "host": "https://native.test",
            "timeout": 12.5,
            "headers": {"Authorization": "Bearer local", "x-profile": "yes"},
        }
        assert client._client.is_closed


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_native_sdk_authentication_errors_remain_native(native_sdk_factory, protocol):
    from anthropic import AuthenticationError
    from google.genai.errors import ClientError
    from ollama import ResponseError

    client, _ = native_sdk_factory(protocol, [], status=401)
    try:
        with pytest.raises(
            {"anthropic": AuthenticationError, "gemini": ClientError, "ollama": ResponseError}[
                protocol
            ]
        ):
            await generate(
                provider=ProviderProfile(protocol=protocol),
                model=ModelProfile(model="demo", max_tokens=2048),
                messages=[{"role": "user", "content": "x"}],
                client=client,
            )
    finally:
        await close_client(protocol, client)
