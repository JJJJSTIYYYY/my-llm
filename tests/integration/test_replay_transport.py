"""Two real-SDK requests replay encrypted thinking through persisted Chat history."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic

from my_llm import ModelProfile, MyLLM, ProviderProfile
from my_llm.codec.shared.conversion import REPLAY
from tests.helpers.protocols import stream_events
from tests.helpers.schema import validate_request
from tests.helpers.transport import sse


@pytest.mark.parametrize("stream", [False, True])
async def test_signed_tool_round_trip_uses_only_supported_wire_fields(monkeypatch, stream):
    requests, clients = [], []
    start = stream_events("anthropic")[0]
    thinking = {"type": "thinking", "thinking": "summary", "signature": "signature-payload"}
    redacted = {"type": "redacted_thinking", "data": "encrypted-payload"}
    call = {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "上海"}}
    events = [
        start,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "",
                "future_block": {"unknown": True},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "summary"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signature-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "payload"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": redacted},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {**call, "input": {}}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":"上海"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]

    def handler(request):
        body = json.loads(request.content)
        validate_request("anthropic", body)
        assert REPLAY not in request.content.decode()
        assert "future_block" not in request.content.decode()
        requests.append(body)
        response = events if len(requests) == 1 else stream_events("anthropic")
        return httpx.Response(
            200, text=sse(response, anthropic=True), headers={"Content-Type": "text/event-stream"}
        )

    def factory(**kwargs):
        client = AsyncAnthropic(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", factory)
    async with MyLLM(
        provider=ProviderProfile(protocol="anthropic", api_key="local-test"),
        model=ModelProfile(model="demo", max_tokens=4096, supports_function_calling=True),
        stdio_protocol="openai_chat",
    ) as llm:
        history = [{"role": "user", "content": "weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        response = await llm.generate(messages=history, tools=tools, reasoning=True, stream=stream)
        if stream:
            async with response:
                async for _ in response:
                    pass
            response = response.result
        assert "encrypted-payload" in json.dumps(response)
        assert "future_block" in json.dumps(response)
        history.append(json.loads(json.dumps(response)))
        history.append(
            {"role": "tool", "tool_call_id": response["tool_calls"][0]["id"], "content": "Sunny"}
        )
        final = await llm.generate(messages=history, tools=tools, reasoning=True)
        assert final["content"] == "Hello"
    assert requests[1]["messages"][1]["content"] == [thinking, redacted, call]
    assert requests[1]["messages"][2]["content"][0]["tool_use_id"] == "call_1"
    assert len(clients) == 1 and clients[0].is_closed()
