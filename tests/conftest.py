from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from my_llm import ModelProfile, ProviderProfile
from my_llm.profiles import resolve_profiles


@pytest.fixture
def function_tool():
    return {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "strict": False,
        },
    }


@pytest.fixture
def tool_call():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"上海"}'},
    }


@pytest.fixture
def profile_factory():
    def make(api_type="chat_completions", *, provider=None, **model):
        return resolve_profiles(
            provider or ProviderProfile(api_type=api_type), ModelProfile(model="demo", **model)
        )

    return make


def chat_event(*, content=None, reasoning=None, calls=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if calls is not None:
        delta["tool_calls"] = calls
    return {
        "id": "chat_1",
        "created": 1,
        "model": "demo",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **({"usage": usage} if usage is not None else {}),
    }


def response_completed(output=None, usage=None):
    return {
        "type": "response.completed",
        "sequence_number": 10,
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "demo",
            "status": "completed",
            "output": output or [],
            "usage": usage,
            "parallel_tool_calls": True,
            "tools": [],
            "tool_choice": "auto",
        },
    }


@pytest.fixture
def sdk_factory():
    clients = []

    def make(events, *, status=200):
        requests = []

        def handler(request):
            requests.append(request)
            if status != 200:
                return httpx.Response(
                    status,
                    json={"error": {"message": "test error", "type": "test", "code": "test"}},
                )
            body = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
            body += "data: [DONE]\n\n"
            return httpx.Response(
                200, content=body.encode(), headers={"Content-Type": "text/event-stream"}
            )

        client = AsyncOpenAI(
            api_key="local-test-only",
            base_url="https://local.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client, requests

    yield make
    for client in clients:
        # Individual async tests close SDK clients in async context managers.
        assert client.is_closed()


class FakeStream:
    def __init__(self, events, *, failure=None):
        self.events = iter(events)
        self.failure = failure
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.failure is not None:
                raise self.failure
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, events, *, failure=None):
        self.upstream = FakeStream(events, failure=failure)
        self.closed = False
        self.requests = []
        self.responses = SimpleNamespace(create=self.request)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))

    async def request(self, **kwargs):
        self.requests.append(kwargs)
        return self.upstream

    async def close(self):
        self.closed = True
