from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from my_llm import ModelProfile, MyCodec, ProviderProfile, resolve_profiles

PROTOCOLS = ("openai_chat", "openai_responses", "anthropic", "gemini", "ollama")
CHAT_HISTORY = [
    {"role": "user", "content": "Weather?"},
    {
        "role": "assistant",
        "content": "Checking",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city":"上海"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
]
CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "strict": False,
    },
}


# These native fixtures are written independently of MyCodec. Generating them
# through the converter would let a shared encoding/decoding bug validate itself.
def history(protocol):
    if protocol == "openai_chat":
        return deepcopy(CHAT_HISTORY)
    if protocol == "openai_responses":
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Weather?"}],
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Checking", "annotations": []}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": '{"city":"上海"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
        ]
    if protocol == "anthropic":
        return [
            {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "weather",
                        "input": {"city": "上海"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny"}],
            },
        ]
    if protocol == "gemini":
        return [
            {"role": "user", "parts": [{"text": "Weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "Checking"},
                    {
                        "function_call": {
                            "id": "call_1",
                            "name": "weather",
                            "args": {"city": "上海"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call_1",
                            "name": "weather",
                            "response": {"result": "Sunny"},
                        }
                    }
                ],
            },
        ]
    return [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [
                {
                    "function": {"name": "weather", "arguments": {"city": "上海"}},
                }
            ],
        },
        {"role": "tool", "tool_name": "weather", "content": "Sunny"},
    ]


def tools(protocol):
    function = deepcopy(CHAT_TOOL["function"])
    if protocol == "openai_chat":
        return [deepcopy(CHAT_TOOL)]
    if protocol == "openai_responses":
        return [{"type": "function", **function}]
    if protocol == "anthropic":
        return [
            {
                "name": "weather",
                "description": "Weather",
                "strict": False,
                "input_schema": function["parameters"],
            }
        ]
    if protocol == "gemini":
        return [
            {
                "function_declarations": [
                    {
                        "name": "weather",
                        "description": "Weather",
                        "parameters_json_schema": function["parameters"],
                    }
                ]
            }
        ]
    function.pop("strict")
    return [{"type": "function", "function": function}]


def chat_event(*, content=None, reasoning=None, calls=None, finish=None, usage=None):
    """Build a chunk; reasoning_content is a compatible-provider extension."""
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
            "tool_choice": "auto",
            "tools": [],
        },
    }


def stream_events(protocol, *, tool=False):
    call = CHAT_HISTORY[1]["tool_calls"][0]
    if protocol == "openai_chat":
        return [
            {
                **chat_event(content="Hello"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            *(
                [
                    chat_event(
                        calls=[
                            {
                                "index": 0,
                                **call,
                                "function": {"name": "weather", "arguments": '{"city":'},
                            }
                        ]
                    ),
                    chat_event(calls=[{"index": 0, "function": {"arguments": '"上海"}'}}]),
                ]
                if tool
                else []
            ),
            chat_event(finish="tool_calls" if tool else "stop"),
            {
                **chat_event(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
                "choices": [],
            },
        ]
    if protocol == "openai_responses":
        output = [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ]
        if tool:
            output.append(
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "status": "completed",
                    "name": "weather",
                    "arguments": '{"city":"上海"}',
                }
            )
        response = response_completed()["response"]
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**output[0], "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
                "logprobs": [],
            },
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Hello",
                "logprobs": [],
            },
            {
                "type": "response.content_part.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": output[0]["content"][0],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": output[0]},
        ]
        if tool:
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {**output[1], "arguments": "", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "output_index": 1,
                        "delta": '{"city":',
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "output_index": 1,
                        "delta": '"上海"}',
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": "fc_1",
                        "output_index": 1,
                        "arguments": output[1]["arguments"],
                        "name": "weather",
                    },
                    {"type": "response.output_item.done", "output_index": 1, "item": output[1]},
                ]
            )
        events.append(
            response_completed(
                output,
                {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3,
                },
            )
        )
        return [{**event, "sequence_number": index} for index, event in enumerate(events)]
    if protocol == "anthropic":
        result = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "demo",
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {"type": "content_block_stop", "index": 0},
        ]
        if tool:
            result.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "weather",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"city":"上海"}'},
                    },
                    {"type": "content_block_stop", "index": 1},
                ]
            )
        return [
            *result,
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
    if protocol == "gemini":
        parts = [{"text": "Hello"}]
        if tool:
            parts.append(
                {
                    "function_call": {"id": "call_1", "name": "weather", "args": {"city": "上海"}},
                    "thought_signature": "c2lnbmVk",
                }
            )
        return [
            {
                "response_id": "gem_1",
                "candidates": [{"index": 0, "content": {"role": "model", "parts": parts}}],
            },
            {
                "candidates": [{"index": 0, "finish_reason": "STOP"}],
                "usage_metadata": {
                    "prompt_token_count": 1,
                    "candidates_token_count": 2,
                    "total_token_count": 3,
                },
            },
        ]
    return [
        {
            "model": "demo",
            "created_at": "2026-10-17T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello",
                **(
                    {
                        "tool_calls": [
                            {
                                "function": {"name": "weather", "arguments": {"city": "上海"}},
                            }
                        ]
                    }
                    if tool
                    else {}
                ),
            },
            "done": False,
        },
        {
            "model": "demo",
            "created_at": "2026-10-17T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 2,
        },
    ]


@pytest.fixture
def profile_factory():
    def make(protocol="openai_chat", *, provider=None, **model):
        return resolve_profiles(
            provider or ProviderProfile(protocol=protocol), ModelProfile(model="demo", **model)
        )

    return make


@pytest.fixture
def function_tool():
    return deepcopy(CHAT_TOOL)


@pytest.fixture
def tool_call():
    return deepcopy(CHAT_HISTORY[1]["tool_calls"][0])


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

    aclose = close


class FakeClient:
    def __init__(self, events, *, failure=None):
        self.upstream = FakeStream(events, failure=failure)
        self.closed = False
        self.requests = []
        self.responses = SimpleNamespace(create=self.request)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))
        self.messages = SimpleNamespace(create=self.request)

    async def request(self, **kwargs):
        self.requests.append(kwargs)
        return self.upstream

    async def close(self):
        self.closed = True


def sse(events, *, anthropic=False, done=True):
    return "".join(
        (f"event: {e['type']}\n" if anthropic or e.get("type", "").startswith("response.") else "")
        + f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        for e in events
    ) + ("" if anthropic or not done else "data: [DONE]\n\n")


def encode_request(profile, messages=None, tools=None, **parameters):
    codec = MyCodec()
    intermediate = codec.encode(
        messages or [{"role": "user", "content": "x"}], input_format="openai_chat"
    )
    if tools is not None and not isinstance(tools, __import__("openai").Omit):
        intermediate.update(
            codec.encode(tools=tools, input_format=codec.match_protocol(tools) or "openai_chat")
        )
    return codec.request(intermediate, profile, **parameters)
