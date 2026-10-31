"""Independent native protocol samples shared by tests."""

from __future__ import annotations

from copy import deepcopy

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
                "content": [
                    {"logprobs": [], "type": "output_text", "text": "Checking", "annotations": []}
                ],
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
                    "id": "call_1",
                    "function": {"name": "weather", "arguments": {"city": "上海"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "tool_name": "weather", "content": "Sunny"},
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
            "access_programs": None,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "temperature": None,
            "top_p": None,
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
                "content": [
                    {"logprobs": [], "type": "output_text", "text": "Hello", "annotations": []}
                ],
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
                "part": {"logprobs": [], "type": "output_text", "text": "", "annotations": []},
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
