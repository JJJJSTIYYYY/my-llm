"""Verify both message formats through actual OpenAI SDK request serialization."""

import json

import pytest
from conftest import chat_event, response_completed

from my_llm import ModelProfile, ProviderProfile, generate


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
@pytest.mark.parametrize("message_format", ["responses", "chat_completions"])
@pytest.mark.parametrize("tool_format", ["responses", "chat_completions"])
async def test_real_sdk_messages_and_tools_are_detected_independently(
    sdk_factory, function_tool, tool_call, api_type, message_format, tool_format
):
    chat_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "weather?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png", "detail": "low"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
            "reasoning_content": "lookup",
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    native_history = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "weather?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.test/image.png",
                    "detail": "low",
                },
            ],
        },
        {
            "type": "reasoning",
            "id": "rs_previous",
            "summary": [{"type": "summary_text", "text": "lookup"}],
            "encrypted_content": "opaque-history",
        },
        {
            "type": "function_call",
            "id": "fc_previous",
            "call_id": "call_1",
            "name": "weather",
            "arguments": tool_call["function"]["arguments"],
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]
    history = native_history if message_format == "responses" else chat_history
    tools = (
        [{"type": "function", **function_tool["function"]}]
        if tool_format == "responses"
        else [function_tool]
    )
    events = (
        [
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg_final",
                "delta": "done",
                "sequence_number": 1,
                "logprobs": [],
            },
            response_completed(),
        ]
        if api_type == "responses"
        else [chat_event(content="done"), chat_event(finish="stop")]
    )
    sdk, requests = sdk_factory(events)
    async with sdk:
        result = await generate(
            provider=ProviderProfile(api_type=api_type),
            model=ModelProfile(model="demo", supports_function_calling=True),
            messages=history,
            tools=tools,
            client=sdk,
        )
        assert result["role"] == "assistant" and result["content"] == "done"
        body = json.loads(requests[0].content)
        if api_type == "responses":
            if message_format == "responses":
                assert body["input"] == native_history
            else:
                assert body["input"][1] == {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": tool_call["function"]["arguments"],
                }
                assert body["input"][2] == {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny",
                }
            assert body["tools"] == [{"type": "function", **function_tool["function"]}]
        else:
            assert body["messages"] == chat_history
            assert body["tools"] == [function_tool]
            assert "opaque-history" not in str(body)


async def test_real_sdk_previous_response_id_accepts_native_tool_result(sdk_factory):
    sdk, requests = sdk_factory([response_completed()])
    item = {"type": "function_call_output", "call_id": "previous_call", "output": "result"}
    async with sdk:
        await generate(
            provider=ProviderProfile(api_type="responses"),
            model="demo",
            messages=[item],
            previous_response_id="resp_previous",
            client=sdk,
        )
        body = json.loads(requests[0].content)
        assert body["input"] == [item]
        assert body["previous_response_id"] == "resp_previous"
