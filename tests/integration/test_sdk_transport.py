"""Use the actual AsyncOpenAI SDK with an in-process HTTP/SSE transport."""

import json

import pytest
from conftest import chat_event, response_completed
from openai import AuthenticationError

from my_llm import MessageChunk, ModelProfile, MyLLM, ProviderProfile, ResponseFailedError, generate


@pytest.mark.parametrize("stream", [None, False, True])
async def test_actual_chat_sdk_wire_request_and_modes(sdk_factory, stream):
    sdk, requests = sdk_factory(
        [
            chat_event(content="你", reasoning="思"),
            chat_event(content="好", reasoning="考"),
            chat_event(finish="stop"),
        ]
    )
    async with sdk:
        result = await generate(
            provider=ProviderProfile(),
            model="demo",
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            client=sdk,
            max_tokens=32,
            extra_headers={"x-local-test": "yes"},
        )
        if stream:
            from my_llm import MessageChunk

            combined = MessageChunk()
            async with result:
                async for chunk in result:
                    combined += chunk
            result = combined.to_message(require_finished=True)
        assert result["content"] == "你好" and result["reasoning_content"] == "思考"
        assert result["finish_reason"] == "stop"
        body = json.loads(requests[0].content)
        assert body["stream"] is True
        assert body["max_completion_tokens"] == 32
        assert requests[0].url.path == "/v1/chat/completions"
        assert requests[0].headers["x-local-test"] == "yes"


async def test_actual_responses_sdk_tool_wire_format_and_replay(sdk_factory, function_tool):
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "weather",
        "arguments": "",
        "status": "in_progress",
    }
    final = {**item, "arguments": '{"city":"上海"}', "status": "completed"}
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
            "sequence_number": 1,
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": final["arguments"],
            "sequence_number": 2,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": final,
            "sequence_number": 3,
        },
        response_completed([final], {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}),
    ]
    sdk, requests = sdk_factory(events)
    async with sdk:
        result = await generate(
            provider=ProviderProfile(api_type="responses", usage_mode="native"),
            model=ModelProfile(model="demo", supports_function_calling=True),
            messages=[{"role": "user", "content": "weather?"}],
            tools=[function_tool],
            reasoning_effort="high",
            max_tokens=42,
            client=sdk,
        )
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][0]["function"]["arguments"] == final["arguments"]
        assert result["response_output"] == [final]
        assert result["usage"]["total_tokens"] == 5
        body = json.loads(requests[0].content)
        assert requests[0].url.path == "/v1/responses"
        assert body["stream"] is True
        assert body["tools"] == [{"type": "function", **function_tool["function"]}]
        assert body["reasoning"] == {"effort": "high", "summary": "detailed"}
        assert body["max_output_tokens"] == 42


async def test_actual_chat_usage_chunk_with_empty_choices(sdk_factory):
    events = [
        chat_event(content="done"),
        chat_event(finish="stop"),
        {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "demo",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    sdk, requests = sdk_factory(events)
    async with sdk:
        result = await generate(
            provider=ProviderProfile(usage_mode="stream_options"),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
            client=sdk,
        )
        assert result["usage"]["total_tokens"] == 2
        assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}


async def test_sdk_http_errors_remain_sdk_types(sdk_factory):
    sdk, _ = sdk_factory([], status=401)
    async with sdk:
        with pytest.raises(AuthenticationError):
            await generate(
                provider=ProviderProfile(),
                model="demo",
                messages=[{"role": "user", "content": "x"}],
                client=sdk,
            )


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
async def test_real_sdk_two_turn_local_tool_roundtrip(sdk_factory, api_type, function_tool):
    native_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "weather",
        "arguments": '{"city":"上海"}',
        "status": "completed",
    }
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "reason"}],
        "encrypted_content": "preserved",
    }
    if api_type == "responses":
        events = [response_completed([reasoning_item, native_call])]
    else:
        events = [
            chat_event(
                reasoning="reason",
                calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": native_call
                        | {"name": "weather", "arguments": native_call["arguments"]},
                    }
                ],
            ),
            chat_event(finish="tool_calls"),
        ]
    sdk, requests = sdk_factory(events)
    provider = ProviderProfile(api_type=api_type)
    model = ModelProfile(model="demo", supports_function_calling=True)
    history = [{"role": "user", "content": "weather?"}]
    async with sdk:
        result = await generate(
            provider=provider, model=model, messages=history, tools=[function_tool], client=sdk
        )
        history.extend(
            [
                result,
                {"role": "tool", "tool_call_id": result["tool_calls"][0]["id"], "content": "sunny"},
            ]
        )
        if api_type == "responses":
            events[:] = [
                {"type": "response.output_text.delta", "delta": "sunny", "sequence_number": 1},
                response_completed(),
            ]
        else:
            events[:] = [chat_event(content="sunny"), chat_event(finish="stop")]
        final = await generate(
            provider=provider, model=model, messages=history, tools=[function_tool], client=sdk
        )
        assert final["content"] == "sunny"
        body = json.loads(requests[-1].content)
        if api_type == "responses":
            assert body["input"][1:3] == [reasoning_item, native_call]
            assert body["input"][-1] == {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            }
        else:
            assert body["messages"][1]["reasoning_content"] == "reason"
            assert body["messages"][-1]["tool_call_id"] == "call_1"


async def test_real_sdk_parallel_invocations_have_independent_state(sdk_factory):
    import asyncio

    sdk, requests = sdk_factory(
        [chat_event(content="a"), chat_event(content="b"), chat_event(finish="stop")]
    )
    async with sdk:
        invoke = MyLLM(provider=ProviderProfile(), model=ModelProfile(model="demo"), client=sdk)
        results = await asyncio.gather(
            *(
                invoke.generate(messages=[{"role": "user", "content": str(index)}])
                for index in range(4)
            )
        )
        assert [result["content"] for result in results] == ["ab"] * 4
        assert len(requests) == 4


async def test_real_sdk_responses_stream_text_and_reasoning(sdk_factory):
    events = [
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_1",
            "delta": "text",
            "logprobs": [],
            "sequence_number": 1,
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 1,
            "summary_index": 0,
            "item_id": "rs_1",
            "delta": "reason",
            "sequence_number": 2,
        },
        response_completed(),
    ]
    sdk, _ = sdk_factory(events)
    async with sdk:
        stream = await generate(
            provider=ProviderProfile(api_type="responses"),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
            stream=True,
            client=sdk,
        )
        combined = MessageChunk()
        async with stream:
            async for chunk in stream:
                combined += chunk
        assert combined.content == "text"
        assert combined.reasoning_content == "reason"
        assert combined.response_status == "completed"


async def test_real_sdk_native_failed_event(sdk_factory):
    events = [
        {
            "type": "response.failed",
            "sequence_number": 1,
            "response": {
                "id": "resp_failed",
                "status": "failed",
                "error": {"code": "server_error", "message": "failed"},
            },
        }
    ]
    sdk, _ = sdk_factory(events)
    async with sdk:
        with pytest.raises(ResponseFailedError) as failure:
            await generate(
                provider=ProviderProfile(api_type="responses"),
                model="demo",
                messages=[{"role": "user", "content": "x"}],
                client=sdk,
            )
        assert failure.value.details["code"] == "server_error"
        assert failure.value.partial_message["response_status"] == "failed"


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
@pytest.mark.parametrize("input_format", ["responses", "chat_completions"])
async def test_real_sdk_accepts_both_tool_templates_and_returns_unified_calls(
    sdk_factory, function_tool, api_type, input_format
):
    arguments = '{"city":"上海"}'
    if api_type == "responses":
        events = [
            response_completed(
                [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "weather",
                        "arguments": arguments,
                        "status": "completed",
                    }
                ]
            )
        ]
    else:
        events = [
            chat_event(
                calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": arguments},
                    }
                ]
            ),
            chat_event(finish="tool_calls"),
        ]
    input_tool = (
        {"type": "function", **function_tool["function"]}
        if input_format == "responses"
        else function_tool
    )
    sdk, requests = sdk_factory(events)
    async with sdk:
        result = await generate(
            provider=ProviderProfile(api_type=api_type),
            model=ModelProfile(model="demo", supports_function_calling=True),
            messages=[{"role": "user", "content": "weather?"}],
            tools=[input_tool],
            client=sdk,
        )
        expected_tool = (
            {"type": "function", **function_tool["function"]}
            if api_type == "responses"
            else function_tool
        )
        assert json.loads(requests[0].content)["tools"] == [expected_tool]
        assert result["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": arguments},
            }
        ]
