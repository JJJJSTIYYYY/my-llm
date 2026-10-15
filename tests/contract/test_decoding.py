from types import SimpleNamespace

import pytest
from conftest import chat_event, response_completed
from openai.types.chat import ChatCompletionChunk

from my_llm import (
    IncompleteStreamError,
    MessageChunk,
    ProviderProfile,
    ResponseFailedError,
    ResponseIncompleteError,
    omit,
)
from my_llm.codec import ChatCompletionsCodec, ResponsesCodec


def test_chat_sdk_chunks_text_reasoning_and_usage(profile_factory):
    decoder = ChatCompletionsCodec(
        profile_factory(provider=ProviderProfile(usage_mode="stream_options"))
    ).decoder()
    chunks = []
    for event in [
        chat_event(content="你", reasoning="想"),
        chat_event(content="好", reasoning="法"),
        chat_event(finish="stop"),
        {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "demo",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
    ]:
        chunks.extend(decoder.feed(ChatCompletionChunk.model_validate(event)))
    final = decoder.finish()
    assert final.finish_reason == "stop"
    assert final.usage["total_tokens"] == 5
    combined = MessageChunk()
    for chunk in [*chunks, final]:
        combined += chunk
    assert combined.content == "你好" and combined.reasoning_content == "想法"


def test_chat_interleaved_parallel_tools_are_only_emitted_complete(profile_factory):
    decoder = ChatCompletionsCodec(profile_factory()).decoder()
    events = [
        chat_event(
            calls=[
                {
                    "index": 1,
                    "id": "b",
                    "type": "function",
                    "function": {"name": "second", "arguments": '{"b":'},
                },
                {
                    "index": 0,
                    "id": "a",
                    "type": "function",
                    "function": {"name": "first", "arguments": '{"a":'},
                },
            ]
        ),
        chat_event(
            calls=[
                {"index": 0, "function": {"arguments": "1}"}},
                {"index": 1, "function": {"arguments": "2}"}},
            ]
        ),
        chat_event(finish="tool_calls"),
    ]
    for event in events:
        assert decoder.feed(event) == []
    calls = decoder.finish().tool_calls
    assert [call["id"] for call in calls] == ["a", "b"]
    assert [call["function"]["arguments"] for call in calls] == ['{"a":1}', '{"b":2}']


@pytest.mark.parametrize(
    "finish,arguments", [(None, "{}"), ("length", '{"x":'), ("tool_calls", '{"x":'), ("stop", "{}")]
)
def test_chat_incomplete_tool_results_are_never_returned(profile_factory, finish, arguments):
    decoder = ChatCompletionsCodec(profile_factory()).decoder()
    decoder.feed(
        chat_event(
            calls=[
                {
                    "index": 0,
                    "id": "call",
                    "type": "function",
                    "function": {"name": "tool", "arguments": arguments},
                }
            ],
            finish=finish,
        )
    )
    with pytest.raises(IncompleteStreamError):
        decoder.finish()


def test_chat_no_finish_and_error(profile_factory):
    decoder = ChatCompletionsCodec(profile_factory()).decoder()
    decoder.feed(chat_event(content="partial"))
    with pytest.raises(IncompleteStreamError, match="finish_reason"):
        decoder.finish()
    with pytest.raises(ResponseFailedError) as error:
        ChatCompletionsCodec(profile_factory()).decoder().feed({"error": {"message": "bad"}})
    assert error.value.details == {"message": "bad"}


def test_chat_custom_paths_and_omit(profile_factory):
    decoder = ChatCompletionsCodec(
        profile_factory(content_path="$.choices[0].delta.custom.text", reasoning_content_path=omit)
    ).decoder()
    event = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(custom=SimpleNamespace(text="custom")), finish_reason="stop"
            )
        ]
    )
    chunks = decoder.feed(event)
    assert chunks[0].content == "custom" and chunks[0].reasoning_content == ""
    assert decoder.finish().finish_reason == "stop"


def test_responses_tool_snapshots_do_not_duplicate_fragments(profile_factory):
    decoder = ResponsesCodec(profile_factory("responses")).decoder()
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "weather",
        "arguments": "",
        "status": "in_progress",
    }
    final_item = {**item, "arguments": '{"city":"上海"}', "status": "completed"}
    events = [
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city":'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '"上海"}'},
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "arguments": final_item["arguments"],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": final_item},
        response_completed([final_item]),
    ]
    for event in events:
        assert decoder.feed(event) == []
    result = decoder.finish()
    assert result.tool_calls[0] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "weather", "arguments": final_item["arguments"]},
    }
    assert result.response_output == [final_item]
    assert result.response_id == "resp_1"


def test_responses_text_reasoning_native_usage_and_replay(profile_factory):
    from my_llm import ProviderProfile

    decoder = ResponsesCodec(
        profile_factory(
            "responses", provider=ProviderProfile(api_type="responses", usage_mode="native")
        )
    ).decoder()
    assert (
        decoder.feed({"type": "response.output_text.delta", "delta": "hello"})[0].content == "hello"
    )
    assert (
        decoder.feed({"type": "response.reasoning_summary_text.delta", "delta": "reason"})[
            0
        ].reasoning_content
        == "reason"
    )
    native = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "reason"}],
            "encrypted_content": "secret",
        }
    ]
    decoder.feed(
        response_completed(native, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
    )
    result = decoder.finish()
    assert result.finish_reason == "stop" and result.response_status == "completed"
    assert result.usage["total_tokens"] == 5
    assert result.response_output == native


@pytest.mark.parametrize(
    "kind,error_type",
    [
        ("response.failed", ResponseFailedError),
        ("error", ResponseFailedError),
        ("response.incomplete", ResponseIncompleteError),
    ],
)
def test_responses_failure_statuses(profile_factory, kind, error_type):
    decoder = ResponsesCodec(profile_factory("responses")).decoder()
    with pytest.raises(error_type):
        decoder.feed(
            {
                "type": kind,
                "response": {
                    "error": {"code": "server_error"},
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        )


def test_responses_truncation_and_unknown_call(profile_factory):
    decoder = ResponsesCodec(profile_factory("responses")).decoder()
    with pytest.raises(IncompleteStreamError, match="unknown"):
        decoder.feed(
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{}"}
        )
    with pytest.raises(IncompleteStreamError, match="response.completed"):
        decoder.finish()


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "function_call",
            "call_id": "a",
            "name": "f",
            "arguments": "{",
            "status": "completed",
        },
        {
            "type": "function_call",
            "call_id": "a",
            "name": "f",
            "arguments": "{}",
            "status": "incomplete",
        },
        {"type": "web_search_call", "id": "web_1"},
    ],
)
def test_invalid_response_output_items(profile_factory, item):
    decoder = ResponsesCodec(profile_factory("responses")).decoder()
    decoder.feed(response_completed([item]))
    with pytest.raises(IncompleteStreamError):
        decoder.finish()


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_usage_none_never_exposes_native_usage(profile_factory, api_type):
    decoder = (ResponsesCodec if api_type == "responses" else ChatCompletionsCodec)(
        profile_factory(api_type)
    ).decoder()
    decoder.feed(
        response_completed(usage={"total_tokens": 9})
        if api_type == "responses"
        else chat_event(finish="stop", usage={"total_tokens": 9})
    )
    assert decoder.finish().usage is None
