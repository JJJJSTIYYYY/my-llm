"""Cross-API input recognition, content conversion and native-history replay."""

from copy import deepcopy

import pytest

from my_llm import MessageValidationError, omit
from my_llm.codec import ChatCompletionsCodec, ResponsesCodec


def encode(profile_factory, api_type, messages, **kwargs):
    profile = profile_factory(api_type, **kwargs)
    codec = ResponsesCodec(profile) if api_type == "responses" else ChatCompletionsCodec(profile)
    request = codec.encode(messages, tools=omit)
    return request["input" if api_type == "responses" else "messages"]


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
@pytest.mark.parametrize("input_format", ["responses", "chat_completions"])
def test_text_messages_recognized_and_encoded_for_target(profile_factory, api_type, input_format):
    chat = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    responses = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    messages = responses if input_format == "responses" else chat
    before = deepcopy(messages)
    result = encode(profile_factory, api_type, messages)
    expected = (
        responses
        if api_type == "responses" and input_format == "responses"
        else (
            [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
            if api_type == "responses"
            else chat
        )
    )
    assert result == expected
    assert messages == before


def test_responses_same_format_passes_original_items_with_metadata(profile_factory):
    messages = [
        {
            "type": "message",
            "id": "in_1",
            "status": "completed",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "question",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "reason"}],
            "encrypted_content": "opaque",
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "answer",
                    "annotations": [{"type": "custom", "value": 1}],
                    "logprobs": [],
                }
            ],
        },
    ]
    result = encode(profile_factory, "responses", messages)
    assert result == messages
    assert all(item is original for item, original in zip(result, messages))


def test_untyped_responses_assistant_phase_is_preserved(profile_factory):
    message = {"role": "assistant", "phase": "commentary", "content": "working"}
    assert encode(profile_factory, "responses", [message]) == [message]
    assert encode(profile_factory, "chat_completions", [message]) == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "working"}],
            "reasoning_content": "",
        }
    ]


@pytest.fixture
def native_history():
    return [
        {"role": "user", "content": "weather?"},
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [
                {"type": "summary_text", "text": "look up "},
                {"type": "summary_text", "text": "both cities"},
            ],
            "encrypted_content": "never-send-to-chat",
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "checking", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "weather",
            "arguments": '{"city":"上海"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "call_id": "call_2",
            "name": "weather",
            "arguments": '{"city":"北京"}',
            "status": None,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": [{"type": "input_text", "text": "cloudy"}],
        },
    ]


@pytest.mark.parametrize("replay", [True, False])
def test_native_parallel_calls_and_results_become_one_chat_assistant(
    profile_factory, native_history, replay
):
    before = deepcopy(native_history)
    result = encode(
        profile_factory,
        "chat_completions",
        native_history,
        replay_reasoning_content_when_tool_calls=replay,
    )
    assert len(result) == 4
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == [{"type": "text", "text": "checking"}]
    assert result[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"上海"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"北京"}'},
        },
    ]
    assert result[2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert result[3] == {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": [{"type": "text", "text": "cloudy"}],
    }
    assert ("reasoning_content" in result[1]) is replay
    if replay:
        assert result[1]["reasoning_content"] == "look up both cities"
    assert "never-send-to-chat" not in str(result)
    assert native_history == before


def test_native_parallel_history_remains_unchanged_for_responses(profile_factory, native_history):
    assert encode(profile_factory, "responses", native_history) == native_history


def test_native_reasoning_is_attached_to_following_chat_function_call(profile_factory, tool_call):
    history = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "native reason"}],
            "encrypted_content": "opaque",
        },
        {"role": "assistant", "tool_calls": [tool_call], "reasoning_content": " chat reason"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
    ]
    result = encode(profile_factory, "chat_completions", history)
    assert result[0]["reasoning_content"] == "native reason chat reason"
    assert result[0]["tool_calls"] == [tool_call]
    assert len(result) == 2


def test_parallel_calls_split_across_formats_merge_into_one_assistant(profile_factory, tool_call):
    history = [
        {"role": "assistant", "content": "first", "tool_calls": [tool_call]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "second", "annotations": []}],
        },
        {"type": "function_call", "call_id": "call_2", "name": "weather", "arguments": "{}"},
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"type": "function_call_output", "call_id": "call_2", "output": "two"},
    ]
    result = encode(profile_factory, "chat_completions", history)
    assert len(result) == 3
    assert [call["id"] for call in result[0]["tool_calls"]] == ["call_1", "call_2"]
    assert result[0]["content"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]


def test_same_format_chat_assistant_messages_are_not_combined(profile_factory):
    history = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    assert encode(profile_factory, "chat_completions", history) == [
        {**message, "reasoning_content": ""} for message in history
    ]


@pytest.mark.parametrize("summary", [[], [{"type": "summary_text", "text": "reason"}]])
def test_native_reasoning_between_chat_calls_keeps_a_single_assistant_turn(
    profile_factory, tool_call, summary
):
    second = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "weather", "arguments": "{}"},
    }
    history = [
        {"role": "assistant", "tool_calls": [tool_call]},
        {"type": "reasoning", "summary": summary, "encrypted_content": "opaque"},
        {"role": "assistant", "tool_calls": [second]},
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "two"},
    ]
    result = encode(profile_factory, "chat_completions", history)
    assert len(result) == 3
    assert result[0]["tool_calls"] == [tool_call, second]
    assert result[0]["reasoning_content"] == ("reason" if summary else "")


@pytest.mark.parametrize("text_format", ["chat_completions", "responses"])
def test_native_reasoning_survives_interleaved_text_before_call(
    profile_factory, tool_call, text_format
):
    text = (
        {"role": "assistant", "content": "checking"}
        if text_format == "chat_completions"
        else {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "checking", "annotations": []}],
        }
    )
    history = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "reason"}]},
        text,
        {"type": "function_call", "call_id": "call_1", **tool_call["function"]},
    ]
    result = encode(profile_factory, "chat_completions", history)
    assert len(result) == 1
    assert result[0]["content"] == (
        "checking" if text_format == "chat_completions" else [{"type": "text", "text": "checking"}]
    )
    assert result[0]["reasoning_content"] == "reason"
    assert result[0]["tool_calls"] == [tool_call]


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_mixed_messages_and_tool_history(profile_factory, api_type, tool_call):
    history = [
        {"role": "user", "content": "weather?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "weather",
            "arguments": tool_call["function"]["arguments"],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "sunny"}]},
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_final",
            "status": "completed",
            "content": [{"type": "output_text", "text": "final", "annotations": []}],
        },
    ]
    result = encode(profile_factory, api_type, history)
    if api_type == "responses":
        assert result[1] == history[1]
        assert result[2] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "sunny"}],
        }
        assert result[3] == history[3]
    else:
        assert result[1]["tool_calls"] == [tool_call]
        assert result[2]["role"] == "tool"
        assert result[3]["content"] == [{"type": "text", "text": "final"}]


@pytest.mark.parametrize(
    "kind", ["duplicate", "unknown_result", "incomplete", "partial_json", "missing_id"]
)
@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_invalid_native_tool_history_rejected(profile_factory, api_type, kind):
    call = {"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{}"}
    if kind == "duplicate":
        messages = [call, call]
    elif kind == "unknown_result":
        messages = [{"type": "function_call_output", "call_id": "unknown", "output": "result"}]
    else:
        if kind == "incomplete":
            call["status"] = "incomplete"
        elif kind == "partial_json":
            call["arguments"] = "{"
        else:
            call.pop("call_id")
        messages = [call]
    with pytest.raises(MessageValidationError):
        encode(profile_factory, api_type, messages)


def test_native_result_with_previous_response_id(profile_factory):
    profile = profile_factory("responses")
    item = {"type": "function_call_output", "call_id": "previous_call", "output": "result"}
    request = ResponsesCodec(profile).encode(
        [item], tools=omit, previous_response_id="resp_previous"
    )
    assert request["input"] == [item]
    assert request["previous_response_id"] == "resp_previous"


@pytest.mark.parametrize(
    "block,expected",
    [
        (
            {"type": "input_image", "image_url": "https://example.test/image.png", "detail": "low"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png", "detail": "low"},
            },
        ),
        (
            {"type": "input_file", "file_id": "file_1", "filename": "test.pdf"},
            {"type": "file", "file": {"file_id": "file_1", "filename": "test.pdf"}},
        ),
        (
            {"type": "input_file", "file_data": "base64-data", "filename": "test.pdf"},
            {"type": "file", "file": {"file_data": "base64-data", "filename": "test.pdf"}},
        ),
    ],
)
def test_native_media_conversion_to_chat(profile_factory, block, expected):
    assert encode(
        profile_factory,
        "chat_completions",
        [{"type": "message", "role": "user", "content": [block]}],
    ) == [{"role": "user", "content": [expected]}]


def test_chat_files_convert_to_responses(profile_factory):
    message = {
        "role": "user",
        "content": [{"type": "file", "file": {"file_id": "file_1", "filename": "test.pdf"}}],
    }
    assert encode(profile_factory, "responses", [message]) == [
        {
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file_1", "filename": "test.pdf"}],
        }
    ]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_image", "file_id": "image_1", "detail": "auto"},
        {"type": "input_file", "file_url": "https://example.test/test.pdf"},
    ],
)
def test_responses_only_media_passes_responses_and_rejects_chat(profile_factory, block):
    message = {"type": "message", "role": "user", "content": [block]}
    assert encode(profile_factory, "responses", [message]) == [message]
    with pytest.raises(MessageValidationError, match="cannot represent"):
        encode(profile_factory, "chat_completions", [message])


def test_mixed_content_blocks_convert_per_block(profile_factory):
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "one"}, {"type": "input_text", "text": "two"}],
    }
    assert encode(profile_factory, "responses", [message])[0]["content"] == [
        {"type": "input_text", "text": "one"},
        {"type": "input_text", "text": "two"},
    ]
    assert encode(profile_factory, "chat_completions", [message])[0]["content"] == [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two"},
    ]


def test_native_refusals_convert_without_losing_text(profile_factory):
    native = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "cannot comply"}],
    }
    assert encode(profile_factory, "chat_completions", [native]) == [
        {"role": "assistant", "content": native["content"], "reasoning_content": ""}
    ]
    assert encode(
        profile_factory,
        "responses",
        [{"role": "assistant", "refusal": "cannot comply", "content": None}],
    ) == [{"role": "assistant", "content": native["content"]}]


def test_item_references_are_not_silently_lost(profile_factory):
    reference = {"type": "item_reference", "id": "msg_previous"}
    assert encode(profile_factory, "responses", [reference]) == [reference]
    with pytest.raises(MessageValidationError, match="item_reference"):
        encode(profile_factory, "chat_completions", [reference])


def test_chat_audio_stays_native_and_rejects_responses(profile_factory):
    message = {
        "role": "user",
        "content": [{"type": "input_audio", "input_audio": {"data": "base64", "format": "wav"}}],
    }
    assert encode(profile_factory, "chat_completions", [message]) == [message]
    with pytest.raises(MessageValidationError):
        encode(profile_factory, "responses", [message])


@pytest.mark.parametrize(
    "messages",
    [
        [],
        "hello",
        ["hello"],
        [{"type": "unsupported"}],
        [{"type": "message", "role": "user"}],
        [{"type": "reasoning", "summary": "invalid"}],
        [{"type": "message", "role": "assistant", "content": "x", "tool_calls": []}],
    ],
)
def test_invalid_or_ambiguous_input_rejected(profile_factory, messages):
    with pytest.raises(MessageValidationError):
        encode(profile_factory, "responses", messages)
