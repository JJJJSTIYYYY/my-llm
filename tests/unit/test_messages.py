import pytest

from my_llm import MessageChunk
from my_llm.codec.utils.common import path_value
from my_llm.codec.utils.validation import validate_messages
from my_llm.types.exceptions import ConfigurationError, MessageValidationError


def test_chunk_addition_and_message_finalization(tool_call):
    first = MessageChunk(content="你好", reasoning_content="思考", tool_calls=[tool_call])
    final = MessageChunk(
        content="世界",
        finish_reason="tool_calls",
        usage={"total_tokens": 3},
        response_id="resp_1",
        response_output=[{"type": "reasoning", "id": "rs_1"}],
        response_status="completed",
    )
    combined = first + final
    assert first.content == "你好"
    assert combined.content == "你好世界"
    assert combined.has_delta and combined.is_finished
    message = combined.to_message(require_finished=True)
    assert message["tool_calls"] == [tool_call]
    assert message["usage"] == {"total_tokens": 3}
    assert message["response_id"] == "resp_1"
    message["tool_calls"][0]["function"]["name"] = "changed"
    assert first.tool_calls[0]["function"]["name"] == "weather"
    assert combined.tool_calls[0]["function"]["name"] == "weather"


def test_empty_chunk_and_unfinished_message():
    chunk = MessageChunk()
    assert not chunk.has_delta and not chunk.is_finished
    assert chunk.to_message() == {"role": "assistant", "content": None}
    with pytest.raises(ValueError, match="unfinished"):
        chunk.to_message(require_finished=True)
    with pytest.raises(TypeError):
        chunk + "invalid"


def test_conflicting_finish_reasons():
    with pytest.raises(ValueError, match="conflicting"):
        MessageChunk(finish_reason="stop") + MessageChunk(finish_reason="length")


@pytest.mark.parametrize(
    "message",
    [
        {"role": "developer", "content": "x"},
        {"role": "user"},
        {"role": "user", "content": 4},
        {"role": "tool", "content": "x", "tool_call_id": "unknown"},
        {"role": "user", "content": "x", "tool_calls": []},
        {"role": "user", "content": "x", "reasoning_content": "x"},
    ],
)
def test_invalid_message_fields(message):
    with pytest.raises(MessageValidationError):
        validate_messages([message])


def test_tool_result_association(tool_call):
    history = [
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    assert validate_messages(history) == history
    with pytest.raises(MessageValidationError, match="unanswered"):
        validate_messages([*history, history[-1]])


def test_path_reads_sdk_attributes_without_dumping():
    from types import SimpleNamespace

    class Native:
        choices = [SimpleNamespace(delta=SimpleNamespace(content="text"))]

        def model_dump(self):
            raise AssertionError("must not serialize for path extraction")

    assert path_value(Native(), "$.choices[0].delta.content") == "text"
    assert path_value(Native(), "choices[1].delta.content") is None
    assert path_value({"value": {"text": "x"}}, "$.value.text") == "x"
    with pytest.raises(ConfigurationError):
        path_value(Native(), "$.choices[*].delta.content")
