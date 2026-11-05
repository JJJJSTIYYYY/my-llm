"""Native SDK values normalize into the finite middleware model contract."""

import pytest

from my_llm import MyCodec
from my_llm.runtime.state import StreamState
from tests.helpers.protocols import chat_event


@pytest.mark.parametrize(
    "item,reasoning",
    [
        ({"type": "reasoning", "encrypted_content": "opaque"}, None),
        (
            {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
            None,
        ),
        (
            {
                "type": "reasoning",
                "encrypted_content": "opaque",
                "summary": [{"type": "summary_text", "text": "summary"}],
            },
            "summary",
        ),
    ],
)
def test_reasoning_separates_visible_text_and_native_feedback(item, reasoning):
    codec = MyCodec()
    encoded = codec.encode([item], input_format="openai_responses")
    block = encoded.messages[0].content[0]
    assert block.reasoning == reasoning
    assert block.feedback_item == item
    restored = codec.decode(encoded.messages, target_format="openai_responses")["messages"]
    assert restored[0]["encrypted_content"] == "opaque"


@pytest.mark.parametrize("sdk", [False, True])
def test_nullable_chat_argument_fragments_are_absent_fragments(sdk):
    from openai.types.chat import ChatCompletionChunk

    events = [
        chat_event(
            calls=[
                {
                    "index": 0,
                    "id": "c",
                    "type": "function",
                    "function": {"name": "clock", "arguments": None},
                }
            ]
        ),
        chat_event(calls=[{"index": 0, "function": {"arguments": "{}"}}]),
        chat_event(finish="tool_calls"),
    ]
    if sdk:
        events = [ChatCompletionChunk.model_validate(event) for event in events]
    encoded = MyCodec().encode(chunks=events, input_format="openai_chat")
    state = StreamState("demo")
    state.feed(encoded["chunks"])
    finished = state.finish()
    assert finished[-1]["messages"][0]["content"][0]["input"] == {}
