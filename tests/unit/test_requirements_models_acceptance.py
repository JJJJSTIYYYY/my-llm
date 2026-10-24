"""Native SDK values normalize into the finite middleware model contract."""

import pytest

from my_llm import MyCodec, ToolCall
from my_llm.runtime.state import StreamState
from tests.helpers.protocols import PROTOCOLS, chat_event


@pytest.mark.parametrize(
    "item,mark,reasoning",
    [
        ({"type": "reasoning", "encrypted_content": "opaque"}, "encrypted", ""),
        (
            {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
            "encrypted",
            "",
        ),
        (
            {
                "type": "reasoning",
                "encrypted_content": "opaque",
                "summary": [{"type": "summary_text", "text": "summary"}],
            },
            "summary",
            "summary",
        ),
    ],
)
def test_encrypted_only_reasoning_has_the_encrypted_mark(item, mark, reasoning):
    codec = MyCodec()
    encoded = codec.encode([item], input_format="openai_responses")
    block = encoded.messages[0].content[0]
    assert block.mark == mark
    assert block.reasoning == reasoning
    assert block.encrypted_content == "opaque"
    restored = codec.decode(encoded.messages, target_format="openai_responses")["messages"]
    assert restored[0]["encrypted_content"] == "opaque"


@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_nullable_gemini_arguments_form_an_empty_argument_object(sdk, target):
    from google.genai import types

    message = {"role": "model", "parts": [{"function_call": {"name": "clock", "args": None}}]}
    if sdk:
        message = types.Content.model_validate(message)
    codec = MyCodec()
    encoded = codec.encode([message], input_format="gemini")
    assert encoded.messages[0].content[0].arguments == {}
    rendered = codec.decode(encoded.messages, target_format=target)["messages"]
    restored = codec.encode(rendered, input_format=target)
    calls = [part for part in restored.messages[0].content if isinstance(part, ToolCall)]
    assert len(calls) == 1
    assert calls[0].arguments == {}


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
