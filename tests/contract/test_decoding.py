"""Recognized stream semantics through all protocol pairs.

Official native schemas and lifecycle assertions are checked independently in
test_protocol_conformance.py; unknown fields must not enter translated output.
"""

from copy import deepcopy

import pytest

from my_llm import (
    IncompleteStreamError,
    MyCodec,
    ResponseFailedError,
    ResponseIncompleteError,
    SemanticLossError,
)
from my_llm.runtime.state import StreamState
from tests.helpers.protocols import PROTOCOLS, stream_events
from tests.helpers.streaming import render


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("tool", [False, True])
def test_native_streams_convert_to_complete_target_streams(source, target, tool):
    incoming = stream_events(source, tool=tool)
    before = deepcopy(incoming)
    if source == "gemini" and target != "gemini" and tool:
        with pytest.raises(SemanticLossError, match="signature"):
            render(source, target, incoming)
        assert incoming == before
        return
    output, state = render(source, target, incoming)
    assert incoming == before
    assert state.finished
    normalized = MyCodec().encode(chunks=output, input_format=target)["chunks"]
    reconstructed = StreamState("demo")
    reconstructed.feed(normalized)
    reconstructed.finish()
    content = [block for message in reconstructed.messages() for block in message["content"]]
    assert "Hello" in "".join(block.get("text", "") for block in content)
    calls = [block for block in content if block["type"] == "tool_use"]
    assert bool(calls) is tool
    if tool:
        assert calls[0]["name"] == "weather"
        assert calls[0]["input"] == {"city": "上海"}
    if target == "openai_responses":
        assert [e["sequence_number"] for e in output] == list(range(len(output)))
        assert output[0]["type"] == "response.created"
        assert output[-1]["type"] == "response.completed"
    if target == "anthropic":
        assert output[0]["type"] == "message_start"
        assert output[-1]["type"] == "message_stop"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_native_stream_conversion_does_not_reuse_previous_call_state(protocol):
    converter = MyCodec()
    for tool in (True, False):
        incoming = stream_events(protocol, tool=tool)
        before = deepcopy(incoming)
        converted = converter.convert(
            chunks=incoming, input_format=protocol, target_format=protocol
        )["chunks"]
        _, state = render(protocol, protocol, converted)
        content = state.messages()[0]["content"]
        assert any(block["type"] == "tool_use" for block in content) is tool
        assert "Hello" in str(content)
        assert incoming == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_missing_terminal_event_is_a_message_stream_error(protocol):
    events = stream_events(protocol)
    with pytest.raises(IncompleteStreamError):
        render(protocol, "openai_chat", events[:-2] if protocol == "openai_chat" else events[:-1])


@pytest.mark.parametrize(
    "event,error",
    [
        (
            {"type": "response.failed", "response": {"error": {"message": "failed"}}},
            ResponseFailedError,
        ),
        (
            {
                "type": "response.incomplete",
                "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            },
            ResponseIncompleteError,
        ),
        ({"error": "offline"}, ResponseFailedError),
    ],
)
def test_explicit_provider_failures(event, error):
    with pytest.raises(error):
        MyCodec().encode(chunks=[event], input_format="openai_responses")


def test_function_argument_fragments_are_not_emitted():
    codec = MyCodec()
    state = StreamState("demo")
    for event in stream_events("openai_chat", tool=True):
        converted = state.feed(codec.encode(chunks=[event], input_format="openai_chat")["chunks"])
        assert all(c["type"] not in ("tool", "tool_delta") for c in converted)
    tools = [c for c in state.finish() if c["type"] == "tool"]
    assert tools[0]["block"]["input"] == {"city": "上海"}


def test_gemini_thought_signature_and_anthropic_signature_survive_stream_collection():
    _, gemini = render("gemini", "gemini", stream_events("gemini", tool=True))
    block = next(b for m in gemini.messages() for b in m["content"] if b.get("type") == "tool_use")
    assert block["signature"] == "c2lnbmVk"
    incoming = stream_events("anthropic")
    incoming[1]["content_block"] = {"type": "thinking", "thinking": "", "signature": ""}
    incoming[2]["delta"] = {"type": "thinking_delta", "thinking": "Hello"}
    incoming.insert(
        3,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed"},
        },
    )
    _, anthropic = render("anthropic", "anthropic", incoming)
    assert anthropic.messages()[0]["content"][0]["signature"] == "signed"


def test_gemini_signatures_on_ordinary_text_are_outside_the_supported_set():
    incoming = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "first", "thought_signature": "c2lnMQ=="},
                            {"text": "second", "thought_signature": "c2lnMg=="},
                        ],
                    }
                }
            ]
        },
        {"candidates": [{"finish_reason": "STOP"}]},
    ]
    _, state = render("gemini", "gemini", incoming)
    message = MyCodec().decode(state.messages(), target_format="gemini")["messages"][0]
    assert message["parts"] == [{"text": "first"}, {"text": "second"}]


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_standalone_converter_handles_complete_stream_batches(source, target):
    codec = MyCodec()
    incoming = stream_events(source, tool=True)
    if source == "gemini" and target != "gemini":
        with pytest.raises(SemanticLossError, match="signature"):
            codec.convert(chunks=incoming, input_format=source, target_format=target)
        return
    result = codec.convert(chunks=incoming, input_format=source, target_format=target)["chunks"]
    state = StreamState("demo")
    state.feed(codec.encode(chunks=result, input_format=target)["chunks"])
    state.finish()
    content = [block for message in state.messages() for block in message["content"]]
    assert "Hello" in "".join(block.get("text", "") for block in content)
    assert next(block for block in content if block["type"] == "tool_use")["name"] == "weather"
    # Reusing the same converter retains no state from the prior batch.
    second = codec.convert(chunks=stream_events(target), input_format=target, target_format=target)[
        "chunks"
    ]
    _, second_state = render(target, "openai_chat", second)
    assert second_state.finished
    content = second_state.messages()[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Hello"


def test_responses_completed_output_matches_emitted_item_ids_and_content():
    output, _ = render("openai_chat", "openai_responses", stream_events("openai_chat", tool=True))
    done = [event["item"] for event in output if event["type"] == "response.output_item.done"]
    assert output[-1]["response"]["output"] == done
    assert done[0]["content"][0].get("annotations", []) == []
    assert any(event["type"] == "response.content_part.done" for event in output)


def test_redacted_anthropic_block_survives_collected_and_streamed_output():
    incoming = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    output, state = render("anthropic", "anthropic", incoming)
    assert state.messages()[0]["content"] == [
        {"type": "redacted_thinking", "data": "opaque", "mark": "encrypted"}
    ]
    block_event = next(e for e in output if e["type"] == "content_block_start")
    assert block_event["content_block"] == {"type": "redacted_thinking", "data": "opaque"}


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_unknown_stream_events_are_ignored_for_all_protocol_pairs(source, target):
    unknown = {"type": "vendor.future_event", "payload": {"nested": [{"data": 1}]}}
    incoming = [unknown, *stream_events(source)]
    output, _ = render(source, target, incoming)
    assert unknown not in output
    assert "vendor.future_event" not in str(output)


def test_unknown_responses_output_item_does_not_enter_stream_output():
    item = {"type": "future_output", "id": "future_1", "payload": {"nested": [1, 2]}}
    from tests.helpers.protocols import response_completed

    output, _ = render("openai_responses", "openai_responses", [response_completed([item])])
    assert output[-1]["response"]["output"] == []


@pytest.mark.parametrize("target", PROTOCOLS)
def test_buffered_tool_unknown_fields_do_not_survive_stream_conversion(target):
    incoming = stream_events("openai_chat", tool=True)
    incoming[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["future"] = {"nested": [1, 2]}
    converted, _ = render("openai_chat", target, incoming)
    restored, _ = render(target, "openai_chat", converted)
    calls = [
        call
        for event in restored
        for choice in event.get("choices", [])
        for call in choice["delta"].get("tool_calls", [])
    ]
    assert "future" not in calls[0]["function"]
    assert calls[0]["function"]["name"] == "weather"


def test_gemini_text_extension_cannot_change_the_translated_block_type():
    incoming = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Hello",
                                "type": "tool_use",
                                "id": "wrong",
                                "name": "wrong",
                                "input": {},
                            },
                        ]
                    },
                    "finish_reason": "STOP",
                }
            ]
        }
    ]
    _, state = render("gemini", "openai_chat", incoming)
    assert state.messages()[0]["content"][0]["type"] == "text"
    chat = MyCodec().decode(state.messages(), target_format="openai_chat")["messages"]
    assert "tool_calls" not in chat[0]
    assert chat[0]["content"] == "Hello"


@pytest.mark.parametrize("source", ["openai_chat", "anthropic"])
def test_native_argument_named_extension_does_not_modify_stream_buffer(source):
    incoming = stream_events(source, tool=True)
    if source == "openai_chat":
        call = incoming[1]["choices"][0]["delta"]["tool_calls"][0]
        call["__arguments__"] = {"vendor": True}
    else:
        event = next(
            event for event in incoming if event.get("content_block", {}).get("type") == "tool_use"
        )
        event["content_block"]["__arguments__"] = {"vendor": True}
    _, state = render(source, "openai_chat", incoming)
    call = next(block for block in state.messages()[0]["content"] if block["type"] == "tool_use")
    assert call["input"] == {"city": "上海"}


def test_native_root_call_name_cannot_override_a_previously_translated_name():
    incoming = stream_events("openai_chat", tool=True)
    incoming[2]["choices"][0]["delta"]["tool_calls"][0]["name"] = "wrong"
    _, state = render("openai_chat", "openai_chat", incoming)
    call = next(block for block in state.messages()[0]["content"] if block["type"] == "tool_use")
    assert call["name"] == "weather"
