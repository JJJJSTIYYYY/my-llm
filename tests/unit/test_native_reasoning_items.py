"""Visible reasoning and native replay items remain independent across native codecs."""

from copy import deepcopy

import pytest

from my_llm import Message, Parameter, ReasoningContent, SemanticLossError
from my_llm.codec.anthropic.decoding.decoder import AnthropicDecoder
from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
from my_llm.codec.gemini.decoding.decoder import GeminiDecoder
from my_llm.codec.gemini.encoding.encoder import GeminiEncoder
from my_llm.runtime.state import StreamState
from tests.helpers.streaming import render


@pytest.mark.parametrize(
    "native",
    [
        {"type": "thinking", "thinking": "visible", "signature": "opaque"},
        {"type": "redacted_thinking", "data": "opaque"},
    ],
)
def test_anthropic_reasoning_owns_and_replays_complete_native_block(native):
    message = {"role": "assistant", "content": [native]}
    encoded = AnthropicEncoder().messages([message])
    block = encoded[0].content[0]
    assert block.reasoning == native.get("thinking")
    assert block.feedback_item == native
    assert AnthropicDecoder().messages(encoded) == [message]
    block.feedback_item["owned"] = True
    assert "owned" not in native


def test_anthropic_stream_accumulates_signature_privately_and_replays_redacted_blocks():
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "first"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "second"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opaque-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signature"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "redacted"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    before = deepcopy(events)
    encoder = AnthropicEncoder()
    state = StreamState("demo")
    public = []
    for event in events:
        public.extend(state.feed(encoder.chunks([event])))
    public.extend(state.finish())
    assert [(event["field"], event["value"]) for event in public if event["type"] == "delta"] == [
        ("reasoning", "first"),
        ("reasoning", "second"),
    ]
    blocks = state.messages()[0]["content"]
    assert blocks[0]["reasoning"] == "firstsecond"
    assert blocks[0]["feedback_item"] == {
        "type": "thinking",
        "thinking": "firstsecond",
        "signature": "opaque-signature",
    }
    assert blocks[1]["reasoning"] is None
    assert blocks[1]["feedback_item"] == {"type": "redacted_thinking", "data": "redacted"}
    assert events == before


def test_anthropic_custom_feedback_parameter_selects_complete_private_item(profile_factory):
    native = {"type": "thinking", "thinking": "visible", "signature": "custom"}
    profile = profile_factory(
        "anthropic", feedback_reasoning_item=Parameter(path="$.delta.private")
    )
    state = StreamState("demo")
    encoder = AnthropicEncoder()
    for event in [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "visible"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "ignored", "private": native},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]:
        state.feed(encoder.chunks([event], profile))
    state.finish()
    assert state.messages()[0]["content"][0]["feedback_item"] == native


def test_anthropic_visible_override_does_not_change_native_feedback_text(profile_factory):
    profile = profile_factory("anthropic", reasoning_content=Parameter(path="$.delta.summary"))
    state = StreamState("demo")
    encoder = AnthropicEncoder()
    for event in [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "native", "summary": "visible"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opaque"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]:
        state.feed(encoder.chunks([event], profile))
    state.finish()
    block = state.messages()[0]["content"][0]
    assert block["reasoning"] == "visible"
    assert block["feedback_item"] == {
        "type": "thinking",
        "thinking": "native",
        "signature": "opaque",
    }


@pytest.mark.parametrize("text", ["visible", ""])
def test_gemini_native_thought_part_is_the_feedback_item(text):
    native = {"text": text, "thought": True, "thought_signature": "c2lnbmVk"}
    message = {"role": "model", "parts": [native]}
    encoded = GeminiEncoder().messages([message])
    block = encoded[0].content[0]
    assert block.reasoning == text
    assert block.feedback_item == native
    assert GeminiDecoder().messages(encoded) == [message]
    block.feedback_item["owned"] = True
    assert "owned" not in native


def test_gemini_signature_only_native_thought_part_survives_streaming():
    native = {"thought": True, "thought_signature": "c2lnbmVk"}
    output, state = render(
        "gemini",
        "gemini",
        [
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [native]}, "finish_reason": "STOP"}
                ]
            }
        ],
    )
    block = state.messages()[0]["content"][0]
    assert block["reasoning"] is None
    assert block["feedback_item"] == native
    assert output[0]["candidates"][0]["content"]["parts"] == [native]


def test_gemini_custom_feedback_parameter_preserves_visible_reasoning(profile_factory):
    native = {"text": "visible", "thought": True, "thought_signature": "ignored"}
    feedback = {"text": "visible", "thought": True, "thought_signature": "custom"}
    event = {
        "private": feedback,
        "candidates": [{"content": {"role": "model", "parts": [native]}, "finish_reason": "STOP"}],
    }
    profile = profile_factory("gemini", feedback_reasoning_item=Parameter(path="$.private"))
    state = StreamState("demo")
    state.feed(GeminiEncoder().chunks([event], profile))
    state.finish()
    blocks = state.messages()[0]["content"]
    assert "".join(block.get("reasoning") or "" for block in blocks) == "visible"
    assert [
        block["feedback_item"] for block in blocks if block.get("feedback_item") is not None
    ] == [feedback]


@pytest.mark.parametrize(
    "decoder,protocol", [(AnthropicDecoder, "anthropic"), (GeminiDecoder, "gemini")]
)
def test_visible_reasoning_is_not_recreated_as_request_feedback(profile_factory, decoder, protocol):
    message = Message("assistant", [ReasoningContent("visible")], protocol=protocol)
    draft = decoder().messages([message], profile_factory(protocol))
    assert "visible" not in str(draft)


@pytest.mark.parametrize("decoder", [AnthropicDecoder, GeminiDecoder])
def test_foreign_feedback_is_lost_but_visible_reasoning_survives_output(decoder):
    message = Message(
        "assistant",
        [ReasoningContent("visible", {"encrypted_content": "opaque"})],
        protocol="openai_responses",
    )
    with pytest.raises(SemanticLossError) as captured:
        decoder().messages([message])
    error = captured.value
    assert "visible" in str(error.draft)
    assert "opaque" not in str(error.draft)
    assert len(error.losses.items) == 1


@pytest.mark.parametrize(
    "decoder,protocol", [(AnthropicDecoder, "anthropic"), (GeminiDecoder, "gemini")]
)
def test_native_stream_reports_unrepresentable_feedback(decoder, protocol):
    event = {
        "type": "block_stop",
        "index": 0,
        "block": {"type": "thinking", "reasoning": "visible", "feedback_item": "opaque"},
    }
    with pytest.raises(SemanticLossError, match="native feedback item"):
        decoder().chunks([event], source_protocol=protocol)


def test_native_gemini_stream_output_does_not_share_its_feedback_part():
    feedback = {"thought": True, "thought_signature": "c2lnbmVk"}
    event = {
        "type": "block",
        "index": 0,
        "block": {"type": "thinking", "reasoning": None, "feedback_item": feedback},
    }
    before = deepcopy(event)
    output = GeminiDecoder().chunks([event], source_protocol="gemini")
    output[0]["candidates"][0]["content"]["parts"][0]["thought_signature"] = "changed"
    assert event == before
