"""Scalar reasoning feedback is independent of visible text and stays private."""

from copy import deepcopy

import pytest

from my_llm import (
    Message,
    MyCodec,
    Origin,
    OutputParameter,
    ReasoningContent,
    SemanticLossError,
    omit,
)
from my_llm.basic.protocols import codec_for
from my_llm.runtime.state import StreamState
from tests.helpers.profiles import config


@pytest.fixture
def scalar():
    protocol = "openai_chat"
    field = "reasoning_content"

    def chunk(**delta):
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}

    terminal = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return protocol, field, chunk, terminal


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize(
    "fragments,expected",
    [([], None), ([None], None), ([""], ""), (["", "thought", ""], "thought")],
)
def test_chat_reasoning_stream_preserves_empty_text_and_matches_native_history(
    scalar, profile_factory, profiled, fragments, expected
):
    protocol, field, chunk, terminal = scalar
    profile = profile_factory(protocol) if profiled else None
    codec = MyCodec()
    encoder = codec.codecs[protocol].encoder
    state = StreamState("demo")
    for event in [
        *(chunk(**{field: value}) for value in fragments),
        chunk(content="answer"),
        terminal,
    ]:
        state.feed(encoder.chunks([event], profile))
    state.finish()
    result = codec.result(state.messages(), input_format=protocol, target_format="middleware")
    reasoning = [part for part in result.content if isinstance(part, ReasoningContent)]
    if expected is None:
        assert reasoning == []
    else:
        assert len(reasoning) == 1
        assert reasoning[0].reasoning == reasoning[0].feedback_item == expected
    native = {"role": "assistant", "content": "answer"}
    if fragments:
        native[field] = expected
    restored = encoder.messages([native], profile)[0]
    assert [part.to_wire() for part in reasoning] == [
        part.to_wire() for part in restored.content if isinstance(part, ReasoningContent)
    ]


def test_scalar_feedback_stream_uses_its_own_output_parameter(scalar, profile_factory):
    protocol, field, chunk, terminal = scalar
    root = "$.choices[0].delta"
    profile = profile_factory(
        protocol,
        reasoning_content=OutputParameter(path=f"{root}.visible"),
        feedback_reasoning_item=OutputParameter(path=f"{root}.private"),
    )
    encoder = codec_for(protocol).encoder
    state = StreamState("demo")
    public = []
    for event in [
        chunk(visible="shown ", private="raw "),
        chunk(visible="text", private="item"),
        terminal,
    ]:
        public.extend(state.feed(encoder.chunks([event], profile)))
    public.extend(state.finish())
    assert [event["value"] for event in public if event["type"] == "delta"] == ["shown ", "text"]
    block = state.messages()[0]["content"][0]
    assert block["reasoning"] == "shown text"
    assert block["feedback_item"] == "raw item"
    native = MyCodec().decoder(protocol).messages(state.messages(), profile)
    assert native[0][field] == "raw item"
    assert "shown text" not in str(native)


def test_scalar_encrypted_object_is_retained_without_public_delta(scalar, profile_factory):
    protocol, _, chunk, terminal = scalar
    root = "$.choices[0].delta"
    profile = profile_factory(
        protocol, feedback_reasoning_item=OutputParameter(path=f"{root}.private")
    )
    feedback = {"encrypted": ["opaque"]}
    event = chunk(private=feedback)
    before = deepcopy(event)
    encoder = codec_for(protocol).encoder
    encoded = encoder.chunks([event], profile)
    feedback["encrypted"].append("source mutation")
    assert encoded[0].events[-1]["block"]["feedback_item"] == {"encrypted": ["opaque"]}
    feedback["encrypted"].pop()
    state = StreamState("demo")
    public = state.feed(encoder.chunks([event], profile))
    public.extend(state.feed(encoder.chunks([terminal], profile)))
    public.extend(state.finish())
    assert not [event for event in public if event["type"] == "delta"]
    block = state.messages()[0]["content"][0]
    assert block["reasoning"] is None
    assert block["feedback_item"] == feedback
    block["feedback_item"]["encrypted"].append("owned")
    assert event == before


def test_scalar_native_message_custom_feedback_is_separate(scalar, profile_factory):
    protocol, field, _, _ = scalar
    root = "$.choices[0].delta"
    message = {"role": "assistant", "content": "answer", field: "visible", "private": "raw"}
    profile = profile_factory(
        protocol, feedback_reasoning_item=OutputParameter(path=f"{root}.private")
    )
    block = codec_for(protocol).encoder.messages([message], profile)[0].content[1]
    assert block.reasoning == "visible"
    assert block.feedback_item == "raw"


@pytest.mark.parametrize(
    "custom,expected", [({}, "visible"), ({"private": None}, None), ({"private": "raw"}, "raw")]
)
def test_native_chat_result_replays_standard_reasoning_when_custom_field_is_absent(
    scalar, profile_factory, custom, expected
):
    protocol, field, _, _ = scalar
    message = {"role": "assistant", "content": "answer", field: "visible", **custom}
    profile = profile_factory(
        protocol, feedback_reasoning_item=OutputParameter(path="$.choices[0].delta.private")
    )
    block = codec_for(protocol).encoder.messages([message], profile)[0].content[1]
    assert block.reasoning == "visible"
    assert block.feedback_item == expected


@pytest.mark.parametrize("disabled", [omit, OutputParameter(path="$.private", supports=False)])
def test_output_feedback_declaration_filters_only_feedback(scalar, profile_factory, disabled):
    protocol, _, _, _ = scalar
    message = Message("assistant", [ReasoningContent("visible", "raw")], protocol=protocol)
    before = deepcopy(message)
    profile = profile_factory(protocol, feedback_reasoning_item=disabled)
    with pytest.raises(SemanticLossError) as captured:
        MyCodec().result(
            [message], input_format=protocol, target_format="middleware", semantic_profile=profile
        )
    block = captured.value.draft.content[0]
    assert block.reasoning == "visible"
    assert block.feedback_item is None
    assert all("feedback_reasoning_item" in loss.reason for loss in captured.value.losses)
    assert message == before


def test_request_never_recreates_feedback_from_visible_reasoning(scalar, profile_factory):
    protocol, field, _, _ = scalar
    message = Message("assistant", [ReasoningContent("visible")], protocol=protocol)
    native = (
        MyCodec()
        .decoder(protocol)
        .messages(
            [message], profile_factory(protocol, should_feedback_reasoning_item=config("always"))
        )
    )
    assert field not in native[0]
    assert "visible" not in str(native)


def test_unrepresentable_scalar_feedback_reports_loss_instead_of_crashing(scalar, profile_factory):
    protocol, _, _, _ = scalar
    message = Message(
        "assistant", [ReasoningContent("visible", {"encrypted": "raw"})], protocol=protocol
    )
    with pytest.raises(SemanticLossError, match="feedback_item must be text"):
        MyCodec().decoder(protocol).messages([message], profile_factory(protocol))


@pytest.mark.parametrize("feedback", ["raw", {"encrypted": "raw"}])
def test_native_output_reports_feedback_it_cannot_preserve(scalar, feedback):
    protocol, _, _, _ = scalar
    message = Message("assistant", [ReasoningContent("visible", feedback)], protocol=protocol)
    with pytest.raises(SemanticLossError, match="feedback_item") as captured:
        MyCodec().decoder(protocol).messages([message])
    assert "visible" in str(captured.value.draft)
    assert "raw" not in str(captured.value.draft)


def test_feedback_update_keeps_earlier_public_event_stable(scalar, profile_factory):
    protocol, field, chunk, _ = scalar
    root = "$.choices[0].delta"
    profile = profile_factory(
        protocol, feedback_reasoning_item=OutputParameter(path=f"{root}.private")
    )
    encoder = codec_for(protocol).encoder
    state = StreamState("demo")
    first = state.feed(encoder.chunks([chunk(**{field: "visible"})], profile))
    before = deepcopy(first)
    state.feed(encoder.chunks([chunk(private={"encrypted": "opaque"})], profile))
    assert first == before
    assert state.messages()[0]["content"][0]["feedback_item"] == {"encrypted": "opaque"}


def test_independent_output_filtering_preserves_source_loss_paths(scalar, profile_factory):
    protocol, _, _, _ = scalar
    source = {"type": "reasoning", "encrypted_content": "opaque"}
    message = Message(
        "assistant",
        [ReasoningContent("visible", source, origin=Origin("$.native.blocks[3]", source))],
        protocol="openai_responses",
    )
    profile = profile_factory(protocol, reasoning_content=omit)
    with pytest.raises(SemanticLossError) as captured:
        MyCodec().decoder(protocol).messages([message], semantic_profile=profile)
    assert len(captured.value.losses) == 2
    assert all(loss.origin_path == "$.native.blocks[3]" for loss in captured.value.losses)


def test_disabling_both_output_fields_removes_reasoning_lifecycle(scalar, profile_factory):
    protocol, _, _, _ = scalar
    profile = profile_factory(protocol, reasoning_content=omit, feedback_reasoning_item=omit)
    event = {
        "type": "block_start",
        "index": 0,
        "block": {"type": "thinking", "reasoning": "visible", "feedback_item": "raw"},
    }
    with pytest.raises(SemanticLossError) as captured:
        MyCodec().decoder(protocol).chunks(
            [event], semantic_profile=profile, source_protocol=protocol
        )
    assert captured.value.draft == []
