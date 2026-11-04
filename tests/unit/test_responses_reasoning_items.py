"""Responses reasoning keeps one visible stream and one native replay item."""

from copy import deepcopy

import pytest

from my_llm import (
    InputParameter,
    Message,
    OutputParameter,
    ReasoningContent,
    SemanticLossError,
    omit,
)
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.runtime.state import StreamState
from tests.helpers.profiles import config
from tests.helpers.protocols import response_completed
from tests.helpers.streaming import render


def reasoning_item(*texts, encrypted="opaque"):
    return {
        "type": "reasoning",
        "id": "rs_native",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": text} for text in texts],
        "encrypted_content": encrypted,
    }


@pytest.mark.parametrize("texts", [("first", "second"), ()])
def test_reasoning_item_round_trip_retains_native_item_and_one_visible_block(texts):
    native = reasoning_item(*texts)
    before = deepcopy(native)
    encoded = ResponsesEncoder().messages([native])
    assert len(encoded) == 1
    assert len(encoded[0].content) == 1
    block = encoded[0].content[0]
    assert block.reasoning == ("\n".join(texts) or None)
    assert block.feedback_item == native
    assert ResponsesDecoder().messages(encoded) == [native]
    block.feedback_item["summary"].append({"type": "summary_text", "text": "owned"})
    assert native == before


def test_feedback_output_parameter_selects_item_envelope_path(profile_factory):
    native = reasoning_item("visible")
    profile = profile_factory(
        "openai_responses",
        feedback_reasoning_item=OutputParameter(path="$.item.encrypted_content"),
    )
    block = ResponsesEncoder().messages([native], profile)[0].content[0]
    assert block.reasoning == "visible"
    assert block.feedback_item == "opaque"


def test_visible_reasoning_without_feedback_is_not_recreated_in_history(profile_factory):
    message = Message("assistant", [ReasoningContent("visible")], protocol="openai_responses")
    result = ResponsesDecoder().messages([message], profile_factory("openai_responses"))
    assert all(item.get("type") != "reasoning" for item in result)
    assert "visible" not in str(result)


@pytest.mark.parametrize("is_request", [False, True])
def test_unreplayable_feedback_keeps_visible_output_but_is_not_request_feedback(
    profile_factory, is_request
):
    message = Message(
        "assistant", [ReasoningContent("visible", "opaque")], protocol="openai_responses"
    )
    profile = profile_factory("openai_responses") if is_request else None
    with pytest.raises(SemanticLossError) as captured:
        ResponsesDecoder().messages([message], profile)
    assert ("visible" in str(captured.value.draft)) is not is_request
    assert "opaque" not in str(captured.value.draft)


def test_feedback_mode_only_filters_history(profile_factory):
    native = reasoning_item("visible")
    encoded = ResponsesEncoder().messages([native])
    profile = profile_factory("openai_responses", should_feedback_reasoning_item=config("none"))
    assert all(
        item.get("type") != "reasoning" for item in ResponsesDecoder().messages(encoded, profile)
    )
    assert ResponsesDecoder().messages(encoded) == [native]


def test_foreign_feedback_loss_retains_visible_reasoning_in_output():
    message = Message(
        "assistant",
        [
            ReasoningContent(
                "visible", {"type": "thinking", "thinking": "visible", "signature": "opaque"}
            )
        ],
        protocol="anthropic",
    )
    with pytest.raises(SemanticLossError) as captured:
        ResponsesDecoder().messages([message])
    error = captured.value
    assert error.draft[0]["summary"] == [{"type": "summary_text", "text": "visible"}]
    assert "opaque" not in str(error.draft)


def test_multi_summary_stream_accumulates_one_reasoning_item_without_feedback_deltas():
    native = reasoning_item("first", "second")
    encoder = ResponsesEncoder()
    state = StreamState("demo")
    public = []
    for summary_index, text in enumerate(("first", "second")):
        events = encoder.chunks(
            [
                {
                    "type": "response.reasoning_summary_part.added",
                    "output_index": 0,
                    "summary_index": summary_index,
                    "part": {"type": "summary_text", "text": ""},
                },
                {
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": 0,
                    "summary_index": summary_index,
                    "item_id": native["id"],
                    "delta": text,
                },
            ]
        )
        public.extend(state.feed(events))
        assert state.messages()[0]["content"][0].get("feedback_item") is None
    assert [event["value"] for event in public if event["type"] == "delta"] == [
        "first",
        "\n",
        "second",
    ]
    public.extend(
        state.feed(
            encoder.chunks(
                [{"type": "response.output_item.done", "output_index": 0, "item": native}]
            )
        )
    )
    public.extend(state.feed(encoder.chunks([response_completed([native])])))
    public.extend(state.finish())
    blocks = state.messages()[0]["content"]
    assert len(blocks) == 1
    assert blocks[0]["reasoning"] == "first\nsecond"
    assert blocks[0]["feedback_item"] == native
    assert all(event.get("field") != "feedback_item" for event in public)
    output = ResponsesDecoder().chunks(
        public, **{**state.metadata, "source_protocol": "openai_responses"}
    )
    assert output[-1]["response"]["output"] == [native]


def test_encrypted_only_stream_preserves_feedback_without_visible_delta():
    native = reasoning_item()
    output, state = render("openai_responses", "openai_responses", [response_completed([native])])
    blocks = state.messages()[0]["content"]
    assert len(blocks) == 1
    assert blocks[0].get("reasoning") is None
    assert blocks[0]["feedback_item"] == native
    assert output[-1]["response"]["output"] == [native]
    assert sum(event["type"] == "response.output_item.added" for event in output) == 1
    assert sum(event["type"] == "response.output_item.done" for event in output) == 1
    assert all(event["type"] != "response.reasoning_summary_text.delta" for event in output)


@pytest.mark.parametrize("filtered", ["both_fields", "capability", "opaque_feedback"])
def test_filtered_reasoning_is_not_recreated_from_terminal_metadata(profile_factory, filtered):
    native = reasoning_item(*(() if filtered == "opaque_feedback" else ("visible",)))
    options = (
        {"reasoning_content": omit, "feedback_reasoning_item": omit}
        if filtered == "both_fields"
        else {"reasoning_enable": InputParameter(path="$.reasoning", supports=False)}
        if filtered == "capability"
        else {"feedback_reasoning_item": omit}
    )
    profile = profile_factory("openai_responses", **options)
    state = StreamState("demo")
    public = state.feed(ResponsesEncoder().chunks([response_completed([native])]))
    public.extend(state.finish())
    with pytest.raises(SemanticLossError) as captured:
        ResponsesDecoder().chunks(public, semantic_profile=profile, **state.metadata)
    output = captured.value.draft
    assert output[-1]["response"]["output"] == []
    assert not [event for event in output if event["type"].startswith("response.output_item.")]


def test_replay_uses_feedback_item_when_visible_reasoning_is_edited(profile_factory):
    native = reasoning_item("original reasoning")
    messages = ResponsesEncoder().messages([native])
    messages[0].content[0].reasoning = "edited visible summary"
    assert ResponsesDecoder().messages(messages, profile_factory("openai_responses")) == [native]


def test_feedback_path_is_applied_to_completed_item_snapshots(profile_factory):
    native = reasoning_item("visible")
    selected = {**native, "encrypted_content": "selected-opaque"}
    source_item = {**native, "vendor": {"feedback": selected}}
    profile = profile_factory(
        "openai_responses", feedback_reasoning_item=OutputParameter(path="$.item.vendor.feedback")
    )
    state = StreamState("demo")
    encoder = ResponsesEncoder()
    state.feed(
        encoder.chunks(
            [{"type": "response.output_item.done", "output_index": 0, "item": source_item}], profile
        )
    )
    state.feed(encoder.chunks([response_completed([source_item])], profile))
    state.finish()
    block = state.messages()[0]["content"][0]
    assert block["reasoning"] == "visible"
    assert block["feedback_item"] == selected


def test_duplicate_completed_item_updates_do_not_repeat_visible_deltas():
    native = reasoning_item("visible")
    state = StreamState("demo")
    encoder = ResponsesEncoder()
    done = {"type": "response.output_item.done", "output_index": 0, "item": native}
    first = state.feed(encoder.chunks([done]))
    repeated = state.feed(encoder.chunks([done]))
    final = state.feed(encoder.chunks([response_completed([native])]))
    final.extend(state.finish())
    assert [event["value"] for event in first if event["type"] == "delta"] == ["visible"]
    assert all(event["type"] != "delta" for event in [*repeated, *final])
    assert len(state.messages()[0]["content"]) == 1
    assert state.messages()[0]["content"][0]["feedback_item"] == native


def test_native_reasoning_text_content_uses_the_visible_reasoning_stream():
    native = reasoning_item(encrypted=None)
    native["content"] = [{"type": "reasoning_text", "text": "raw visible text"}]
    state = StreamState("demo")
    encoder = ResponsesEncoder()
    public = state.feed(
        encoder.chunks(
            [
                {
                    "type": "response.reasoning_text.delta",
                    "output_index": 0,
                    "item_id": native["id"],
                    "delta": "raw visible text",
                }
            ]
        )
    )
    assert [event["value"] for event in public if event["type"] == "delta"] == ["raw visible text"]
    state.feed(encoder.chunks([response_completed([native])]))
    state.finish()
    block = state.messages()[0]["content"][0]
    assert block["reasoning"] == "raw visible text"
    assert block["feedback_item"] == native


def test_scalar_feedback_reports_semantic_loss_instead_of_failing_to_render(profile_factory):
    message = Message(
        "assistant", [ReasoningContent("visible", "opaque")], protocol="openai_responses"
    )
    with pytest.raises(SemanticLossError, match="native reasoning item") as captured:
        ResponsesDecoder().messages([message], profile_factory("openai_responses"))
    assert all(item.get("type") != "reasoning" for item in captured.value.draft)


def test_scalar_feedback_from_custom_path_reports_loss_on_native_stream_finish(profile_factory):
    native = reasoning_item("visible")
    profile = profile_factory(
        "openai_responses", feedback_reasoning_item=OutputParameter(path="$.item.encrypted_content")
    )
    state = StreamState("demo")
    state.feed(ResponsesEncoder().chunks([response_completed([native])], profile))
    public = state.finish()
    with pytest.raises(SemanticLossError, match="native reasoning item") as captured:
        ResponsesDecoder().chunks(public, **state.metadata)
    final_item = captured.value.draft[-1]["response"]["output"][0]
    assert final_item["summary"] == [{"type": "summary_text", "text": "visible"}]
    assert "encrypted_content" not in final_item


def test_source_protocol_context_owns_a_bare_middleware_feedback_item():
    native = reasoning_item("visible")
    message = Message("assistant", [ReasoningContent("visible", native)])
    assert ResponsesDecoder().messages([message], source_protocol="openai_responses") == [native]


def test_native_stream_output_owns_nested_feedback_item_content():
    native = reasoning_item("visible")
    state = StreamState("demo")
    state.feed(ResponsesEncoder().chunks([response_completed([native])]))
    public = state.finish()
    before = deepcopy(public)
    output = ResponsesDecoder().chunks(public, **state.metadata)
    final_item = output[-1]["response"]["output"][0]
    final_item["summary"][0]["text"] = "changed by consumer"
    assert public == before
    assert state.messages()[0]["content"][0]["feedback_item"] == native


@pytest.mark.parametrize("completed", [False, True])
def test_encoded_stream_batches_own_their_native_feedback_items(completed):
    native = reasoning_item("visible")
    event = (
        response_completed([native])
        if completed
        else {"type": "response.output_item.done", "output_index": 0, "item": native}
    )
    encoded = ResponsesEncoder().chunks([event])
    operation = next(
        op for op in encoded[0].events if op["type"] == ("snapshot" if completed else "block_set")
    )
    block = operation["blocks"][0]["block"] if completed else operation["block"]
    native["summary"][0]["text"] = "source mutation"
    assert block["feedback_item"]["summary"][0]["text"] == "visible"
    block["feedback_item"]["summary"].append({"type": "summary_text", "text": "owned"})
    assert len(native["summary"]) == 1


def test_foreign_opaque_only_reasoning_does_not_create_an_empty_response_item():
    block = {
        "type": "thinking",
        "reasoning": None,
        "feedback_item": {"type": "redacted_thinking", "data": "opaque"},
    }
    public = [
        {"type": "block", "index": 0, "block": block},
        {
            "type": "finish",
            "reason": "stop",
            "messages": [],
            "output_blocks": [{"index": 0, "block": block}],
        },
    ]
    with pytest.raises(SemanticLossError, match="feedback_item") as captured:
        ResponsesDecoder().chunks(public, source_protocol="anthropic")
    assert captured.value.draft[-1]["response"]["output"] == []
    assert all(
        not event["type"].startswith("response.output_item") for event in captured.value.draft
    )
