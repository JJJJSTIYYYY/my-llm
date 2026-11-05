"""Independent regressions for the 2026-10-17 official protocol audit."""

import json
from copy import deepcopy

import pytest

from my_llm import (
    IncompleteStreamError,
    MyCodec,
    SemanticLossError,
)
from my_llm.runtime.state import StreamState
from tests.helpers.profiles import config
from tests.helpers.protocols import PROTOCOLS, response_completed, stream_events
from tests.helpers.sdk import (
    EVENT_TYPES,
    MESSAGE_TYPES,
    TOOL_TYPES,
    assert_native,
    assert_output_native,
)
from tests.helpers.streaming import render
from tests.helpers.tool_results import application_result_history

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="


def native_events(protocol, events, *, source=False):
    for event in events:
        schema = EVENT_TYPES[protocol]
        projected = event
        if protocol == "openai_chat" and not source:
            projected = deepcopy(event)
            for choice in projected.get("choices", []):
                reasoning = choice.get("delta", {}).pop("reasoning_content", None)
                assert reasoning is None or isinstance(reasoning, str)
        (assert_native if source else assert_output_native)(
            projected, schema[event["type"]] if isinstance(schema, dict) else schema
        )


def refusal_events(source):
    incoming = stream_events(source)
    if source == "openai_chat":
        incoming[0]["choices"][0]["delta"] = {"role": "assistant"}
        incoming[1]["choices"][0]["delta"] = {"refusal": "No"}
        return incoming
    for event in incoming:
        kind = event["type"]
        if kind in ("response.content_part.added", "response.content_part.done"):
            event["part"] = {"type": "refusal", "refusal": "" if kind.endswith("added") else "No"}
        elif kind == "response.output_text.delta":
            event.update(type="response.refusal.delta", delta="No")
            event.pop("logprobs", None)
        elif kind == "response.output_text.done":
            event.update(type="response.refusal.done", refusal="No")
            event.pop("text", None)
            event.pop("logprobs", None)
        if kind == "response.output_item.done":
            event["item"]["content"] = [{"type": "refusal", "refusal": "No"}]
        if kind == "response.completed":
            event["response"]["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    return incoming


@pytest.mark.parametrize("source", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_a03_refusal_is_accumulated_inside_target_lifecycle(source, target):
    incoming = refusal_events(source)
    native_events(source, incoming, source=True)
    output, state = render(source, target, incoming)
    native_events(target, output)
    assert '"No"' not in json.dumps(
        MyCodec().decode(state.messages(), target_format=target)["messages"]
    )
    assert "refusal" not in str(output)
    assert "refusal" not in str(state.messages())
    if target == "openai_responses":
        assert output[0]["type"] == "response.created"
        assert output[-1]["type"] == "response.completed"
        assert [event["sequence_number"] for event in output] == list(range(len(output)))


def test_a04_stop_sequence_is_preserved_natively_and_translates_to_chat_stop():
    incoming = stream_events("anthropic")
    incoming[-2]["delta"] = {"stop_reason": "stop_sequence", "stop_sequence": "END"}
    for target in ("anthropic", "openai_chat"):
        output, _ = render("anthropic", target, incoming)
        native_events(target, output)
        if target == "anthropic":
            assert output[-2]["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "END"}
        else:
            assert (
                next(
                    event["choices"][0]["finish_reason"]
                    for event in output
                    if event.get("choices") and event["choices"][0]["finish_reason"] is not None
                )
                == "stop"
            )


def test_a07_citation_deltas_are_outside_the_supported_field_set():
    incoming = stream_events("anthropic")
    citation = {
        "type": "char_location",
        "cited_text": "source",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 0,
        "end_char_index": 6,
    }
    incoming.insert(
        3,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": citation},
        },
    )
    output, state = render("anthropic", "anthropic", incoming)
    native_events("anthropic", output)
    collected = MyCodec().decode(state.messages(), target_format="anthropic")["messages"][0]
    assert collected["content"][0] == {"type": "text", "text": "Hello"}
    assert "citations" not in str(output)
    assert "citation" not in str(state.messages())


@pytest.mark.parametrize(
    "source,incoming",
    [("openai_chat", [{"role": "assistant", "content": None, "reasoning_content": "Thinking"}])],
)
def test_a09_foreign_thinking_does_not_forge_an_anthropic_signature(source, incoming):
    with pytest.raises(SemanticLossError) as captured:
        MyCodec().convert(incoming, input_format=source, target_format="anthropic")
    blocks = captured.value.draft["messages"][0]["content"]
    thinking = next(block for block in blocks if block["type"] == "thinking")
    assert thinking["thinking"] == "Thinking"
    assert thinking["signature"] == ""
    assert "Thinking" in str(captured.value.losses.items[0].origin_value)


def test_a09_foreign_thinking_stream_uses_anthropic_text_blocks():
    incoming = stream_events("openai_chat")
    incoming[1]["choices"][0]["delta"] = {"reasoning_content": "Thinking"}
    with pytest.raises(SemanticLossError) as captured:
        render("openai_chat", "anthropic", incoming)
    assert "signature_delta" not in str(captured.value.draft)
    assert "Thinking" not in str(captured.value.draft)


def test_a10_document_url_is_silently_discarded():
    incoming = [
        {
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}}
            ],
        }
    ]
    result = MyCodec().convert(incoming, input_format="anthropic", target_format="openai_chat")[
        "messages"
    ]
    assert result == [{"role": "user", "content": ""}]
    assert_native(result[0], MESSAGE_TYPES["openai_chat"])


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a11_duplicate_native_terminals_are_rejected(protocol):
    incoming = stream_events(protocol)
    terminal = incoming[-2] if protocol == "openai_chat" else incoming[-1]
    with pytest.raises(IncompleteStreamError, match="terminal"):
        render(protocol, protocol, [*incoming, deepcopy(terminal)])


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a11_content_after_native_terminal_is_rejected(protocol):
    incoming = stream_events(protocol)
    late = {
        "openai_chat": {
            "choices": [{"index": 0, "delta": {"content": "late"}, "finish_reason": None}]
        },
        "openai_responses": {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "late",
        },
        "anthropic": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "late"},
        },
    }[protocol]
    with pytest.raises(IncompleteStreamError, match="terminal"):
        render(protocol, protocol, [*incoming, late])


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_foreign_thinking_streams_can_be_consumed_without_forged_signatures(source, target):
    native = stream_events(source)
    if source == "openai_chat":
        native[1]["choices"][0]["delta"] = {"reasoning_content": "Thinking"}
    elif source == "anthropic":
        native[1]["content_block"] = {"type": "thinking", "thinking": "", "signature": ""}
        native[2]["delta"] = {"type": "thinking_delta", "thinking": "Thinking"}
        native.insert(
            3,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "signed"},
            },
        )
    else:
        item = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Thinking"}],
            "encrypted_content": "opaque",
        }
        address = {"item_id": "rs_1", "output_index": 0, "summary_index": 0}
        native = [
            native[0],
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "summary": []},
            },
            {
                "type": "response.reasoning_summary_part.added",
                **address,
                "part": {"type": "summary_text", "text": ""},
            },
            {"type": "response.reasoning_summary_text.delta", **address, "delta": "Thinking"},
            {"type": "response.reasoning_summary_part.done", **address, "part": item["summary"][0]},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            response_completed([item]),
        ]
        for index, event in enumerate(native):
            event["sequence_number"] = index
    if source != target:
        with pytest.raises(SemanticLossError):
            render(source, target, native)
        return
    output, _ = render(source, target, native)
    native_events(target, output)
    reconstructed = StreamState("demo")
    reconstructed.feed(MyCodec().encode(chunks=output, input_format=target)["chunks"])
    reconstructed.finish()
    assert "Thinking" in str(reconstructed.messages())
    if source != target and target in ("anthropic",):
        assert not any(event.get("delta", {}).get("type") == "signature_delta" for event in output)


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_foreign_message_extensions_cannot_become_native_request_fields(
    source, target, profile_factory
):
    from tests.helpers.protocols import history

    incoming = history(source)
    incoming[0]["vendor_metadata"] = {"nested": [1, 2]}
    before = deepcopy(incoming)
    converted = MyCodec().convert(incoming, input_format=source, target_format=target)["messages"]
    assert all("vendor_metadata" not in message for message in converted)
    filtered = (
        MyCodec()
        .decoder(target)
        .filter_messages(
            converted,
            profile_factory(
                target,
                supports_function_calling=config(True),
                should_feedback_reasoning_item=config("none"),
            ),
        )
    )
    assert all("vendor_metadata" not in message for message in filtered)
    for message in filtered:
        assert_native(message, MESSAGE_TYPES[target])
    assert incoming == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_foreign_tool_extensions_cannot_become_native_request_fields(
    source, target, profile_factory
):
    from tests.helpers.protocols import tools

    incoming = tools(source)
    incoming[0]["vendor_metadata"] = {"nested": [1, 2]}
    before = deepcopy(incoming)
    converted = MyCodec().convert(tools=incoming, input_format=source, target_format=target)[
        "tools"
    ]
    assert all("vendor_metadata" not in tool for tool in converted)
    filtered = MyCodec().decoder(target).filter_tools(converted, profile_factory(target))
    assert all("vendor_metadata" not in tool for tool in filtered)
    for tool in filtered:
        assert_native(tool, TOOL_TYPES[target])
    assert incoming == before


def test_a07_deferred_citations_are_discarded_without_changing_text():
    citation = {
        "type": "char_location",
        "cited_text": "source",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 0,
        "end_char_index": 6,
    }
    incoming = [
        stream_events("anthropic")[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "c", "name": "f", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "citations_delta", "citation": citation},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
    output, state = render("anthropic", "anthropic", incoming)
    native_events("anthropic", output)
    deltas = [
        event["delta"]["type"]
        for event in output
        if event.get("index") == 1 and event["type"] == "content_block_delta"
    ]
    assert deltas == ["text_delta"]
    assert "citations" not in str(state.messages())
    reconstructed = StreamState("demo")
    reconstructed.feed(MyCodec().encode(chunks=output, input_format="anthropic")["chunks"])
    reconstructed.finish()
    assert "citations" not in str(reconstructed.messages())
    assert "Hello" in str(reconstructed.messages())


@pytest.mark.parametrize(
    "protocol,tail",
    [
        (
            "openai_chat",
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            },
        )
    ],
)
def test_a11_documented_usage_tails_remain_legal(protocol, tail):
    _, state = render(protocol, "openai_chat", [*stream_events(protocol), tail])
    usage = MyCodec().decoder("openai_chat").usage(state.usage)
    assert usage == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}


def test_multimodal_tool_results_keep_compatible_responses_parts():
    history = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c",
                    "content": [
                        {"type": "text", "text": "Image"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                        },
                    ],
                }
            ],
        },
    ]
    codec = MyCodec()
    converted = codec.convert(history, input_format="anthropic", target_format="openai_responses")[
        "messages"
    ]
    assert converted[-1] == {
        "type": "function_call_output",
        "call_id": "c",
        "output": [
            {"type": "input_text", "text": "Image"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}", "detail": "auto"},
        ],
    }
    for item in converted:
        assert_native(item, MESSAGE_TYPES["openai_responses"])
    restored = codec.convert(converted, input_format="openai_responses", target_format="anthropic")[
        "messages"
    ]
    assert restored[-1]["content"] == history[-1]["content"]


@pytest.mark.parametrize(
    "field",
    [
        "access_programs",
        "error",
        "incomplete_details",
        "instructions",
        "metadata",
        "temperature",
        "top_p",
    ],
)
def test_latest_response_required_fields_cannot_be_hidden_by_sdk_defaults(field):
    from jsonschema import ValidationError

    from tests.helpers.schema import validate_event

    output, _ = render("openai_chat", "openai_responses", stream_events("openai_chat"))
    complete = deepcopy(output[-1])
    complete["response"].pop(field)
    with pytest.raises(ValidationError):
        validate_event("openai_responses", complete)


def test_latest_response_metadata_is_discarded_and_required_fields_use_empty_defaults():
    incoming = stream_events("openai_responses")
    metadata = {
        "temperature": 0.2,
        "top_p": 0.7,
        "metadata": {"case": "native"},
        "instructions": "Retain context",
    }
    for event in incoming:
        if "response" in event:
            event["response"].update(metadata)
    output, _ = render("openai_responses", "openai_responses", incoming)
    native_events("openai_responses", output)
    for frame in [output[0], output[-1]]:
        assert frame["response"]["metadata"] == {}
        assert frame["response"]["instructions"] is None
        assert frame["response"]["temperature"] is None
        assert frame["response"]["top_p"] is None
    parts = [
        event["part"]
        for event in output
        if event["type"] in ("response.content_part.added", "response.content_part.done")
    ]
    assert all(part["logprobs"] == [] for part in parts)


def test_native_response_reasoning_and_tool_item_ids_remain_replayable():
    reasoning = {
        "type": "reasoning",
        "id": "rs_original",
        "status": "completed",
        "summary": [
            {"type": "summary_text", "text": "first"},
            {"type": "summary_text", "text": "second"},
        ],
        "encrypted_content": "original-encrypted-payload",
    }
    call = {
        "type": "function_call",
        "id": "fc_original",
        "call_id": "call_original",
        "name": "f",
        "arguments": '{"x":1}',
        "status": "completed",
    }
    native = [
        stream_events("openai_responses")[0],
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**reasoning, "status": "in_progress", "summary": []},
        },
    ]
    for index, part in enumerate(reasoning["summary"]):
        address = {"item_id": "rs_original", "output_index": 0, "summary_index": index}
        native += [
            {
                "type": "response.reasoning_summary_part.added",
                **address,
                "part": {"type": "summary_text", "text": ""},
            },
            {"type": "response.reasoning_summary_text.delta", **address, "delta": part["text"]},
            {"type": "response.reasoning_summary_part.done", **address, "part": part},
        ]
    native += [
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**call, "status": "in_progress", "arguments": ""},
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "fc_original",
            "delta": call["arguments"],
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call},
        response_completed([reasoning, call]),
    ]
    for index, event in enumerate(native):
        event["sequence_number"] = index
    output, _ = render("openai_responses", "openai_responses", native)
    native_events("openai_responses", output)
    done = [event["item"] for event in output if event["type"] == "response.output_item.done"]
    assert done == output[-1]["response"]["output"]
    assert done[0] == reasoning
    assert {key: value for key, value in done[1].items() if key != "arguments"} == {
        key: value for key, value in call.items() if key != "arguments"
    }
    assert json.loads(done[1]["arguments"]) == json.loads(call["arguments"])
    assert [
        event["item"]["id"] for event in output if event["type"] == "response.output_item.added"
    ] == ["rs_original", "fc_original"]
    assert [
        event["summary_index"]
        for event in output
        if event["type"] == "response.reasoning_summary_text.delta"
    ] == [0, 0, 0]
    assert not any(event["type"] == "response.function_call_arguments.delta" for event in output)
    complete_arguments = [
        event["arguments"]
        for event in output
        if event["type"] == "response.function_call_arguments.done"
    ]
    assert [json.loads(value) for value in complete_arguments] == [json.loads(call["arguments"])]


def test_unsupported_response_progress_and_items_do_not_pass_through_as_opaque():
    item = {
        "type": "web_search_call",
        "id": "ws_original",
        "status": "completed",
        "action": {"type": "search", "query": "test"},
    }
    created = stream_events("openai_responses")[0]
    native = [
        created,
        {"type": "response.in_progress", "response": deepcopy(created["response"])},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "status": "in_progress"},
        },
        {
            "type": "response.web_search_call.in_progress",
            "output_index": 0,
            "item_id": "ws_original",
        },
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        response_completed([item]),
    ]
    for index, event in enumerate(native):
        event["sequence_number"] = index
    output, _ = render("openai_responses", "openai_responses", native)
    native_events("openai_responses", output)
    assert output[0]["type"] == "response.created"
    assert [event["sequence_number"] for event in output] == list(range(len(output)))
    assert not any(event["type"] == "response.web_search_call.in_progress" for event in output)
    assert not any(event["type"] == "response.output_item.added" for event in output)
    assert output[-1]["response"]["output"] == []


def test_response_stream_metadata_does_not_alias_request_state():
    codec, state = MyCodec(), StreamState("demo")
    created = stream_events("openai_responses")[0]
    created["response"]["metadata"] = {"source": "original"}
    encoded = codec.encode(chunks=[created], input_format="openai_responses")["chunks"]
    events = state.feed(encoded)
    output = codec.decode(chunks=events, target_format="openai_responses", **state.metadata)[
        "chunks"
    ]
    before = deepcopy(state.metadata)
    output[0]["response"]["metadata"]["source"] = "changed"
    assert state.metadata == before
    assert created["response"]["metadata"] == {"source": "original"}


@pytest.mark.parametrize("role", ["user", "assistant", "system", "tool"])
@pytest.mark.parametrize(
    "protocol, incoming",
    [
        (
            "openai_chat",
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                    {"type": "text", "text": "after"},
                ],
            },
        ),
        (
            "openai_responses",
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "before"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"},
                    {"type": "input_text", "text": "after"},
                ],
            },
        ),
        (
            "anthropic",
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                    },
                    {"type": "text", "text": "after"},
                ],
            },
        ),
    ],
)
def test_source_image_roles_preserve_text_order_tool_images_and_origins(protocol, incoming, role):
    incoming = deepcopy(incoming)
    if role == "tool":
        payload = incoming["content"]
        messages = application_result_history(protocol, payload)
    else:
        incoming["role"] = role
        messages = [incoming]
    original = deepcopy(messages)
    message = MyCodec().encode(messages, input_format=protocol).messages[-1]
    content = message.content
    assert "".join(part.text for part in content if part.type == "text") == ("beforeafter")
    images = [part for part in content if part.type == "image"]
    assert len(images) == (1 if role in ("user", "tool") else 0)
    if role in ("user", "tool"):
        assert [part.type for part in content] == ["text", "image", "text"]
    if role != "tool":
        field = "content"
        assert content[-1].origin.path == f"$.messages[0].{field}[2]"
        assert content[-1].origin.value is incoming[field][2]
    assert messages == original
