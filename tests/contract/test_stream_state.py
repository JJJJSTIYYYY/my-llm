"""Accumulation invariants across native event boundaries and JSON round trips.

Protocol sources: docs/design/stream-state.md. Fixtures below are written as
native events, independently of the operation handlers under test.
"""

import json
from copy import deepcopy

import pytest

from my_llm import IncompleteStreamError, MessageValidationError, MyCodec
from my_llm.codec.shared.stream import metadata
from my_llm.stream._state import StreamState
from tests.helpers.protocols import PROTOCOLS, chat_event, response_completed, stream_events
from tests.helpers.sdk import EVENT_TYPES, assert_native
from tests.helpers.streaming import render


def test_metadata_paths_merge_projected_values_without_protocol_field_knowledge():
    state = StreamState("demo")
    payload = {"label": None, "application": {"values": [1]}}
    state.feed([metadata(payload, path=("session", "first"))])
    original_session = state.metadata["session"]
    original_first = original_session["first"]
    original_application = original_first["application"]
    payload["application"]["values"].append(2)
    assert original_first == {"label": None, "application": {"values": [1]}}

    state.feed([metadata({"phase": "complete"}, path=("session", "first"))])
    current = state.metadata["session"]["first"]
    assert state.metadata["session"] is not original_session
    assert current is not original_first
    assert current == {**original_first, "phase": "complete"}
    assert current["application"] is original_application
    assert "phase" not in original_first

    # An explicitly supplied complete value replaces the previous subtree.
    state.feed([metadata({"session": {"replacement": None}})])
    assert state.metadata["session"] == {"replacement": None}
    assert state.metadata["model"] == "demo"


def test_metadata_operation_survives_json_round_trip_and_captures_each_update():
    state = StreamState("demo")
    operation = json.loads(json.dumps(metadata({"items": [{"value": 1}]}, path=("session",))))
    state.feed([operation])
    operation["values"]["items"][0]["value"] = 2
    assert state.metadata["session"] == {"items": [{"value": 1}]}


def validate_events(protocol, events):
    for event in events:
        schema = EVENT_TYPES[protocol]
        assert_native(event, schema[event["type"]] if isinstance(schema, dict) else schema)


def response_message(index, texts):
    return {
        "type": "message",
        "id": f"msg_{index}",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "logprobs": [],
                "type": "output_text",
                "text": text,
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": len(text),
                        "title": f"citation_{index}_{part}",
                        "url": f"https://example.com/{index}/{part}",
                    }
                ],
            }
            for part, text in enumerate(texts)
        ],
    }


def response_text_events(output):
    events = [stream_events("openai_responses")[0]]
    for index, item in enumerate(output):
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "status": "in_progress", "content": []},
            }
        )
        for part, content in enumerate(item["content"]):
            address = {"item_id": item["id"], "output_index": index, "content_index": part}
            events.extend(
                [
                    {
                        "type": "response.content_part.added",
                        **address,
                        "part": {
                            "logprobs": [],
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                    {
                        "type": "response.output_text.delta",
                        **address,
                        "delta": content["text"],
                        "logprobs": [],
                    },
                    {"type": "response.content_part.done", **address, "part": content},
                ]
            )
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append(response_completed(output))
    for sequence, event in enumerate(events):
        event["sequence_number"] = sequence
    return events


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("batch_size", [1, 100])
def test_accumulation_survives_json_round_trips_and_batch_boundaries(source, batch_size):
    codec, state, emitted = MyCodec(), StreamState("demo"), []
    native = stream_events(source, tool=True)
    before = deepcopy(native)
    for start in range(0, len(native), batch_size):
        encoded = codec.encode(chunks=native[start : start + batch_size], input_format=source)[
            "chunks"
        ]
        emitted.extend(state.feed(json.loads(json.dumps(encoded))))
    emitted.extend(state.finish())
    output = codec.decode(chunks=emitted, target_format="anthropic")["chunks"]
    validate_events("anthropic", output)
    reconstructed = StreamState("demo")
    reconstructed.feed(codec.encode(chunks=output, input_format="anthropic")["chunks"])
    reconstructed.finish()
    collected = codec.decode(state.messages(), target_format="openai_chat")["messages"]
    replayed = codec.decode(reconstructed.messages(), target_format="openai_chat")["messages"]
    assert replayed == collected
    assert native == before


@pytest.mark.parametrize("parts", [1, 2, 100])
def test_text_and_tool_results_do_not_depend_on_fragment_partition(parts):
    text = "中文🙂 and quoted text"
    argument_text = json.dumps({"city": "上海", "nested": {"values": [1, 2]}}, ensure_ascii=False)

    def split(value):
        width = max(1, len(value) // parts)
        return ["", *(value[i : i + width] for i in range(0, len(value), width)), ""]

    events = [chat_event(content=fragment) for fragment in split(text)]
    for index, fragment in enumerate(split(argument_text)):
        call = {"index": 0, "function": {"arguments": fragment}}
        if index == 0:
            call.update(id="call_1", type="function")
            call["function"]["name"] = "weather"
        events.append(chat_event(calls=[call]))
    events.append(chat_event(finish="tool_calls"))
    emitted, state = render("openai_chat", "openai_chat", events)
    expected = {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": argument_text},
            }
        ],
    }
    assert MyCodec().decode(state.messages(), target_format="openai_chat")["messages"] == [expected]
    replayed = StreamState("demo")
    replayed.feed(MyCodec().encode(chunks=emitted, input_format="openai_chat")["chunks"])
    replayed.finish()
    assert replayed.messages() == state.messages()


def test_equal_text_keeps_citations_on_its_own_item_and_part():
    incoming = response_text_events(
        [response_message(0, ["same", "same"]), response_message(1, ["same"])]
    )
    validate_events("openai_responses", incoming)
    output, state = render("openai_responses", "openai_responses", incoming)
    validate_events("openai_responses", output)
    done = [event["item"] for event in output if event["type"] == "response.output_item.done"]
    assert [part["annotations"][0]["title"] for item in done for part in item["content"]] == [
        "citation_0_0",
        "citation_0_1",
        "citation_1_0",
    ]
    assert done == output[-1]["response"]["output"]
    assert (
        MyCodec().decode(state.messages(), target_format="openai_responses")["messages"]
        == incoming[-1]["response"]["output"]
    )


@pytest.mark.parametrize("live_text", ["", "A", "AB"])
def test_snapshot_supplements_each_unstreamed_block_and_suffix(live_text):
    output = [response_message(0, ["ABC", "DEF"]), response_message(1, ["GHI"])]
    incoming = [
        {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_0",
            "delta": live_text,
            "logprobs": [],
        },
        response_completed(output),
    ]
    converted, _ = render("openai_responses", "openai_chat", incoming)
    assert (
        "".join(
            choice["delta"].get("content", "") for event in converted for choice in event["choices"]
        )
        == "ABCDEFGHI"
    )


def test_snapshot_cannot_silently_replace_already_emitted_text():
    codec, state = MyCodec(), StreamState("demo")
    state.feed(
        codec.encode(
            chunks=[{"type": "response.output_text.delta", "delta": "old"}],
            input_format="openai_responses",
        )["chunks"]
    )
    with pytest.raises(IncompleteStreamError, match="contradicts"):
        state.feed(
            codec.encode(
                chunks=[response_completed([response_message(0, ["new"])])],
                input_format="openai_responses",
            )["chunks"]
        )
    assert not state.finished
    assert state.messages(partial=True)[0]["content"][0]["text"] == "old"


def anthropic_events(blocks):
    events = [stream_events("anthropic")[0]]
    for index, (block, deltas) in enumerate(blocks):
        events.append({"type": "content_block_start", "index": index, "content_block": block})
        events.extend(
            {"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas
        )
        events.append({"type": "content_block_stop", "index": index})
    events.extend(stream_events("anthropic")[-2:])
    return events


def test_empty_thinking_and_redacted_blocks_keep_order_and_signatures():
    incoming = anthropic_events(
        [
            ({"type": "redacted_thinking", "data": "opaque"}, []),
            (
                {"type": "thinking", "thinking": "", "signature": ""},
                [
                    {"type": "thinking_delta", "thinking": ""},
                    {"type": "signature_delta", "signature": "signed"},
                ],
            ),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "visible"}]),
        ]
    )
    validate_events("anthropic", incoming)
    output, state = render("anthropic", "anthropic", incoming)
    validate_events("anthropic", output)
    starts = [
        event["content_block"]["type"] for event in output if event["type"] == "content_block_start"
    ]
    assert starts == ["redacted_thinking", "thinking", "text"]
    signatures = [
        event["delta"]["signature"]
        for event in output
        if event.get("delta", {}).get("type") == "signature_delta"
    ]
    assert signatures == ["signed"]
    assert state.messages()[0]["content"][1] == {
        "type": "thinking",
        "thinking": "",
        "signature": "signed",
    }
    reconstructed = StreamState("demo")
    reconstructed.feed(MyCodec().encode(chunks=output, input_format="anthropic")["chunks"])
    reconstructed.finish()
    assert reconstructed.messages() == state.messages()


@pytest.mark.parametrize("target", ["anthropic", "openai_responses", "gemini"])
def test_tool_and_following_text_wait_for_success_and_keep_source_order(target):
    incoming = anthropic_events(
        [
            (
                {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}},
                [
                    {"type": "input_json_delta", "partial_json": '{"city":'},
                    {"type": "input_json_delta", "partial_json": '"上海"}'},
                ],
            ),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "after"}]),
        ]
    )
    codec, state = MyCodec(), StreamState("demo")
    for event in incoming:
        assert state.feed(codec.encode(chunks=[event], input_format="anthropic")["chunks"]) == []
    final = state.finish()
    output = codec.decode(chunks=final, target_format=target)["chunks"]
    validate_events(target, output)
    assert [entry["block"]["type"] for entry in final[-1]["output_blocks"]] == ["tool_use", "text"]
    reconstructed = StreamState("demo")
    reconstructed.feed(codec.encode(chunks=output, input_format=target)["chunks"])
    reconstructed.finish()
    chat = codec.decode(reconstructed.messages(), target_format="openai_chat")["messages"][0]
    assert chat["content"] == "after"
    assert json.loads(chat["tool_calls"][0]["function"]["arguments"]) == {"city": "上海"}
    if target == "anthropic":
        assert [
            event["content_block"]["type"]
            for event in output
            if event["type"] == "content_block_start"
        ] == ["tool_use", "text"]
    elif target == "openai_responses":
        assert [item["type"] for item in output[-1]["response"]["output"]] == [
            "function_call",
            "message",
        ]
    else:
        parts = [
            part
            for event in output
            for candidate in event["candidates"]
            for part in candidate["content"]["parts"]
        ]
        assert "function_call" in parts[0] and parts[1]["text"] == "after"


@pytest.mark.parametrize("thinking", [False, True])
def test_signed_empty_gemini_parts_are_not_dropped_or_merged(thinking):
    parts = [
        {"text": "", "thought_signature": "c2lnMQ=="},
        {"text": "", "thought_signature": "c2lnMg=="},
    ]
    if thinking:
        for part in parts:
            part["thought"] = True
    incoming = [
        {"candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": "STOP"}]}
    ]
    validate_events("gemini", incoming)
    output, state = render("gemini", "gemini", incoming)
    validate_events("gemini", output)
    streamed = [
        part
        for event in output
        for candidate in event["candidates"]
        for part in candidate["content"]["parts"]
    ]
    assert streamed == parts
    assert (
        MyCodec().decode(state.messages(), target_format="gemini")["messages"][0]["parts"] == parts
    )


def test_reasoning_summary_parts_use_their_native_summary_indexes():
    item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [
            {"type": "summary_text", "text": "first"},
            {"type": "summary_text", "text": "second"},
        ],
        "encrypted_content": "opaque",
    }
    incoming = [stream_events("openai_responses")[0]]
    for index, part in enumerate(item["summary"]):
        address = {"item_id": item["id"], "output_index": 0, "summary_index": index}
        incoming.extend(
            [
                {
                    "type": "response.reasoning_summary_part.added",
                    **address,
                    "part": {"type": "summary_text", "text": ""},
                },
                {"type": "response.reasoning_summary_text.delta", **address, "delta": part["text"]},
                {"type": "response.reasoning_summary_part.done", **address, "part": part},
            ]
        )
    incoming.append(response_completed([item]))
    for sequence, event in enumerate(incoming):
        event["sequence_number"] = sequence
    validate_events("openai_responses", incoming)
    output, state = render("openai_responses", "openai_responses", incoming)
    validate_events("openai_responses", output)
    done = [event["item"] for event in output if event["type"] == "response.output_item.done"]
    assert done == [item]
    assert [
        event["summary_index"]
        for event in output
        if event["type"] == "response.reasoning_summary_part.done"
    ] == [0, 1]
    assert done == output[-1]["response"]["output"]
    assert MyCodec().decode(state.messages(), target_format="openai_responses")["messages"] == [
        item
    ]


def test_invalid_tool_json_does_not_mark_stream_finished_or_mask_partial_text():
    codec, state = MyCodec(), StreamState("demo")
    native = [
        chat_event(content="partial"),
        chat_event(
            calls=[{"index": 0, "id": "call_1", "function": {"name": "weather", "arguments": "{"}}]
        ),
        chat_event(finish="length"),
    ]
    emitted = state.feed(codec.encode(chunks=native, input_format="openai_chat")["chunks"])
    assert all(event["type"] != "tool" for event in emitted)
    assert state.messages(partial=True)[0]["content"] == [{"type": "text", "text": "partial"}]
    with pytest.raises(MessageValidationError, match="JSON object"):
        state.finish()
    assert not state.finished


def test_missing_terminal_and_post_finish_updates_are_rejected():
    codec, state = MyCodec(), StreamState("demo")
    state.feed(
        codec.encode(chunks=[chat_event(content="partial")], input_format="openai_chat")["chunks"]
    )
    with pytest.raises(IncompleteStreamError, match="terminal event"):
        state.finish()
    state.feed(
        codec.encode(chunks=[chat_event(finish="stop")], input_format="openai_chat")["chunks"]
    )
    state.finish()
    with pytest.raises(IncompleteStreamError, match="already been finalized"):
        state.finish()
    with pytest.raises(IncompleteStreamError, match="already been finalized"):
        state.feed(
            codec.encode(chunks=[chat_event(content="late")], input_format="openai_chat")["chunks"]
        )


@pytest.mark.parametrize("final_texts", [[], ["kept"]])
def test_snapshot_cannot_omit_previously_streamed_blocks(final_texts):
    codec, state = MyCodec(), StreamState("demo")
    output = [response_message(0, ["kept"]), response_message(1, ["removed"])]
    live = response_text_events(output)[:-1]
    state.feed(codec.encode(chunks=live, input_format="openai_responses")["chunks"])
    final = [response_message(0, final_texts)] if final_texts else []
    with pytest.raises(IncompleteStreamError, match="omits streamed blocks"):
        state.feed(
            codec.encode(chunks=[response_completed(final)], input_format="openai_responses")[
                "chunks"
            ]
        )
    assert not state.finished
    assert [block["text"] for block in state.messages(partial=True)[0]["content"]] == [
        "kept",
        "removed",
    ]


@pytest.mark.parametrize(
    "content", ["native extension", {"nested": [1, 2]}, [{"type": "text", "text": "extension"}]]
)
def test_unrecognized_output_item_does_not_enter_intermediate_or_native_output(content):
    item = {"type": "future_output", "id": "future_1", "content": content}
    output, state = render("openai_responses", "openai_responses", [response_completed([item])])
    assert output[-1]["response"]["output"] == []
    assert state.messages()[0]["content"] == []
    decoded = MyCodec().decode(state.messages(), target_format="openai_responses")["messages"]
    assert all(
        part["type"] == "output_text" and part["text"] == ""
        for message in decoded
        for part in message["content"]
    )
    assert "future_output" not in str(decoded)
    assert "future_1" not in str(decoded)
