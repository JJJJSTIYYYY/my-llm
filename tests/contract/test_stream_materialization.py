"""Request-local materialization reuse without changing public value ownership."""

from copy import deepcopy

import pytest

from my_llm import MessageValidationError, MyCodec
from my_llm.codec.shared.conversion import SOURCE
from my_llm.codec.shared.stream import BlockRef
from my_llm.stream import _state
from my_llm.stream._state import StreamState
from tests.helpers.protocols import PROTOCOLS


def complete_tool_stream():
    """Use explicit internal operations so SDK encoders cannot pre-parse JSON."""
    text, tool = BlockRef("text"), BlockRef("tool")
    return [
        {"type": "block_set", "ref": text, "block": {"type": "text", "text": ""}},
        {"type": "block_delta", "ref": text, "kind": "text", "field": "text", "value": "done"},
        {"type": "block_stop", "ref": text},
        {
            "type": "block_set",
            "ref": tool,
            "block": {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
        },
        {
            "type": "block_delta",
            "ref": tool,
            "kind": "tool_use",
            "field": "arguments",
            "value": '{"nested":',
        },
        {
            "type": "block_delta",
            "ref": tool,
            "kind": "tool_use",
            "field": "arguments",
            "value": '{"items":[1,2]}}',
        },
        {"type": "finish", "reason": "tool_calls"},
    ]


def test_completed_blocks_and_arguments_are_materialized_once(monkeypatch):
    parse = _state.arguments
    parsed = []

    def counted(value):
        parsed.append(value)
        return parse(value)

    monkeypatch.setattr(_state, "arguments", counted)
    state = StreamState("demo")
    live = state.feed(complete_tool_stream())
    closed_text = next(event["block"] for event in live if event["type"] == "block_stop")
    final = state.finish()
    finish = final[-1]
    text, tool = [entry["block"] for entry in finish["output_blocks"]]
    published_tool = next(event["block"] for event in final if event["type"] == "tool")

    assert text is closed_text
    assert finish["messages"][0]["content"] == [text, tool]
    assert finish["messages"][0]["content"][1] is tool
    assert published_tool["input"] is tool["input"]
    expected = state.messages()
    exported = state.messages()
    exported[0]["content"][1]["input"]["nested"]["items"].append(99)
    assert state.messages() == expected
    assert len(parsed) == 1


@pytest.mark.parametrize("source", [None, *PROTOCOLS])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_public_conversions_do_not_expose_cached_terminal_material(source, target):
    state, codec = StreamState("demo"), MyCodec()
    operations = complete_tool_stream()
    incoming = (
        operations if source is None else [{SOURCE: {"protocol": source, "events": operations}}]
    )
    events = [*state.feed(incoming), *state.finish()]
    assert state.source_protocol == source
    expected = state.messages()

    def corrupt(value):
        if isinstance(value, dict):
            for child in tuple(value.values()):
                corrupt(child)
            value["caller_mutation"] = True
        elif isinstance(value, list):
            for child in value:
                corrupt(child)
            value.append("caller_mutation")

    messages = codec.decoder(target).messages(events[-1]["messages"])
    corrupt(messages)
    assert state.messages() == expected, f"{source} -> {target}: messages exposed cache"
    result = codec.result(events[-1]["messages"], target_format=target)
    corrupt(result)
    assert state.messages() == expected, f"{source} -> {target}: result exposed cache"
    chunks = codec.decode(chunks=events, target_format=target, source_protocol=source)["chunks"]
    corrupt(chunks)
    assert state.messages() == expected, f"{source} -> {target}: chunks exposed cache"


def test_nested_updates_leave_previous_materialized_version_unchanged():
    state = StreamState("demo")
    ref = BlockRef("text")
    block = {"type": "text", "text": "", "citations": [{"title": "first"}], "extra": {"a": 1}}
    state.feed([{"type": "block_set", "ref": ref, "block": block}])
    block["citations"][0]["title"] = "external mutation"
    previous = state._blocks[0].materialize()
    before = deepcopy(previous)
    state.feed(
        [
            {
                "type": "block_annotation",
                "ref": ref,
                "field": "citations",
                "value": {"title": "second"},
            },
            {"type": "block_update", "ref": ref, "block": {"extra": {"b": 2}}, "merge": ["extra"]},
            {"type": "block_delta", "ref": ref, "kind": "text", "field": "text", "value": "next"},
            {"type": "finish", "reason": "stop"},
        ]
    )
    state.finish()
    assert previous == before
    assert state.messages()[0]["content"] == [
        {
            "type": "text",
            "text": "next",
            "citations": [{"title": "first"}, {"title": "second"}],
            "extra": {"a": 1, "b": 2},
        }
    ]


def test_failed_terminal_materialization_keeps_partial_result_available():
    operations = complete_tool_stream()
    operations[-2]["value"] = "invalid"
    state = StreamState("demo")
    state.feed(operations)
    partial = state.messages(partial=True)
    with pytest.raises(MessageValidationError, match="JSON object"):
        state.finish()
    assert not state.finished
    assert state.messages(partial=True) == partial
    assert partial[0]["content"] == [{"type": "text", "text": "done"}]
