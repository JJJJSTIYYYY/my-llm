"""Every middleware field survives its neutral projection, without native syntax."""

from dataclasses import fields, is_dataclass

import pytest

import my_llm
from my_llm import (
    AssistantMessage,
    FunctionTool,
    ImageContent,
    ImageSource,
    InstructionMessage,
    ItemIdentity,
    Message,
    MessageEvent,
    MessageValidationError,
    Origin,
    ProtocolMiddleWare,
    ReasoningContent,
    StreamBatch,
    TextContent,
    ToolCall,
    ToolMessage,
    ToolResult,
    Usage,
    UserMessage,
)
from my_llm.middleware.models import content_from_wire, messages_from_wire, tools_from_wire
from my_llm.middleware.protocol import generation_input
from my_llm.middleware.stream import stream_from_wire


def assert_fields_exercised(model, supplied, *, implicit=()):
    """New fields must be added to populated roundtrip fixtures deliberately."""
    assert set(supplied) | set(implicit) == {field.name for field in fields(model)}


def assert_model_equal(actual, expected):
    assert type(actual) is type(expected)
    assert model_value(actual) == model_value(expected)


def model_value(value):
    if is_dataclass(value):
        return {
            field.name: model_value(getattr(value, field.name))
            for field in fields(value)
            if field.name != "origin"
        }
    if isinstance(value, (list, tuple)):
        return type(value)(model_value(child) for child in value)
    if isinstance(value, dict):
        return {key: model_value(child) for key, child in value.items()}
    return value


def identity():
    return ItemIdentity(id="item-7", status="completed")


CONTENT_CASES = [
    (TextContent, {"text": "answer", "item": identity()}),
    (
        ImageContent,
        {
            "source": ImageSource(
                type="url",
                url="https://example.com/image.png",
                data="aW1hZ2U=",
                media_type="image/png",
            ),
            "item": identity(),
        },
    ),
    *[
        (
            ReasoningContent,
            {
                "reasoning": "reason",
                "mark": mark,
                "signature": "signature",
                "encrypted_content": "opaque",
                "opaque_protocol": "example",
                "summary": ["first", "second"],
                "redacted": redacted,
                "item": identity(),
            },
        )
        for mark, redacted in (
            ("raw", False),
            ("summary", False),
            ("encrypted", False),
            ("encrypted", True),
        )
    ],
    (
        ToolCall,
        {
            "call_id": "call-3",
            "name": "add",
            "arguments": {"a": 19, "b": 23, "nested": [None, True, {"key": 0.5}]},
            "signature": "signature",
            "opaque_protocol": "example",
            "item": identity(),
        },
    ),
    (
        ToolResult,
        {
            "call_id": "call-3",
            "name": "add",
            "content": [
                TextContent("42", item=identity()),
                ImageContent(
                    ImageSource(type="base64", data="aW1hZ2U=", media_type="image/png"),
                    item=identity(),
                ),
            ],
            "item": identity(),
        },
    ),
]


@pytest.mark.parametrize("model,supplied", CONTENT_CASES)
def test_all_populated_content_fields_survive_roundtrip(model, supplied):
    assert_fields_exercised(model, supplied, implicit=("origin", "type"))
    original = model(**supplied, origin=Origin("$.original.content", object()))
    wire = original.to_wire()
    restored = content_from_wire(wire)
    assert_model_equal(restored, original)
    assert restored.origin.path == original.origin.path
    assert restored.origin.value is original.origin.value
    assert restored.to_wire() == wire
    assert wire["item"] == {"id": "item-7", "status": "completed"}
    assert not {"status", "item_id", "item_status"}.intersection(wire)
    if isinstance(original, ToolCall):
        assert wire["id"] == original.call_id
    else:
        assert "id" not in wire


@pytest.mark.parametrize(
    "model,supplied",
    [
        (TextContent, {"text": ""}),
        (ImageContent, {"source": ImageSource(type="url", url="https://example.com/image.png")}),
        (ReasoningContent, {}),
        (ToolCall, {"call_id": None, "name": "add"}),
        (ToolResult, {"call_id": None, "content": []}),
    ],
)
def test_minimal_content_defaults_survive_roundtrip(model, supplied):
    original = model(**supplied)
    assert_model_equal(content_from_wire(original.to_wire()), original)


def test_item_identity_has_only_independent_item_id_and_status():
    supplied = {"id": "item-7", "status": "completed"}
    assert_fields_exercised(ItemIdentity, supplied)
    original = TextContent("answer", item=identity())
    assert content_from_wire(original.to_wire()).item == identity()
    assert (
        content_from_wire(TextContent("answer", item=ItemIdentity()).to_wire()).item
        == ItemIdentity()
    )


@pytest.mark.parametrize("kind", ["url", "base64"])
def test_all_image_source_fields_survive_roundtrip(kind):
    supplied = {
        "type": kind,
        "url": "https://example.com/image.png",
        "data": "aW1hZ2U=",
        "media_type": "image/png",
    }
    assert_fields_exercised(ImageSource, supplied)
    original = ImageContent(ImageSource(**supplied))
    assert content_from_wire(original.to_wire()).source == original.source


@pytest.mark.parametrize(
    "model,role,content",
    [
        (Message, "user", [TextContent("question")]),
        (UserMessage, "user", [TextContent("question")]),
        (AssistantMessage, "assistant", [TextContent("answer", item=identity())]),
        (InstructionMessage, "system", [TextContent("instruction")]),
        (InstructionMessage, "developer", [TextContent("instruction")]),
        (ToolMessage, "user", [ToolResult("call-3", [TextContent("42")], name="add")]),
    ],
)
def test_all_message_fields_survive_roundtrip(model, role, content):
    supplied = {
        "role": role,
        "content": content,
        "name": "named-message",
        "protocol": "example",
        "item": identity(),
        "coalesce": True,
    }
    assert_fields_exercised(model, supplied, implicit=("origin",))
    original = model(**supplied, origin=Origin("$.original.message", object()))
    restored = messages_from_wire([original.to_wire()])[0]
    # Wire roles select their public specialization, including base Message input.
    assert model_value(restored) == model_value(original)
    assert restored.origin.path == original.origin.path
    assert restored.origin.value is original.origin.value
    assert restored.to_wire() == original.to_wire()


@pytest.mark.parametrize("strict", [False, True, None])
def test_all_function_tool_fields_survive_roundtrip(strict):
    supplied = {
        "name": "add",
        "description": "Add numbers",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}}},
        "strict": strict,
    }
    assert_fields_exercised(FunctionTool, supplied, implicit=("origin",))
    original = FunctionTool(**supplied, origin=Origin("$.original.tool", object()))
    restored = tools_from_wire([original.to_wire()])[0]
    assert_model_equal(restored, original)
    assert restored.origin.path == original.origin.path
    assert restored.origin.value is original.origin.value
    assert restored.to_wire() == original.to_wire()


def test_all_usage_fields_survive_roundtrip():
    supplied = {"input": 10, "output": 20, "total": 30, "cached": 4, "reasoning": 5}
    assert_fields_exercised(Usage, supplied)
    original = Usage(**supplied)
    assert Usage.from_record(original.to_record()) == original
    assert Usage.from_record(Usage().to_record()) == Usage()


OPERATIONS = [
    {"type": "start"},
    {
        "type": "metadata",
        "path": ("extension", "session"),
        "values": {"provider_field": {"nested": [False, None]}, "created_at": 123},
    },
    {"type": "usage", "usage": {"input": {"base": 4}, "output": {"base": 2}}},
    {
        "type": "snapshot",
        "blocks": [{"ref": ("text", 0, 0), "block": {"type": "text", "text": "done"}}],
    },
    {"type": "reason", "reason": "stop"},
    {"type": "finish", "reason": "tool_calls", "allow_tail": ["usage"]},
    {"type": "block_set", "ref": ("text", 0, 0), "block": {"type": "text", "text": ""}},
    {"type": "block_append", "block": {"type": "text", "text": "extra"}},
    {
        "type": "block_update",
        "ref": ("tool_use", 1, 0),
        "block": {"type": "tool_use", "id": "call-3"},
    },
    {
        "type": "block_delta",
        "ref": ("thinking", 2, 0),
        "field": "signature",
        "kind": "thinking",
        "value": "signature-fragment",
    },
    {"type": "block_delta", "ref": ("text", 0, 0), "field": "text", "value": None},
    {"type": "block_stop", "ref": ("text", 0, 0)},
]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_internal_stream_operations_are_preserved_without_dto_reconstruction(operation):
    wire = {"type": "batch", "protocol": "example", "events": [operation]}
    restored = stream_from_wire([wire])[0]
    assert restored.protocol == "example"
    assert restored.events == [operation]
    assert restored.to_wire() == wire


def test_partial_tool_updates_do_not_gain_content_defaults():
    operation = OPERATIONS[8]
    restored = stream_from_wire([{"protocol": "example", "events": [operation]}])[0]
    assert restored.to_wire()["events"][0]["block"] == {"type": "tool_use", "id": "call-3"}


def test_unknown_internal_operations_fail_instead_of_disappearing():
    with pytest.raises(MessageValidationError, match="operation"):
        stream_from_wire([{"protocol": "example", "events": [{"type": "unrecognized"}]}])


def test_unknown_internal_delta_fields_fail_instead_of_disappearing():
    with pytest.raises(MessageValidationError):
        stream_from_wire(
            [
                {
                    "protocol": "example",
                    "events": [
                        {
                            "type": "block_delta",
                            "ref": ("text", 0, 0),
                            "field": "unrecognized",
                            "kind": "text",
                            "value": "value",
                        }
                    ],
                }
            ]
        )


def test_unknown_canonical_content_fails_instead_of_disappearing():
    with pytest.raises(MessageValidationError):
        content_from_wire({"type": "unrecognized"})


@pytest.mark.parametrize("child", [ReasoningContent("private"), ToolCall("a", "f")])
def test_tool_result_rejects_non_result_content_instead_of_dropping_it(child):
    with pytest.raises(MessageValidationError, match="text or image"):
        content_from_wire({"type": "tool_result", "tool_use_id": "a", "content": [child.to_wire()]})


def test_generation_rejects_non_boolean_reasoning_redaction():
    with pytest.raises(MessageValidationError, match="content"):
        generation_input([AssistantMessage("assistant", [ReasoningContent(redacted="no")])], None)


EVENTS = [
    {"type": "start"},
    {
        "type": "block_start",
        "index": 0,
        "source_ref": ("text", 0, 0),
        "block": TextContent("", item=identity()),
    },
    {
        "type": "delta",
        "index": 0,
        "source_ref": ("thinking", 0, 0),
        "field": "thinking",
        "value": "reason",
    },
    {"type": "block_stop", "index": 0, "block": TextContent("answer")},
    {
        "type": "block",
        "index": 1,
        "block": ReasoningContent(mark="encrypted", encrypted_content="opaque", redacted=True),
    },
    {
        "type": "tool",
        "index": 2,
        "source_ref": ("tool_use", 1, 0),
        "tool_index": 0,
        "block": ToolCall("call-3", "add", {"a": 19, "b": 23}, item=identity()),
    },
    {
        "type": "finish",
        "reason": "tool_calls",
        "messages": [AssistantMessage("assistant", [TextContent("answer")], item=identity())],
    },
]


@pytest.mark.parametrize("supplied", EVENTS)
def test_public_message_events_survive_roundtrip(supplied):
    original = MessageEvent(**supplied)
    assert_model_equal(MessageEvent.from_wire(original.to_wire()), original)


def test_public_event_fixtures_exercise_every_field():
    assert_fields_exercised(MessageEvent, {key for supplied in EVENTS for key in supplied})


def test_complete_middleware_envelope_survives_roundtrip():
    supplied = {
        "messages": [UserMessage("user", [TextContent("question")], item=identity())],
        "tools": [FunctionTool("add", parameters={"type": "object"}, strict=False)],
        "stream": [StreamBatch("example", OPERATIONS)],
        "usage": Usage(input=4, output=2, total=6, cached=0, reasoning=1),
    }
    assert_fields_exercised(ProtocolMiddleWare, supplied)
    original = ProtocolMiddleWare(**supplied)
    wire = original.to_wire()
    restored = ProtocolMiddleWare(
        messages=messages_from_wire(wire["messages"]),
        tools=tools_from_wire(wire["tools"]),
        stream=stream_from_wire(wire["chunks"]),
        usage=Usage.from_record(wire["usage"]),
    )
    assert restored.to_wire() == wire
    assert dict(restored) == wire


def test_internal_stream_dto_is_not_a_public_model():
    assert "AssistantMessageChunk" not in my_llm.__all__
    assert not hasattr(my_llm, "AssistantMessageChunk")
