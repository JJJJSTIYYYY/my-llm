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
    {"type": "block_delta", "ref": ("text", 0, 0), "kind": "text", "field": "text", "value": None},
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


@pytest.mark.parametrize(
    "wire,field",
    [
        ({}, "type"),
        ({"type": "text"}, "text"),
        ({"type": "thinking"}, "thinking"),
        ({"type": "thinking", "thinking": ""}, "mark"),
        ({"type": "image"}, "source"),
        ({"type": "image", "source": {}}, "source.type"),
        ({"type": "tool_use"}, "id"),
        ({"type": "tool_use", "id": "call"}, "name"),
        ({"type": "tool_use", "id": "call", "name": "f"}, "input"),
        ({"type": "tool_result"}, "tool_use_id"),
        ({"type": "tool_result", "tool_use_id": "call"}, "content"),
    ],
)
def test_content_reader_reports_missing_canonical_fields(wire, field):
    with pytest.raises(MessageValidationError) as captured:
        content_from_wire(wire, "$.block")
    assert f"$.block.{field}:" in str(captured.value)


@pytest.mark.parametrize(
    "reader,wire,path",
    [
        (content_from_wire, None, "$"),
        (content_from_wire, {"type": "text", "text": "", "item": []}, "$.item"),
        (messages_from_wire, [{}], "$.messages[0].role"),
        (messages_from_wire, [{"role": "user"}], "$.messages[0].content"),
        (messages_from_wire, [{"role": "user", "content": None}], "$.messages[0].content"),
        (tools_from_wire, [{}], "$.tools[0].functions"),
        (tools_from_wire, [{"functions": [{}]}], "$.tools[0].functions[0].name"),
        (stream_from_wire, [{"protocol": "example"}], "$.chunks[0].events"),
        (stream_from_wire, [{"protocol": "example", "events": [None]}], "$.chunks[0].events[0]"),
        (MessageEvent.from_wire, {"type": "delta"}, "$.chunks[0].field"),
        (
            MessageEvent.from_wire,
            {"type": "block", "block": {"type": "thinking", "thinking": ""}},
            "$.chunks[0].block.mark",
        ),
    ],
)
def test_public_wire_readers_report_structural_errors(reader, wire, path):
    with pytest.raises(MessageValidationError) as captured:
        reader(wire)
    assert path in str(captured.value)


@pytest.mark.parametrize("target", [*my_llm.MyCodec().codecs, "middleware"])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
@pytest.mark.parametrize(
    "operation",
    [
        {"type": "unknown"},
        {
            "type": "block_delta",
            "ref": ("text", 0, 0),
            "kind": "text",
            "field": "unknown",
            "value": "x",
        },
        {
            "type": "block_set",
            "ref": ("thinking", 0, 0),
            "block": {"type": "thinking", "thinking": ""},
        },
    ],
)
def test_decode_validates_raw_and_typed_batches_before_state_dispatch(
    target, typed, policy, operation
):
    batch = StreamBatch("openai_chat", [operation, {"type": "finish"}])

    def hook(*args):
        pytest.fail("malformed middleware must not invoke semantic-loss hooks")

    codec = my_llm.MyCodec(policy, input_semantic_loss_hook=hook, output_semantic_loss_hook=hook)
    with pytest.raises(MessageValidationError):
        codec.decode(
            chunks=[batch if typed else batch.to_wire()],
            target_format=target,
            input_format="openai_chat",
        )


def test_encode_and_decode_share_operation_validation(monkeypatch):
    codec = my_llm.MyCodec()
    wire = StreamBatch("openai_chat", [{"type": "unknown"}]).to_wire()
    monkeypatch.setattr(
        type(codec.codecs["openai_chat"].encoder), "_chunks", lambda *a, **kw: [wire]
    )
    with pytest.raises(MessageValidationError) as encoded:
        codec.encode(chunks=[], input_format="openai_chat")
    with pytest.raises(MessageValidationError) as decoded:
        codec.decode(chunks=[wire], target_format="openai_chat")
    assert str(encoded.value) == str(decoded.value)


def test_decode_revalidates_batches_mutated_after_encoding():
    codec = my_llm.MyCodec()
    encoded = codec.encode(
        chunks=[{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}],
        input_format="openai_chat",
    )
    encoded.stream[0].events.append({"type": "unknown"})
    with pytest.raises(MessageValidationError, match="operation"):
        codec.decode(chunks=encoded.stream, target_format="openai_chat")


def test_decode_rejects_mixed_batch_and_public_event_streams():
    with pytest.raises(MessageValidationError, match="cannot mix"):
        my_llm.MyCodec().decode(
            chunks=[StreamBatch("openai_chat", []), MessageEvent("start")],
            target_format="openai_chat",
        )


@pytest.mark.parametrize("target", ["openai_chat", "middleware"])
def test_decode_checks_required_content_before_rendering(target):
    with pytest.raises(MessageValidationError, match="mark"):
        my_llm.MyCodec().decode(
            messages=[{"role": "assistant", "content": [{"type": "thinking", "thinking": ""}]}],
            target_format=target,
            input_format="openai_chat",
        )


def test_wire_validation_does_not_wrap_application_copy_errors():
    class ApplicationValue:
        def __deepcopy__(self, memo):
            raise KeyError("application failure")

    with pytest.raises(KeyError, match="application failure"):
        content_from_wire(
            {"type": "tool_use", "id": "call", "name": "f", "input": {"x": ApplicationValue()}}
        )


@pytest.mark.parametrize(
    "operation",
    [
        {"type": "block_update", "ref": ("text", 0, 0), "block": {"type": "unknown"}},
        {
            "type": "block_delta",
            "ref": ("text", 0, 0),
            "kind": "unknown",
            "field": "text",
            "value": "x",
        },
    ],
)
def test_partial_operations_still_require_a_known_content_kind(operation):
    wire = StreamBatch("openai_chat", [operation]).to_wire()
    for read in (
        lambda: stream_from_wire([wire]),
        lambda: my_llm.MyCodec().decode(chunks=[wire], target_format="openai_chat"),
    ):
        with pytest.raises(MessageValidationError, match="content type"):
            read()


@pytest.mark.parametrize(
    "name,value",
    [
        ("validate_content", {"type": "thinking", "thinking": "reason", "mark": "raw"}),
        (
            "validate_messages",
            [{"role": "user", "content": [{"type": "text", "text": "question"}]}],
        ),
        ("validate_tools", [{"functions": [{"name": "f"}]}]),
        ("validate_operations", [{"protocol": "openai_chat", "events": OPERATIONS}]),
        (
            "validate_events",
            [{"type": "finish", "reason": "stop", "messages": [], "output_blocks": []}],
        ),
    ],
)
def test_public_validators_do_not_format_success_paths(name, value):
    import my_llm.middleware as middleware

    class FailureOnlyPath(str):
        def __format__(self, spec):
            pytest.fail("valid wire data must not format an error path")

    assert getattr(middleware, name)(value, path=FailureOnlyPath("$.custom")) is None


@pytest.mark.parametrize(
    "name,value,path",
    [
        (
            "validate_content",
            {
                "type": "tool_result",
                "tool_use_id": "call",
                "content": [{"type": "image", "source": {}}],
            },
            "$.custom.content[0].source.type",
        ),
        (
            "validate_messages",
            [{"role": "assistant", "content": [{"type": "thinking", "thinking": ""}]}],
            "$.custom[0].content[0].mark",
        ),
        ("validate_tools", [{"functions": [{"name": "f"}, {}]}], "$.custom[0].functions[1].name"),
        (
            "validate_operations",
            [{"protocol": "example", "events": [{"type": "metadata", "path": None, "values": {}}]}],
            "$.custom[0].events[0].path",
        ),
        (
            "validate_operations",
            [
                {
                    "protocol": "example",
                    "events": [
                        {
                            "type": "snapshot",
                            "blocks": [{"ref": ("text", 0, 0), "block": {"type": "text"}}],
                        }
                    ],
                }
            ],
            "$.custom[0].events[0].blocks[0].block.text",
        ),
        (
            "validate_events",
            [{"type": "finish", "messages": [{"role": "user", "content": [{"type": "text"}]}]}],
            "$.custom[0].messages[0].content[0].text",
        ),
        (
            "validate_events",
            [
                {
                    "type": "finish",
                    "output_blocks": [{"index": 0, "block": {"type": "thinking", "thinking": ""}}],
                }
            ],
            "$.custom[0].output_blocks[0].block.mark",
        ),
    ],
)
def test_public_validators_preserve_nested_error_paths(name, value, path):
    import my_llm.middleware as middleware

    with pytest.raises(MessageValidationError) as captured:
        getattr(middleware, name)(value, path="$.custom")
    assert str(captured.value).startswith(f"{path}:")


@pytest.mark.parametrize(
    "finish",
    [
        {"type": "finish"},
        {"type": "finish", "reason": "stop"},
        {"type": "finish", "messages": []},
        {"type": "finish", "output_blocks": []},
    ],
)
def test_finish_structures_are_optional(finish):
    from my_llm.middleware import validate_events

    validate_events([finish])


def test_event_structured_fields_must_match_the_event_kind():
    from my_llm.middleware import validate_events

    with pytest.raises(MessageValidationError, match=r"chunks\[0\].messages"):
        validate_events([{"type": "delta", "field": "text", "value": "x", "messages": []}])
