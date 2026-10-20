"""Only recognized protocol fields cross translation boundaries.

Open application dictionaries are deliberately separate from protocol structures.
Fixtures are independent native values; dirty and clean translations must agree.
"""

import json
from copy import deepcopy

import pytest

from my_llm import MyCodec, UnsupportedCapabilityError
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools

UNKNOWN = "__vendor_unrecognized__"
APPLICATION = {"vendor_payload": [{"role": "business", "type": "custom", "future": [1, None]}]}
OPEN_FIELDS = {
    "arguments",
    "args",
    "input",
    "parameters",
    "input_schema",
    "parameters_json_schema",
    "metadata",
}


def add_unknown_fields(value):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            is_application = key in OPEN_FIELDS or (
                key == "response" and isinstance(child, dict) and child.get("object") != "response"
            )
            result[key] = deepcopy(child) if is_application else add_unknown_fields(child)
        result[UNKNOWN] = {"nested": [1, None, {"future": True}]}
        return result
    if isinstance(value, list):
        return [add_unknown_fields(child) for child in value]
    return value


def assert_no_unknown(value):
    if isinstance(value, dict):
        assert UNKNOWN not in value
        assert not any(key.startswith("__my_llm_codec_") for key in value)
        for child in value.values():
            assert_no_unknown(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_unknown(child)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools", "chunks"])
def test_encoder_ignores_unknown_protocol_fields_without_mutating_input(protocol, kind):
    codec = MyCodec()
    clean = {"messages": history, "tools": tools, "chunks": stream_events}[kind](protocol)
    incoming = add_unknown_fields(clean)
    original = deepcopy(incoming)

    expected = codec.encode(input_format=protocol, **{kind: clean})[kind]
    actual = codec.encode(input_format=protocol, **{kind: incoming})[kind]

    assert actual == expected
    assert_no_unknown(actual)
    assert incoming == original
    assert actual


@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools", "chunks"])
def test_decoder_ignores_unknown_intermediate_fields_without_mutating_input(target, kind):
    codec = MyCodec()
    native = {"messages": history, "tools": tools, "chunks": stream_events}[kind]("openai_chat")
    clean = codec.encode(input_format="openai_chat", **{kind: native})[kind]
    incoming = add_unknown_fields(clean)
    original = deepcopy(incoming)

    expected = codec.decode(target_format=target, **{kind: clean})[kind]
    actual = codec.decode(target_format=target, **{kind: incoming})[kind]

    assert actual == expected
    assert_no_unknown(actual)
    assert incoming == original


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools", "chunks"])
def test_unknown_fields_do_not_leak_through_any_protocol_pair(source, target, kind, monkeypatch):
    class FixedUUID:
        hex = "deterministic"

    monkeypatch.setattr("my_llm.stream._state.uuid4", FixedUUID)
    codec = MyCodec()
    clean = {"messages": history, "tools": tools, "chunks": stream_events}[kind](source)
    incoming = add_unknown_fields(clean)
    original = deepcopy(incoming)

    actual = codec.convert(input_format=source, target_format=target, **{kind: incoming})[kind]
    expected = codec.convert(input_format=source, target_format=target, **{kind: clean})[kind]

    assert actual == expected
    assert_no_unknown(actual)
    assert incoming == original


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_open_tool_arguments_and_json_schema_keep_arbitrary_nested_keys(source, target):
    codec = MyCodec()
    call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": json.dumps(APPLICATION)},
            }
        ],
    }
    declaration = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "vendor_schema_keyword": APPLICATION},
            },
        }
    ]
    native_messages = codec.convert([call], input_format="openai_chat", target_format=source)[
        "messages"
    ]
    native_tools = codec.convert(
        tools=declaration, input_format="openai_chat", target_format=source
    )["tools"]
    original = deepcopy((native_messages, native_tools))

    converted = codec.convert(
        native_messages, native_tools, input_format=source, target_format=target
    )
    normalized = codec.encode(**converted, input_format=target)
    content = [block for message in normalized["messages"] for block in message["content"]]
    assert next(block for block in content if block["type"] == "tool_use")["input"] == APPLICATION
    function = normalized["tools"][0]["functions"][0]
    assert function["parameters"]["vendor_schema_keyword"] == APPLICATION
    assert (native_messages, native_tools) == original


@pytest.mark.parametrize("target", PROTOCOLS)
def test_unknown_intermediate_blocks_are_rejected_instead_of_becoming_opaque_content(target):
    values = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "known"},
                {"type": "vendor_future", UNKNOWN: APPLICATION},
            ],
        }
    ]
    before = deepcopy(values)
    with pytest.raises(UnsupportedCapabilityError, match="content type"):
        MyCodec().decode(values, target_format=target)
    assert values == before


def test_translation_and_sdk_validation_have_separate_interfaces():
    for codec in MyCodec().codecs.values():
        for direction in (codec.encoder, codec.decoder):
            for name in (
                "validate_profile",
                "validate_parameters",
                "validate_request",
                "validate_input",
                "filter_messages",
                "filter_tools",
                "create_client",
                "upstream",
            ):
                assert not hasattr(direction, name)
        assert callable(codec.decoder.request_parameters)
        assert callable(codec.decoder.assemble_request)
        assert callable(codec.transport.request)
        assert callable(codec.transport.validate_profile)
        assert callable(codec.transport.validate_request)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_sdk_event_objects_are_translated_without_retaining_model_extra_fields(protocol):
    from pydantic import ConfigDict

    from tests.helpers.sdk import EVENT_TYPES

    source = stream_events(protocol)
    sdk_objects = []
    for value in source:
        models = EVENT_TYPES[protocol]
        model_type = models[value["type"]] if isinstance(models, dict) else models
        # Compatibility SDKs may enable extras even when their base SDK forbids
        # them. The public model interface remains the same.
        compatible_type = type(
            "CompatibleEvent", (model_type,), {"model_config": ConfigDict(extra="allow")}
        )
        sdk_objects.append(compatible_type.model_validate({**value, UNKNOWN: APPLICATION}))
    original = [value.model_dump(mode="json", exclude_unset=True) for value in sdk_objects]
    codec = MyCodec()

    actual = codec.encode(chunks=sdk_objects, input_format=protocol)["chunks"]
    clean = [{key: child for key, child in value.items() if key != UNKNOWN} for value in original]

    assert actual == codec.encode(chunks=clean, input_format=protocol)["chunks"]
    assert_no_unknown(actual)
    assert [value.model_dump(mode="json", exclude_unset=True) for value in sdk_objects] == original


def test_profile_paths_translate_configured_nonstandard_fields_without_sibling_leakage(
    profile_factory,
):
    profile = profile_factory(
        "openai_chat",
        content=parameter(
            "content", protocol="openai_chat", path="$.choices[0].delta.vendor_answer"
        ),
        reasoning_content=parameter(
            "reasoning_content", protocol="openai_chat", path="$.choices[0].delta.vendor_reasoning"
        ),
        replay_reasoning_content_when_tool_calls=config(
            "replay_reasoning_content_when_tool_calls", True
        ),
    )
    incoming = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "vendor_answer": "configured text",
                        "vendor_reasoning": "configured thought",
                        UNKNOWN: APPLICATION,
                    },
                    "finish_reason": "stop",
                }
            ],
            UNKNOWN: APPLICATION,
        }
    ]
    original = deepcopy(incoming)
    codec = MyCodec()
    encoded = codec.encode(chunks=incoming, input_format="openai_chat", profile=profile)
    decoded = codec.decode(**encoded, target_format="openai_chat", profile=profile)["chunks"]

    assert "configured text" in str(decoded)
    assert "configured thought" in str(decoded)
    assert_no_unknown(encoded)
    assert_no_unknown(decoded)
    assert incoming == original


def test_direct_responses_intermediate_annotations_are_filtered_in_every_event():
    from my_llm.codec.shared.stream import BlockRef, batch

    annotation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 2,
        "title": "cite",
        "url": "https://example.invalid",
        UNKNOWN: APPLICATION,
    }
    chunks = [
        batch(
            [
                {
                    "type": "block_set",
                    "ref": BlockRef("text"),
                    "block": {
                        "type": "text",
                        "text": "ok",
                        "annotations": [annotation],
                        UNKNOWN: APPLICATION,
                    },
                },
                {"type": "finish", "reason": "stop"},
            ],
            "openai_responses",
        )
    ]
    original = deepcopy(chunks)

    output = MyCodec().decode(chunks=chunks, target_format="openai_responses")["chunks"]

    assert_no_unknown(output)
    assert "cite" in str(output)
    assert chunks == original


def test_direct_anthropic_intermediate_citations_filter_unknown_fields():
    citation = {
        "type": "web_search_result_location",
        "url": "https://example.invalid",
        "title": "x",
        "cited_text": "text",
        "encrypted_index": "opaque",
        UNKNOWN: APPLICATION,
    }
    chunks = [{"type": "block_annotation", "index": 0, "field": "citations", "value": citation}]
    before = deepcopy(chunks)

    output = MyCodec().decode(
        chunks=chunks, target_format="anthropic", source_protocol="anthropic"
    )["chunks"]

    assert_no_unknown(output)
    assert output[0]["delta"]["citation"] == {
        key: value for key, value in citation.items() if key != UNKNOWN
    }
    assert chunks == before


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
def test_direct_intermediate_known_structures_filter_nested_unknown_fields(protocol):
    annotation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 4,
        "title": "cite",
        "url": "https://example.invalid",
        UNKNOWN: APPLICATION,
    }
    values = [
        {
            "protocol": protocol,
            "role": "assistant",
            "audio": {
                "id": "audio_1",
                "data": "YWJj",
                "transcript": "text",
                "expires_at": 10,
                UNKNOWN: APPLICATION,
            },
            "content": [
                {"type": "text", "text": "text", "annotations": [annotation], UNKNOWN: APPLICATION}
            ],
        }
    ]
    original = deepcopy(values)

    result = MyCodec().decode(values, target_format=protocol)["messages"]

    assert_no_unknown(result)
    assert "text" in str(result)
    assert values == original


@pytest.mark.parametrize("target", PROTOCOLS)
def test_open_tool_result_objects_preserve_application_keys_in_all_targets(target):
    incoming = [
        {"role": "model", "parts": [{"function_call": {"id": "c1", "name": "f", "args": {}}}]},
        {
            "role": "user",
            "parts": [{"function_response": {"id": "c1", "name": "f", "response": APPLICATION}}],
        },
    ]
    original = deepcopy(incoming)
    codec = MyCodec()

    converted = codec.convert(incoming, input_format="gemini", target_format=target)["messages"]
    intermediate = codec.encode(converted, input_format=target)["messages"]
    result = next(
        block
        for message in intermediate
        for block in message["content"]
        if block["type"] == "tool_result"
    )["content"]
    if isinstance(result, str):
        result = json.loads(result)
    assert result == APPLICATION
    assert incoming == original


@pytest.mark.parametrize("target", PROTOCOLS)
def test_open_application_dictionary_keys_are_not_treated_as_control_fields(target):
    payload = {UNKNOWN: {"type": "application", "__my_llm_codec_replay__": {"values": [1, 2]}}}
    values = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c1", "name": "f", "input": payload}],
        }
    ]
    original = deepcopy(values)
    codec = MyCodec()

    native = codec.decode(values, target_format=target)["messages"]
    normalized = codec.encode(native, input_format=target)["messages"]
    call = next(
        block
        for message in normalized
        for block in message["content"]
        if block["type"] == "tool_use"
    )

    assert call["input"] == payload
    assert values == original


@pytest.mark.parametrize("target", PROTOCOLS)
def test_chat_tool_result_text_blocks_are_protocol_structures_not_open_payloads(target):
    from tests.helpers.sdk import MESSAGE_TYPES, assert_native

    clean = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "ok"}]},
    ]
    for message in clean:
        assert_native(message, MESSAGE_TYPES["openai_chat"])
    incoming = deepcopy(clean)
    incoming[-1]["content"][0][UNKNOWN] = APPLICATION
    original = deepcopy(incoming)
    codec = MyCodec()

    encoded = codec.encode(incoming, input_format="openai_chat")
    result = codec.decode(**encoded, target_format=target)["messages"]

    assert_no_unknown(encoded)
    assert_no_unknown(result)
    assert "ok" in str(result)
    assert incoming == original


def test_gemini_part_metadata_is_open_but_part_siblings_are_protocol_fields():
    from google.genai import types

    from tests.helpers.sdk import assert_native

    payload = {UNKNOWN: {"source": {"nested": [1, None]}, "type": "application"}}
    clean = [{"role": "user", "parts": [{"text": "known", "part_metadata": payload}]}]
    assert_native(clean[0], types.Content)
    incoming = deepcopy(clean)
    incoming[0][UNKNOWN] = {"root": True}
    incoming[0]["parts"][0][UNKNOWN] = {"part": True}
    original = deepcopy(incoming)
    codec = MyCodec()

    encoded = codec.encode(incoming, input_format="gemini")
    result = codec.decode(**encoded, target_format="gemini")["messages"]

    assert encoded["messages"][0]["content"][0]["part_metadata"] == payload
    assert UNKNOWN not in encoded["messages"][0]
    assert UNKNOWN not in encoded["messages"][0]["content"][0]
    assert result == clean
    assert_native(result[0], types.Content)
    assert incoming == original
    result[0]["parts"][0]["part_metadata"][UNKNOWN]["source"]["nested"].append(2)
    assert incoming == original


@pytest.mark.parametrize("behavior", ["BLOCKING", "NON_BLOCKING"])
def test_gemini_function_response_schema_is_open_and_behavior_is_explicit(behavior):
    from google.genai import types

    from tests.helpers.sdk import assert_native

    schema = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
        "vendor_schema_keyword": {UNKNOWN: {"nested": [1, None]}},
    }
    clean = [
        {
            "function_declarations": [
                {
                    "name": "f",
                    "description": "Function",
                    "parameters_json_schema": {"type": "object"},
                    "response_json_schema": schema,
                    "behavior": behavior,
                }
            ]
        }
    ]
    assert_native(clean[0], types.Tool)
    incoming = deepcopy(clean)
    incoming[0][UNKNOWN] = {"tool": True}
    incoming[0]["function_declarations"][0][UNKNOWN] = {"function": True}
    original = deepcopy(incoming)
    codec = MyCodec()

    encoded = codec.encode(tools=incoming, input_format="gemini")
    result = codec.decode(**encoded, target_format="gemini")["tools"]

    function = encoded["tools"][0]["functions"][0]
    assert function["response_json_schema"] == schema
    assert function["behavior"] == behavior
    assert UNKNOWN not in encoded["tools"][0]
    assert UNKNOWN not in function
    assert result == clean
    assert_native(result[0], types.Tool)
    assert incoming == original
    result[0]["function_declarations"][0]["response_json_schema"]["vendor_schema_keyword"][UNKNOWN][
        "nested"
    ].append(2)
    assert incoming == original


@pytest.mark.parametrize("target", PROTOCOLS)
def test_unknown_intermediate_delta_fields_are_ignored_without_becoming_text(target):
    incoming = [
        {"type": "delta", "field": "vendor_future", "value": "unrecognized content", "index": 0}
    ]
    original = deepcopy(incoming)
    codec = MyCodec()

    assert codec.decode(chunks=incoming, target_format=target)["chunks"] == []

    known = {"type": "delta", "field": "text", "value": "known", "index": 0}
    actual = codec.decode(chunks=[*incoming, known], target_format=target)["chunks"]
    assert actual == codec.decode(chunks=[known], target_format=target)["chunks"]
    assert "vendor_future" not in str(actual)
    assert "unrecognized content" not in str(actual)
    assert incoming == original
