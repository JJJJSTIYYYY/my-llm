"""Mutable public conversion results own their nested protocol payloads."""

from copy import deepcopy

import pytest
from google.genai import types
from pydantic import BaseModel

from my_llm import MyCodec
from my_llm.codec.shared.options import resolve_options
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools


def edit_containers(value):
    """Edit mappings and sequences, including payloads inside SDK request models."""
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            edit_containers(getattr(value, name))
    elif isinstance(value, dict):
        for child in tuple(value.values()):
            edit_containers(child)
        value["x_test_edit"] = ["changed"]
    elif isinstance(value, list):
        for child in tuple(value):
            edit_containers(child)
        value.append({"x_test_edit": "changed"})


def native_values(protocol, kind):
    return history(protocol) if kind == "messages" else tools(protocol)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_encoder_owns_nested_output(protocol, kind):
    codec = MyCodec().codecs[protocol]
    values = native_values(protocol, kind)
    before = deepcopy(values)

    converted = getattr(codec, f"encode_{kind}")(values)
    edit_containers(converted)

    assert values == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_encoder_captures_nested_input(protocol, kind):
    codec = MyCodec().codecs[protocol]
    values = native_values(protocol, kind)
    converted = getattr(codec, f"encode_{kind}")(values)
    before = deepcopy(converted)

    edit_containers(values)

    assert converted == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_decoder_owns_nested_output(source, target, kind):
    codec = MyCodec()
    intermediate = codec.encode(**{kind: native_values(source, kind)}, input_format=source)[kind]
    before = deepcopy(intermediate)

    converted = getattr(codec.codecs[target], f"decode_{kind}")(intermediate)
    edit_containers(converted)

    assert intermediate == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_decoder_captures_nested_input(source, target, kind):
    codec = MyCodec()
    intermediate = codec.encode(**{kind: native_values(source, kind)}, input_format=source)[kind]
    converted = getattr(codec.codecs[target], f"decode_{kind}")(intermediate)
    before = deepcopy(converted)

    edit_containers(intermediate)

    assert converted == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_repeated_public_requests_do_not_share_mutable_input(profile_factory, protocol):
    codec = MyCodec()
    intermediate = codec.encode(history("gemini"), tools("gemini"), input_format="gemini")
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        supports_function_calling=config(True),
    )
    before = deepcopy(intermediate)
    first = codec.request(intermediate, profile)
    expected = deepcopy(first)

    edit_containers(first)

    assert intermediate == before
    assert codec.request(intermediate, profile) == expected


def test_expanded_binding_owns_native_sdk_option_model(profile_factory):
    profile = profile_factory(
        "gemini",
        extra_body=config({"thinking_config": types.ThinkingConfig(include_thoughts=True)}),
    )
    before = deepcopy(profile.values)
    options = resolve_options(profile, reasoning=True)

    options.body["thinking_config"].include_thoughts = False

    assert profile.values == before


def test_expanded_binding_owns_mutable_values_inside_tuple(profile_factory):
    profile = profile_factory(
        "gemini",
        extra_body=config(
            {
                "safety_settings": (
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                    ),
                )
            },
        ),
    )
    before = deepcopy(profile.values)
    options = resolve_options(profile, reasoning=True)

    options.body["safety_settings"][0].threshold = "BLOCK_ONLY_HIGH"

    assert profile.values == before


class TrackedJSON(dict):
    """A valid JSON object that records ownership captures of its whole tree."""

    def __init__(self, value, *, copies=None):
        super().__init__(value)
        self.copies = [] if copies is None else copies

    def __deepcopy__(self, memo):
        self.copies.append(1)
        result = type(self)({}, copies=self.copies)
        memo[id(self)] = result
        result.update({key: deepcopy(value, memo) for key, value in self.items()})
        return result


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_encoder_captures_open_tool_schema_once(protocol):
    schema = TrackedJSON({"type": "object", "properties": {"value": {"type": "string"}}})
    values = tools(protocol)
    if protocol in ("openai_chat", "ollama"):
        values[0]["function"]["parameters"] = schema
    elif protocol == "gemini":
        values[0]["function_declarations"][0]["parameters_json_schema"] = schema
    else:
        values[0]["input_schema" if protocol == "anthropic" else "parameters"] = schema

    converted = MyCodec().codecs[protocol].encoder.tools(values)

    assert len(schema.copies) == 1
    edit_containers(converted)
    assert schema == {"type": "object", "properties": {"value": {"type": "string"}}}


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_decoder_captures_open_tool_schema_once(protocol):
    schema = TrackedJSON({"type": "object", "properties": {"value": {"type": "string"}}})
    intermediate = [{"functions": [{"name": "lookup", "parameters": schema}]}]

    converted = MyCodec().decoder(protocol).tools(intermediate)

    assert len(schema.copies) == 1
    edit_containers(converted)
    assert schema == {"type": "object", "properties": {"value": {"type": "string"}}}


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_encoder_captures_native_argument_object_once(protocol):
    payload = TrackedJSON({"nested": ["original"]})
    values = history(protocol)
    if protocol == "anthropic":
        values[1]["content"][1]["input"] = payload
    elif protocol == "gemini":
        values[1]["parts"][1]["function_call"]["args"] = payload
    else:
        values[1]["tool_calls"][0]["function"]["arguments"] = payload

    converted = MyCodec().codecs[protocol].encoder.messages(values)

    assert len(payload.copies) == 1
    edit_containers(converted)
    assert payload == {"nested": ["original"]}


def test_responses_decoder_captures_native_response_metadata_once():
    payload = TrackedJSON({"label": "original"})

    converted = (
        MyCodec()
        .decoder("openai_responses")
        .chunks(
            [{"type": "start"}],
            source_protocol="openai_responses",
            response_parameters={"metadata": payload},
        )
    )

    assert len(payload.copies) == 1
    edit_containers(converted)
    assert payload == {"label": "original"}


class UnreadableExtension:
    """An unconfigured extension must never be copied or inspected."""

    def __deepcopy__(self, memo):
        raise AssertionError("read an unconfigured protocol field")


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
def test_text_projection_does_not_copy_unrecognized_annotation_fields(protocol):
    citation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 5,
        "title": "Source",
        "url": "https://example.com",
        "vendor_unconfigured": UnreadableExtension(),
    }
    messages = [
        {
            "protocol": protocol,
            "role": "assistant",
            "content": [{"type": "text", "text": "hello", "annotations": [citation]}],
        }
    ]

    converted = MyCodec().decoder(protocol).messages(messages)

    annotation = (
        converted[0]["annotations"][0]["url_citation"]
        if protocol == "openai_chat"
        else converted[0]["content"][0]["annotations"][0]
    )
    assert annotation["url"] == "https://example.com"
    assert "vendor_unconfigured" not in annotation


class AddressedFieldsOnly(dict):
    """Protocol selectors must address known names instead of scanning input keys."""

    def __iter__(self):
        raise AssertionError("enumerated protocol fields")

    def items(self):
        raise AssertionError("enumerated protocol values")

    def values(self):
        raise AssertionError("enumerated protocol values")

    def keys(self):
        raise AssertionError("enumerated protocol keys")


@pytest.mark.parametrize(
    "cache_control", [None, {"type": "ephemeral"}, {"type": "ephemeral", "ttl": "1h"}]
)
def test_anthropic_tool_cache_control_survives_request_translation(profile_factory, cache_control):
    codec = MyCodec()
    source = [
        {
            "name": "lookup",
            "input_schema": {"type": "object"},
            "cache_control": deepcopy(cache_control),
        }
    ]
    intermediate = codec.encode(
        messages=[{"role": "user", "content": "hello"}], tools=source, input_format="anthropic"
    )

    request = codec.request(
        intermediate,
        profile_factory(
            "anthropic",
            max_tokens=parameter("max_tokens", 256, protocol="anthropic"),
            supports_function_calling=config(True),
        ),
    )

    assert request["tools"] == source
    if cache_control is not None:
        source[0]["cache_control"]["ttl"] = "5m"
        assert intermediate["tools"][0]["functions"][0]["cache_control"] == cache_control
        request["tools"][0]["cache_control"]["ttl"] = "5m"
        assert intermediate["tools"][0]["functions"][0]["cache_control"] == cache_control


@pytest.mark.parametrize("direction", ["encoder", "decoder"])
def test_anthropic_tool_cache_control_only_addresses_known_fields(direction):
    cache_control = AddressedFieldsOnly(
        type="ephemeral", ttl="1h", vendor_unconfigured=UnreadableExtension()
    )
    function = AddressedFieldsOnly(
        name="lookup", cache_control=cache_control, vendor_unconfigured=UnreadableExtension()
    )
    component = getattr(MyCodec().codecs["anthropic"], direction)
    if direction == "encoder":
        function["input_schema"] = {"type": "object"}
        result = component.tools([function])[0]["functions"][0]
    else:
        function["parameters"] = {"type": "object"}
        result = component.tools([{"functions": [function]}])[0]

    assert result["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "vendor_unconfigured" not in result


def test_gemini_schema_projection_only_addresses_recognized_keywords():
    schema = AddressedFieldsOnly(
        type="OBJECT",
        properties={
            "name": AddressedFieldsOnly(type="STRING", vendor_unconfigured=UnreadableExtension())
        },
        vendor_unconfigured=UnreadableExtension(),
    )
    values = [{"function_declarations": [{"name": "lookup", "parameters": schema}]}]

    converted = MyCodec().codecs["gemini"].encoder.tools(values)

    assert converted == [
        {
            "functions": [
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            ]
        }
    ]


def guarded_protocol(value, *, parent=None):
    # These recognized open dictionaries belong to the caller's application.
    if parent in {
        "parameters",
        "parameters_json_schema",
        "input_schema",
        "args",
        "arguments",
        "input",
        "part_metadata",
        "metadata",
    }:
        return value
    if isinstance(value, dict) and (
        parent == "content" or parent == "response" and value.get("object") != "response"
    ):
        return value
    if isinstance(value, dict):
        return AddressedFieldsOnly(
            {
                **{key: guarded_protocol(item, parent=key) for key, item in value.items()},
                "vendor_unconfigured": UnreadableExtension(),
            }
        )
    if isinstance(value, list):
        return [guarded_protocol(item) for item in value]
    return value


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools", "chunks"])
def test_encoder_only_addresses_recognized_fields(protocol, kind):
    from tests.helpers.protocols import stream_events

    source = stream_events(protocol) if kind == "chunks" else native_values(protocol, kind)
    encoder = MyCodec().codecs[protocol].encoder
    expected = getattr(encoder, kind)(source)

    assert getattr(encoder, kind)(guarded_protocol(source)) == expected


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_encoder_does_not_serialize_sdk_models(protocol):
    from pydantic import ConfigDict

    from tests.helpers.protocols import stream_events
    from tests.helpers.sdk import EVENT_TYPES

    source = stream_events(protocol)
    objects = []
    for event in source:
        schemas = EVENT_TYPES[protocol]
        schema = schemas[event["type"]] if isinstance(schemas, dict) else schemas

        class CompatibleEvent(schema):
            model_config = ConfigDict(extra="allow")

            def model_dump(self, *args, **kwargs):
                raise AssertionError("serialized a complete SDK event")

        objects.append(
            CompatibleEvent.model_validate(
                {
                    **event,
                    "vendor_unconfigured": UnreadableExtension(),
                }
            )
        )
    encoder = MyCodec().codecs[protocol].encoder

    assert encoder.chunks(objects) == encoder.chunks(source)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_encoder_profile_routes_are_sdk_root_paths(protocol, profile_factory):
    text_event = {
        "openai_chat": {"choices": [{"index": 0, "delta": {"content": "native"}}]},
        "openai_responses": {"type": "response.output_text.delta", "delta": "native"},
        "anthropic": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "native"},
        },
        "gemini": {
            "candidates": [{"content": {"parts": [{"text": "native"}]}}],
        },
        "ollama": {"message": {"role": "assistant", "content": "native"}},
    }[protocol]
    event = {**text_event, "vendor": {"answer": "configured"}}
    profile = profile_factory(
        protocol, content=parameter("content", protocol=protocol, path="$.vendor.answer")
    )

    batches = MyCodec().codecs[protocol].encoder.chunks([event], profile)
    operations = [op for batch in batches for op in batch["events"]]
    text = [
        op["value"] if op["type"] == "block_delta" else op["block"]["text"]
        for op in operations
        if op["type"] == "block_delta"
        and op.get("kind") == "text"
        or op["type"] == "block_append"
        and op["block"]["type"] == "text"
    ]
    assert text == ["configured"]


def test_gemini_standard_path_keeps_native_multi_part_thought_semantics(profile_factory):
    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "thought 1", "thought": True},
                        {"text": "answer 1"},
                        {"text": "thought 2", "thought": True},
                        {"text": "answer 2"},
                    ]
                }
            }
        ]
    }
    encoder = MyCodec().codecs["gemini"].encoder

    assert encoder.chunks([event], profile_factory("gemini")) == encoder.chunks([event])


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_decoder_only_addresses_recognized_fields(source, target, kind):
    codec = MyCodec()
    native = native_values(source, kind)
    intermediate = codec.encode(input_format=source, **{kind: native})[kind]
    decoder = codec.codecs[target].decoder
    expected = getattr(decoder, kind)(intermediate)

    assert getattr(decoder, kind)(guarded_protocol(intermediate)) == expected


def test_sdk_view_caches_fields_without_building_children_on_presence_checks(monkeypatch):
    from my_llm.codec.shared import conversion

    reads = {"children": 0, "optional": 0, "missing": 0}
    wrapped = []

    class Child(BaseModel):
        text: str

    class Event(BaseModel):
        children: list[Child]
        optional: str | None = None

        def __getattribute__(self, key):
            if key in reads:
                reads[key] += 1
            return super().__getattribute__(key)

    original_view = conversion._view

    def track_view(value):
        if isinstance(value, (BaseModel, list)):
            wrapped.append(value)
        return original_view(value)

    event = Event(children=[Child(text="first"), Child(text="second")])
    monkeypatch.setattr(conversion, "_view", track_view)
    view = conversion.dictionaries([event])[0]
    assert len(wrapped) == 1

    assert "children" in view
    assert "children" in view
    assert len(wrapped) == 1
    children = view.get("children")
    assert view["children"] is children
    assert conversion.select(view, "children")["children"] is children
    assert len(wrapped) == 4  # One event, one list and its two child models.
    assert reads["children"] == 1

    assert "optional" not in view
    assert view.get("optional", "first fallback") == "first fallback"
    assert view.get("optional", "second fallback") == "second fallback"
    assert reads["optional"] == 1
    assert view.get("missing", "fallback") == "fallback"
    assert "missing" not in view
    with pytest.raises(KeyError, match="missing"):
        view["missing"]
    assert reads["missing"] == 1


def test_sdk_view_keeps_explicit_null_and_non_null_defaults():
    from my_llm.codec.shared.conversion import dictionaries, select

    class Event(BaseModel):
        optional: str | None = None
        defaulted: str = "standard default"

    assert select(dictionaries([Event()])[0], "optional", "defaulted") == {
        "defaulted": "standard default"
    }
    assert select(dictionaries([Event(optional=None)])[0], "optional", "defaulted") == {
        "optional": None,
        "defaulted": "standard default",
    }


def test_sdk_view_cache_is_local_to_each_public_translation():
    class CompatibleMessage(BaseModel):
        role: str = "assistant"
        content: str
        vendor_unconfigured: object

        def __getattribute__(self, key):
            if key == "vendor_unconfigured":
                raise AssertionError("read an unconfigured SDK field")
            return super().__getattribute__(key)

        def model_dump(self, *args, **kwargs):
            raise AssertionError("serialized a complete SDK model")

    message = CompatibleMessage(content="first", vendor_unconfigured=UnreadableExtension())
    encoder = MyCodec().codecs["openai_chat"].encoder
    first = encoder.messages([message])
    message.content = "second"
    second = encoder.messages([message])

    assert first[0]["content"] == [{"type": "text", "text": "first"}]
    assert second[0]["content"] == [{"type": "text", "text": "second"}]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("edit_input", [False, True])
def test_public_stream_encoder_owns_events(protocol, edit_input):
    values = stream_events(protocol, tool=True)
    before = deepcopy(values)
    converted = MyCodec().codecs[protocol].encoder.chunks(values)
    expected = deepcopy(converted)

    edit_containers(values if edit_input else converted)

    assert converted == expected if edit_input else values == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("edit_input", [False, True])
def test_public_stream_decoder_owns_events_and_parameters(source, target, edit_input):
    from my_llm.stream._state import StreamState

    codec = MyCodec()
    state = StreamState("ownership")
    operations = state.feed(codec.codecs[source].encoder.chunks(stream_events(source, tool=True)))
    operations.extend(state.finish())
    parameters = {"source_protocol": source, **state.metadata}
    before = deepcopy((operations, parameters))
    converted = codec.decoder(target).chunks(operations, **parameters)
    assert converted
    expected = deepcopy(converted)

    if edit_input:
        edit_containers(operations)
        edit_containers(parameters)
        assert converted == expected
    else:
        edit_containers(converted)
        assert (operations, parameters) == before


def stream_payload_cases():
    payload = {"nested": ["original"]}
    tool = {"type": "tool_use", "id": "call", "name": "lookup", "input": payload}
    text = {
        "type": "text",
        "text": "hello",
        "logprobs": [{"token": "hello", "logprob": -0.1, "bytes": [104], "top_logprobs": []}],
    }
    return [
        (
            "anthropic",
            [{"type": "content_block_start", "index": 0, "content_block": tool}],
            [{"type": "block", "index": 0, "block": {**tool, "type": "server_tool_use"}}],
            {},
        ),
        (
            "gemini",
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "function_call": {"name": "lookup", "args": payload},
                                        "part_metadata": payload,
                                    },
                                    {"function_response": {"name": "lookup", "response": payload}},
                                ]
                            }
                        }
                    ]
                }
            ],
            [
                {"type": "tool", "index": 0, "block": {**tool, "part_metadata": payload}},
                {
                    "type": "block",
                    "index": 1,
                    "block": {
                        "type": "tool_result",
                        "name": "lookup",
                        "tool_use_id": "call",
                        "content": payload,
                        "part_metadata": payload,
                    },
                },
                {
                    "type": "block_stop",
                    "index": 2,
                    "block": {
                        "type": "text",
                        "text": "hello",
                        "part_metadata": payload,
                    },
                },
            ],
            {},
        ),
        (
            "ollama",
            [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "lookup", "arguments": payload}},
                        ],
                    }
                }
            ],
            [{"type": "tool", "index": 0, "block": tool}],
            {},
        ),
        (
            "openai_responses",
            [
                {
                    "type": "response.created",
                    "response": {
                        "metadata": {"label": "original"},
                        "tools": [{"type": "function", "name": "lookup", "parameters": payload}],
                    },
                },
                {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "part": {
                        **text,
                        "type": "output_text",
                        "annotations": [],
                    },
                },
            ],
            [{"type": "start"}, {"type": "block_stop", "index": 0, "block": text}],
            {
                "response_parameters": {
                    "metadata": {"label": "original"},
                    "tools": [{"type": "function", "name": "lookup", "parameters": payload}],
                }
            },
        ),
    ]


@pytest.mark.parametrize("protocol,incoming,operations,parameters", stream_payload_cases())
@pytest.mark.parametrize("direction", ["encoder", "decoder"])
@pytest.mark.parametrize("edit_input", [False, True])
def test_stream_hooks_capture_recognized_mutable_leaves(
    protocol, incoming, operations, parameters, direction, edit_input
):
    # Parameterized fixtures must stay fresh when this test edits containers.
    incoming, operations, parameters = deepcopy((incoming, operations, parameters))
    values = incoming if direction == "encoder" else operations
    parameters = {} if direction == "encoder" else {"source_protocol": protocol, **parameters}
    before = deepcopy((values, parameters))
    converted = getattr(MyCodec().codecs[protocol], direction).chunks(values, **parameters)
    assert converted
    expected = deepcopy(converted)

    if edit_input:
        edit_containers(values)
        edit_containers(parameters)
        assert converted == expected
    else:
        edit_containers(converted)
        assert (values, parameters) == before


def test_translated_json_capture_preserves_aliases_cycles_and_custom_copy_semantics():
    from dataclasses import dataclass

    from my_llm.codec.shared.conversion import capture

    @dataclass
    class CustomValue:
        payload: object

    shared = {"items": ["original"]}
    custom = TrackedJSON({"nested": ["original"]})
    original = {
        "first": shared,
        "second": [shared],
        "custom": custom,
        "object": CustomValue(shared),
    }
    original["cycle"] = original

    cloned = capture(original)

    assert cloned is not original
    assert cloned["cycle"] is cloned
    assert cloned["first"] is cloned["second"][0] is cloned["object"].payload
    assert cloned["first"] is not shared
    assert type(cloned["custom"]) is TrackedJSON
    assert len(custom.copies) == 1
    cloned["first"]["items"].append("edit")
    cloned["custom"]["nested"].append("edit")
    assert shared == {"items": ["original"]}
    assert custom == {"nested": ["original"]}
