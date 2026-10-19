"""Cached schema plans retain filtering, selection and payload isolation."""

from collections.abc import Iterable
from copy import deepcopy
from gc import collect
from typing import Annotated, Any, Literal, NotRequired, Required
from weakref import ref

import pytest
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from my_llm import MyCodec, UnsupportedCapabilityError
from my_llm.codec.shared import transport
from tests.helpers.protocols import PROTOCOLS, history, tools


class Text(TypedDict):
    type: Literal["text"]
    text: str


class Image(TypedDict):
    type: Literal["image"]
    url: str


class Envelope(TypedDict):
    content: Annotated[Required[list[Text | Image]], "native content"]
    metadata: NotRequired[dict[str, Any]]


class Left(TypedDict):
    common: str
    left: str


class Right(TypedDict):
    common: str
    right: str


class Recursive(BaseModel):
    name: str
    children: list["Recursive"] = []


def test_cached_plan_filters_without_reanalyzing_types(monkeypatch):
    value = {"content": [{"type": "text", "text": "hello", "vendor": True}]}
    assert transport.supported(value, Envelope) == {"content": [{"type": "text", "text": "hello"}]}
    plan = transport._plan(Envelope)

    def unexpected(*args, **kwargs):
        pytest.fail("a cached filtering plan must not inspect annotations again")

    monkeypatch.setattr(transport, "get_origin", unexpected)
    monkeypatch.setattr(transport, "get_args", unexpected)
    monkeypatch.setattr(transport, "get_type_hints", unexpected)
    assert transport._plan(Envelope) is plan
    assert transport.supported(
        {"content": [{"type": "image", "url": "image.png", "vendor": 1}]}, Envelope
    ) == {"content": [{"type": "image", "url": "image.png"}]}
    assert transport.projected(value, Envelope) == {"content": [{"type": "text", "text": "hello"}]}


def test_union_literal_discriminator_takes_precedence_over_shared_fields():
    assert transport.supported(
        {"type": "image", "url": "image.png", "text": "vendor field"}, Text | Image
    ) == {"type": "image", "url": "image.png"}


def test_union_uses_highest_declared_field_score_and_first_candidate_on_ties():
    assert transport.supported({"common": "a", "right": "b"}, Left | Right) == {
        "common": "a",
        "right": "b",
    }
    value = {"common": "a", "left": "b", "right": "c"}
    assert transport.supported(value, Left | Right) == {"common": "a", "left": "b"}
    assert transport.supported(value, Right | Left) == {"common": "a", "right": "c"}


def test_cache_distinguishes_nested_union_order_and_reuses_equivalent_generics():
    value = [{"common": "a", "left": "b", "right": "c"}]
    assert transport.supported(value, list[Left | Right]) == [{"common": "a", "left": "b"}]
    assert transport.supported(value, list[Right | Left]) == [{"common": "a", "right": "c"}]
    assert transport._plan(list[Left | Right]) is transport._plan(list[Left | Right])


def test_union_rejects_values_no_candidate_represents():
    with pytest.raises(UnsupportedCapabilityError, match="target SDK"):
        transport.supported({"unexpected": True}, int | str)


def test_optional_and_wrapped_schemas_keep_existing_filtering_semantics():
    assert transport.supported(None, Text | None) is None
    assert transport.supported(None, int | str) is None
    assert transport.supported(
        {"type": "text", "text": "hello", "vendor": True},
        Annotated[Text | None, "optional native block"],
    ) == {"type": "text", "text": "hello"}


def test_annotations_with_unhashable_metadata_remain_supported():
    schema = Annotated[Text, {"description": ["native block"]}]
    value = {"type": "text", "text": "hello", "vendor": True}
    assert transport.supported(value, schema) == {"type": "text", "text": "hello"}
    assert transport._plan(schema) is transport._plan(schema)


@pytest.mark.parametrize("schema", [list[Text], tuple[Text, ...], Iterable[Text]])
def test_collection_plans_filter_list_items(schema):
    value = [{"type": "text", "text": "hello", "vendor": True}]
    assert transport.supported(value, schema) == [{"type": "text", "text": "hello"}]
    assert value[0]["vendor"] is True


def test_recursive_pydantic_plan_filters_nested_fields():
    value = {
        "name": "root",
        "vendor": True,
        "children": [{"name": "child", "vendor": True, "children": []}],
    }
    assert transport.supported(value, Recursive) == {
        "name": "root",
        "children": [{"name": "child", "children": []}],
    }
    model = Recursive(name="root", children=[Recursive(name="child")])
    copied = transport.supported(model, Recursive)
    assert copied == model.model_dump(mode="json", exclude_none=True)
    copied["children"][0]["name"] = "changed"
    assert model.children[0].name == "child"


@pytest.mark.parametrize("extra", [None, "ignore", "forbid", "allow"])
@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_pydantic_protocol_structures_filter_unknown_keys_regardless_of_extra_policy(
    extra, operation
):
    class Message(BaseModel):
        model_config = ConfigDict(extra=extra)
        content: list[Text]
        metadata: dict[str, Any]

    class Wrapper(TypedDict):
        message: Message

    value = {
        "content": [{"type": "text", "text": "hello", "vendor": "drop"}],
        "metadata": {"business": {"arbitrary": [1, None]}},
        "vendor": {"nested": [{"type": "custom", "new_key": True}]},
    }
    expected = {
        "content": [{"type": "text", "text": "hello"}],
        "metadata": value["metadata"],
    }
    original = deepcopy(value)
    assert operation(value, Message) == expected
    assert operation({"message": value, "outer_vendor": True}, Wrapper) == {"message": expected}
    assert value == original


def test_cached_model_handles_later_optional_fields_and_open_dictionary_keys(monkeypatch):
    class Message(BaseModel):
        model_config = ConfigDict(extra="allow")
        content: str
        metadata: dict[str, Any] | None = None

    assert transport.supported({"content": "first"}, Message) == {"content": "first"}
    plan = transport._plan(Message)

    def unexpected(*args, **kwargs):
        pytest.fail("a cached model must not inspect annotations again")

    monkeypatch.setattr(transport, "fields", unexpected)
    monkeypatch.setattr(transport, "get_origin", unexpected)
    monkeypatch.setattr(transport, "get_args", unexpected)
    value = {"content": "second", "metadata": {"new_key": [{"value": 1}]}, "vendor": True}
    copied = transport.supported(value, Message)
    projected = transport.projected(value, Message)
    expected = {"content": "second", "metadata": value["metadata"]}
    assert copied == projected == expected
    assert transport._plan(Message) is plan
    assert projected["metadata"] is value["metadata"]
    copied["metadata"]["new_key"][0]["value"] = 2
    assert value["metadata"]["new_key"][0]["value"] == 1


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_recursive_models_filter_unknown_protocol_fields_at_every_level(operation):
    class Node(BaseModel):
        model_config = ConfigDict(extra="allow")
        name: str
        children: list["Node"] = []

    value = {
        "name": "root",
        "vendor": {"private": True},
        "children": [{"name": "child", "children": [], "later": [1, None]}],
    }
    original = deepcopy(value)
    assert operation(value, Node) == {
        "name": "root",
        "children": [{"name": "child", "children": []}],
    }
    assert value == original


def test_open_json_and_mutable_payloads_remain_complete_and_isolated():
    payload = {
        "$defs": {"name": {"type": "string", "minLength": 2}},
        "additionalProperties": False,
        "application": [{"type": "image", "unknown_protocol_field": True}],
    }
    value = {"content": [], "metadata": payload}
    filtered = transport.supported(value, Envelope)
    assert filtered == value
    filtered["metadata"]["application"][0]["unknown_protocol_field"] = False
    assert payload["application"][0]["unknown_protocol_field"] is True
    next_value = {"content": [], "metadata": {"different": [1]}}
    assert transport.supported(next_value, Envelope) == next_value


def test_read_only_projection_borrows_application_payload_without_copying():
    class Payload(dict):
        def __deepcopy__(self, memo):
            pytest.fail("read-only projection must borrow atomic application payloads")

    payload = Payload({"application": [{"nested": [1, {"unknown": True}]}]})
    value = {
        "content": [{"type": "image", "url": "image.png", "vendor": True}],
        "metadata": payload,
        "vendor": "drop",
    }
    projected = transport.projected(value, Envelope)
    assert projected == {
        "content": [{"type": "image", "url": "image.png"}],
        "metadata": payload,
    }
    assert projected is not value
    assert projected["content"] is not value["content"]
    assert projected["content"][0] is not value["content"][0]
    assert projected["metadata"] is payload
    assert value["content"][0]["vendor"] is True


@pytest.mark.parametrize("schema", [Left | Right, Right | Left, list[Left | Right]])
def test_projection_and_copy_use_the_same_union_selection(schema):
    value = {"common": "a", "left": "b", "right": "c"}
    if schema == list[Left | Right]:
        value = [value]
    assert transport.projected(value, schema) == transport.supported(value, schema)


def test_projection_handles_optional_annotated_and_recursive_models():
    assert transport.projected(None, Annotated[Text | None, "block"]) is None
    value = {"name": "root", "children": [{"name": "child", "children": [], "vendor": True}]}
    assert transport.projected(value, Recursive) == transport.supported(value, Recursive)
    model = Recursive(name="root", children=[Recursive(name="child")])
    before = model.model_dump()
    assert transport.projected(model, Recursive) == model.model_dump(mode="json", exclude_none=True)
    assert model.model_dump() == before


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_cached_plan_does_not_keep_request_payload_alive(operation):
    class Payload(dict):
        pass

    value = Payload({"nested": [{"private request": True}]})
    observed = ref(value)
    filtered = operation(value, dict[str, Any])
    assert filtered == value
    del value, filtered
    collect()
    assert observed() is None


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_official_sdk_plans_preserve_native_fields_and_prune_protocol_extensions(protocol):
    bound = MyCodec().transport(protocol)
    for values, schema in [
        (history(protocol), bound.message_type),
        (tools(protocol), bound.tool_type),
    ]:
        original = deepcopy(values)
        for value in values:
            value["vendor_extension"] = {"nested": [1, None]}
        assert [transport.supported(value, schema) for value in values] == original
        assert all("vendor_extension" in value for value in values)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_official_sdk_validation_and_serialization_do_not_mutate_projected_inputs(protocol):
    bound = MyCodec().transport(protocol)
    declarations = tools(protocol)
    if protocol in {"openai_chat", "ollama"}:
        parameters = declarations[0]["function"]["parameters"]
    elif protocol == "openai_responses":
        parameters = declarations[0]["parameters"]
    elif protocol == "anthropic":
        parameters = declarations[0]["input_schema"]
    else:
        parameters = declarations[0]["function_declarations"][0]["parameters_json_schema"]
    parameters["properties"]["payload"] = {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {"type": "object", "properties": {"data": {"type": "string"}}},
            }
        },
        "application": [{"vendor": {"nested": [1, None, {"keep": True}]}}],
    }
    for values, schema in [
        (history(protocol), bound.message_type),
        (declarations, bound.tool_type),
    ]:
        original = deepcopy(values)
        validator = transport.adapter(schema)
        for value in values:
            projected = transport.projected(value, schema)
            before_validation = deepcopy(projected)
            parsed = validator.validate_python(projected)
            validator.dump_json(parsed, warnings="error")
            assert projected == before_validation
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                model = schema.model_validate(projected)
                before_model_validation = model.model_dump()
                projected_model = transport.projected(model, schema)
                assert isinstance(projected_model, dict)
                parsed_model = validator.validate_python(projected_model)
                validator.dump_json(parsed_model, warnings="error")
                assert model.model_dump() == before_model_validation
        assert values == original


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_sdk_model_instances_cannot_bypass_nested_structure_filtering(operation):
    class Nested(BaseModel):
        model_config = ConfigDict(extra="allow")
        text: str
        payload: dict[str, Any]

    class Message(BaseModel):
        model_config = ConfigDict(extra="allow")
        content: list[Nested]

    application = {"arbitrary": [{"type": "vendor", "keep": True}]}
    model = Message.model_validate(
        {
            "content": [{"text": "known", "payload": application, "nested_vendor": True}],
            "root_vendor": {"unknown": True},
        }
    )
    before = model.model_dump(mode="json", exclude_none=True)

    actual = operation(model, Message)

    assert actual == {"content": [{"text": "known", "payload": application}]}
    assert model.model_dump(mode="json", exclude_none=True) == before
