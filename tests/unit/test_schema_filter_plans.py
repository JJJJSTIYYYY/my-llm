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


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_literal_pruning_keeps_overlapping_and_unrestricted_union_candidates(operation):
    class Broad(TypedDict):
        type: Literal["text", "image"]
        common: str

    class Narrow(TypedDict):
        type: Literal["text"]
        text: str

    value = {"type": "text", "common": "shared", "text": "answer"}
    assert operation(value, Broad | Narrow) == {"type": "text", "common": "shared"}
    assert operation(value, Narrow | Broad) == {"type": "text", "text": "answer"}
    # Matching a literal cannot bypass a stronger branch without that field.
    value = {"type": "text", "common": "a", "left": "b"}
    assert operation(value, Narrow | Left) == {"common": "a", "left": "b"}
    assert operation({"type": "vendor", "common": "a", "left": "b"}, Narrow | Left) == {
        "common": "a",
        "left": "b",
    }
    # Missing optional discriminators keep the ordinary field-score behavior.
    assert operation({"text": "answer"}, Broad | Narrow) == {"text": "answer"}


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_literal_pruning_combines_declared_discriminators_without_reading_unknowns(operation):
    class User(TypedDict):
        type: Literal["message"]
        role: Literal["user"]
        text: str

    class Assistant(TypedDict):
        type: Literal["message"]
        role: Literal["assistant"]
        text: str

    class AddressedDict(dict):
        def __iter__(self):
            pytest.fail("union routing must not enumerate input fields")

        def get(self, key, default=None):
            assert key in ("type", "role", "text")
            return super().get(key, default)

    value = AddressedDict(type="message", role="assistant", text="answer", vendor=object())
    assert operation(value, User | Assistant) == {
        "type": "message",
        "role": "assistant",
        "text": "answer",
    }
    with pytest.raises(UnsupportedCapabilityError, match="target SDK"):
        operation({"type": "message", "role": "vendor", "text": "answer"}, User | Assistant)


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_explicit_union_discriminator_does_not_score_every_sdk_variant(operation):
    variants = [
        TypedDict(f"Variant{index}", {"type": Literal[f"variant_{index}"], "text": str})
        for index in range(32)
    ]
    schema = variants[0]
    for variant in variants[1:]:
        schema |= variant

    class CountedDict(dict):
        reads = 0

        def get(self, key, default=None):
            if key == "type":
                self.reads += 1
            return super().get(key, default)

    value = CountedDict(type="variant_31", text="answer")
    assert operation(value, schema) == value
    assert value.reads <= 3, "an explicit discriminator must prune unrelated variants"


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_literal_routing_keeps_sdk_model_defaults_none_and_recursive_content(operation):
    class Leaf(BaseModel):
        type: Literal["leaf"] = "leaf"
        text: str | None = None

    class Branch(BaseModel):
        type: Literal["branch"] = "branch"
        children: list["Leaf | Branch"] = []

    model = Branch(children=[Leaf(text=None), Branch(children=[Leaf()])])
    assert operation(model, Leaf | Branch) == {
        "type": "branch",
        "children": [
            {"type": "leaf", "text": None},
            {"type": "branch", "children": [{"type": "leaf"}]},
        ],
    }
    assert operation(
        {"type": "branch", "children": [{"type": "leaf", "text": None, "vendor": True}]},
        Leaf | Branch,
    ) == {"type": "branch", "children": [{"type": "leaf", "text": None}]}


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_unhashable_discriminator_values_keep_union_rejection(operation):
    with pytest.raises(UnsupportedCapabilityError, match="target SDK"):
        operation({"type": [], "text": "answer"}, Text | Image)


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


def test_unhashable_annotations_share_validation_and_path_cache_rules():
    schema = Annotated[Text, {"description": ["independent protocol block"]}]
    value = {"type": "text", "text": "hello"}
    validator = transport.adapter(schema)
    assert validator is transport.adapter(schema)
    assert validator.validate_python(value) == value
    assert transport.declared_path(schema, ("text",))
    assert not transport.declared_path(schema, ("vendor",))


def test_validator_cache_keeps_union_declaration_order():
    first = transport.adapter(Left | Right)
    second = transport.adapter(Right | Left)
    assert first is not second
    assert transport.adapter(Left | Right) is first


def test_distinct_protocol_types_with_identical_names_do_not_share_rules():
    first = TypedDict("NativeMessage", {"actor": str, "utterance": str})
    second = TypedDict("NativeMessage", {"sender": str, "payload": str})
    value = {"actor": "human", "utterance": "first", "sender": "bot", "payload": "second"}
    assert transport.supported(value, first) == {"actor": "human", "utterance": "first"}
    assert transport.supported(value, second) == {"sender": "bot", "payload": "second"}
    assert transport._plan(first) is not transport._plan(second)
    assert transport.adapter(first) is not transport.adapter(second)


def test_dynamic_protocol_types_evict_cached_rules_without_changing_results():
    caches = (transport.fields, transport._cached_adapter, transport._compiled_plan)
    assert all(cache.cache_info().maxsize == 512 for cache in caches)
    first = TypedDict("FirstDynamicProtocol", {"body": str})
    original_plan = transport._plan(first)
    original_adapter = transport.adapter(first)
    for index in range(513):
        schema = TypedDict(f"DynamicProtocol{index}", {"body": str})
        assert transport.supported({"body": "hello", "vendor": True}, schema) == {"body": "hello"}
        transport.adapter(schema)
    assert all(cache.cache_info().currsize <= 512 for cache in caches)
    assert transport._plan(first) is not original_plan
    assert transport.adapter(first) is not original_adapter
    assert transport.supported({"body": "new request", "vendor": True}, first) == {
        "body": "new request"
    }


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


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_protocol_projection_never_enumerates_unknown_dictionary_fields(operation):
    class AddressedDict(dict):
        def __iter__(self):
            pytest.fail("protocol filtering must address declared fields")

        def items(self):
            pytest.fail("protocol filtering must not enumerate input fields")

        def keys(self):
            pytest.fail("union selection must not enumerate input fields")

    class Unreadable:
        def __deepcopy__(self, memo):
            pytest.fail("unknown values must not be read or copied")

    value = AddressedDict(
        content=[AddressedDict(type="image", url="image.png", vendor=Unreadable())],
        vendor=Unreadable(),
    )
    assert operation(value, Envelope) == {"content": [{"type": "image", "url": "image.png"}]}


@pytest.mark.parametrize("operation", [transport.supported, transport.projected])
def test_sdk_models_are_addressed_without_dumping_extensions_and_keep_explicit_none(operation):
    class Message(BaseModel):
        model_config = ConfigDict(extra="allow")
        content: str | None = None
        metadata: dict[str, Any] | None = None

        def model_dump(self, *args, **kwargs):
            pytest.fail("protocol projection must not serialize unknown SDK fields")

    message = Message(content=None, vendor=object())
    assert operation(message, Message) == {"content": None}


def test_delegated_paths_isolate_plan_caches_and_same_type_at_different_positions():
    class Payload(TypedDict):
        application: dict[str, Any]

    class Pair(TypedDict):
        first: Payload
        second: Payload

    first_path = (("first", "application"),)
    second_path = (("second", "application"),)
    plan = transport._plan(Pair, first_path)
    assert transport._plan(Pair, first_path) is plan
    assert transport._plan(Pair, second_path) is not plan
    assert transport._plan(Pair) is not plan
    value = {
        "first": {"application": {"nested": [1]}},
        "second": {"application": {"nested": [2]}},
    }

    first = transport.supported(value, Pair, delegated_paths=first_path)
    assert first["first"] is not value["first"]
    assert first["first"]["application"] is value["first"]["application"]
    assert first["second"]["application"] is not value["second"]["application"]
    second = transport.supported(value, Pair, delegated_paths=second_path)
    assert second["first"]["application"] is not value["first"]["application"]
    assert second["second"]["application"] is value["second"]["application"]
    ordinary = transport.supported(value, Pair)
    ordinary["first"]["application"]["nested"].append(3)
    ordinary["second"]["application"]["nested"].append(4)
    assert value["first"]["application"]["nested"] == [1]
    assert value["second"]["application"]["nested"] == [2]


def test_delegated_paths_are_consumed_at_the_correct_recursive_position():
    class Node(BaseModel):
        application: dict[str, Any]
        child: "Node | None" = None

    value = {
        "application": {"nested": [1]},
        "child": {
            "application": {"nested": [2]},
            "child": {"application": {"nested": [3]}},
        },
    }
    actual = transport.supported(value, Node, delegated_paths=(("child", "application"),))

    assert actual["application"] is not value["application"]
    assert actual["child"] is not value["child"]
    assert actual["child"]["application"] is value["child"]["application"]
    assert actual["child"]["child"]["application"] is not value["child"]["child"]["application"]
    ordinary = transport.supported(value, Node)
    assert ordinary["child"]["application"] is not value["child"]["application"]


def test_whole_delegated_transport_leaf_needs_no_filter_plan(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a whole delegated leaf must reach its owner without a second plan")

    monkeypatch.setattr(transport, "_plan", unexpected)
    value = [{"type": "text", "text": "answer"}]
    assert transport.supported(value, list[Text], delegated_paths=((),)) is value
