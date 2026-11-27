"""Unambiguous extraction of complete JSON objects from model responses."""

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel

from my_llm.utils.json_schema import object_json_schema, parse_json_object


@pytest.mark.parametrize(
    "template",
    [
        "{}",
        " \ufeff{}\n",
        "Here is the result:\n{}\nExplanation follows.",
        "Here is the result: ```json\n{}\n```",
        "```json\n{}\n```",
        "```JSON\r\n{}\r\n```",
        "  ~~~application/json\n{}\n  ~~~~",
        "````\n{}\n`````",
        "```json\n{}",
        "```\n{}",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        {},
        {"中文": '引号" 与 \\ 及 { } [ ]', "items": [False, None, {"n": 0, "f": 1.25}]},
    ],
)
def test_extract_objects_with_wrappers_and_escaped_content(template, value):
    assert parse_json_object(template.format(json.dumps(value, ensure_ascii=False))) == value


@pytest.mark.parametrize(
    "text",
    [
        'Example: {"example":true}\n```json\n{"answer":42}\n```',
        '```\n{"example":true}\n```\n```json\n{"answer":42}\n```',
        '```python\n{"example":true}\n```\nThe answer: {"answer":42}',
        '{not JSON}\nThe answer: {"answer":42}',
        'Example: {"example":true}\n```json\n{"answer":42}',
        '[reference]\nThe answer: {"answer":42}',
        '[{"example":true}]\nThe answer: {"answer":42}',
        '"Escaped example: {\\"example\\":true}"\nThe answer: {"answer":42}',
    ],
)
def test_skip_examples_without_extracting_nested_or_quoted_objects(text):
    assert parse_json_object(text) == {"answer": 42}


@pytest.mark.parametrize(
    "text",
    [
        '{"outer":{"partial":true}',
        '{"outer":{"partial":true},"unfinished":',
        '{"outer":{"partial":true},"invalid":}',
        '{"outer":{"partial":true},"text":"unterminated}',
        'Result: [{"nested":true}]',
        '```json\n[{"nested":true}]\n```',
        'Result: [{"nested":true}',
        'Result: {"outer":[{"nested":true}}',
        'Result: "Escaped object: {\\"nested\\":true}"',
        '{"a":1}\n{"b":2}',
        '```json\n{"a":1}\n```\n```json\n{"b":2}\n```',
        '```\n{"a":1}\n{"b":2}\n```',
        '{"complete":true}\n{"unfinished":',
        '{"key":1,"key":2}',
        '{"outer":{"key":1,"key":2}}',
        '{"a":1,"\\u0061":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e9999}',
        '{"outer":{"value":NaN}}',
        '{"outer":{"ok":true},"value":NaN}',
        '{"text":"literal\nnewline"}',
        '{"text":"bad \\x escape"}',
        '{"text":"bad \\u123 escape"}',
        '```python\n{"example":true}\n```',
        'Example: {"example":true}\n```json\n{"invalid":}\n```',
        'Example: {"example":true}\n```json\n{"invalid":}',
        '```python\n{"example":true}',
        'Example: {"example":true}\n```json\nnull\n```',
        'Example: {"example":true}\n```\n[]\n```',
        pytest.param("{" * 10000, id="many-unclosed-braces"),
    ],
)
def test_reject_partial_ambiguous_or_non_json_objects(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_decoder_recursion_error_is_reported_as_value_error(monkeypatch):
    def fail_decode(self, text):
        raise RecursionError("JSON nesting limit")

    monkeypatch.setattr(json.JSONDecoder, "decode", fail_decode)
    with pytest.raises(ValueError):
        parse_json_object('{"outer":{}}')


class _Details(BaseModel):
    note: str


class _NestedPayload(BaseModel):
    details: _Details
    repeated: list[_Details]


@pytest.mark.parametrize("schema", [_NestedPayload, _NestedPayload.model_json_schema()])
def test_inline_pydantic_references_without_mutating_input(schema):
    before = deepcopy(schema)
    result = object_json_schema(schema)
    assert "$defs" not in result
    assert result["properties"]["details"]["type"] == "object"
    assert result["properties"]["details"]["properties"]["note"]["type"] == "string"
    assert result["properties"]["repeated"]["items"]["type"] == "object"
    result["properties"]["details"]["properties"]["note"]["type"] = "integer"
    assert result["properties"]["repeated"]["items"]["properties"]["note"]["type"] == "string"
    assert schema == before


@pytest.mark.parametrize("definitions", ["$defs", "definitions"])
def test_inline_chained_and_escaped_json_pointers(definitions):
    schema = {
        "type": "object",
        definitions: {
            "a/b~c": {"type": "string"},
            "alias": {"$ref": f"#/{definitions}/a~1b~0c"},
        },
        "properties": {
            "value": {"$ref": f"#/{definitions}/alias", "description": "field description"}
        },
    }
    assert object_json_schema(schema)["properties"]["value"] == {
        "type": "string",
        "description": "field description",
    }


def test_inline_percent_encoded_pointer_and_root_reference():
    schema = {
        "$defs": {"数据": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "$ref": "#/$defs/%E6%95%B0%E6%8D%AE",
    }
    assert object_json_schema(schema) == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


def test_schema_data_values_and_property_names_are_not_interpreted_as_references():
    data = {"$ref": "literal-data", "$defs": {"keep": True}}
    schema = {
        "type": "object",
        "properties": {
            "$ref": {"type": "object", "default": data, "const": data, "examples": [data]},
            "definitions": {"type": "string"},
        },
        "$defs": {"unused": {"$ref": "#/missing"}},
    }
    result = object_json_schema(schema)
    assert "$defs" not in result
    assert result["properties"]["$ref"] == schema["properties"]["$ref"]
    assert result["properties"]["definitions"] == {"type": "string"}
    result["properties"]["$ref"]["default"]["$defs"]["keep"] = False
    assert data["$defs"]["keep"] is True


@pytest.mark.parametrize("minimum", [0, 5])
def test_ref_sibling_constraints_preserve_validation_semantics(minimum):
    schema = {
        "type": "object",
        "$defs": {"N": {"type": "integer", "minimum": 0}},
        "properties": {"value": {"$ref": "#/$defs/N", "minimum": minimum, "maximum": 10}},
    }
    inline = object_json_schema(schema)
    for value in (-1, 0, 4, 5, 10, 11, "5"):
        instance = {"value": value}
        assert Draft202012Validator(schema).is_valid(instance) == Draft202012Validator(
            inline
        ).is_valid(instance)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"child": {"$ref": "#"}}},
        {"$defs": {"A": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"},
        {"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"},
    ],
)
def test_circular_references_fail_explicitly(schema):
    with pytest.raises(ValueError, match="circular"):
        object_json_schema(schema)


@pytest.mark.parametrize(
    "ref",
    [
        "https://example.invalid/schema.json",
        "other.json#/$defs/A",
        "#anchor",
        "#/missing",
        "#/properties/~2",
    ],
)
def test_unsupported_or_unresolved_references_fail_explicitly(ref):
    with pytest.raises(ValueError):
        object_json_schema({"type": "object", "properties": {"value": {"$ref": ref}}})


def test_inline_refs_under_schema_keywords_and_array_pointer():
    schema = {
        "type": "object",
        "$defs": {"S": {"type": "string"}, "F": False},
        "allOf": [{"type": "object"}],
        "properties": {
            "array": {
                "type": "array",
                "prefixItems": [{"$ref": "#/$defs/S"}],
                "items": {"$ref": "#/$defs/F"},
            },
            "map": {"type": "object", "additionalProperties": {"$ref": "#/$defs/S"}},
            "copy": {"$ref": "#/allOf/0"},
        },
        "dependentSchemas": {"array": {"properties": {"other": {"$ref": "#/$defs/S"}}}},
    }
    inline = object_json_schema(schema)
    assert "$ref" not in json.dumps(inline)
    assert inline["properties"]["array"]["items"] is False
    assert inline["properties"]["copy"] == {"type": "object"}
