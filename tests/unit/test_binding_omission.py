"""All protocol bindings merge before consuming omission markers."""

import json
from copy import deepcopy

import pytest

from my_llm import MyCodec, omit
from my_llm.codec.shared.options import merge_bindings, path_value
from tests.helpers.requests import encode_request


@pytest.mark.parametrize(
    "protocol,key,value,replacement,path",
    [
        ("openai_chat", "temperature", 0.2, 0.6, "extra_body"),
        ("openai_responses", "temperature", 0.2, 0.6, "extra_body"),
        ("anthropic", "temperature", 0.2, 0.6, ""),
        ("gemini", "temperature", 0.2, 0.6, "config"),
        ("ollama", "keep_alive", "5m", "10m", ""),
    ],
)
@pytest.mark.parametrize("remove", [True, False])
def test_protocol_bindings_delete_or_replace_previous_body_values(
    profile_factory, protocol, key, value, replacement, path, remove
):
    profile = profile_factory(
        protocol,
        max_tokens=4096,
        supports_reasoning_effort=True,
        reasoning_enable_parameter={"extra_body": {key: value}},
        reasoning_effort_parameter={"extra_body": {key: omit if remove else replacement}},
    )
    request = encode_request(profile, reasoning_effort="high")
    serialized = json.loads(
        json.dumps(request, default=lambda value: value.model_dump(mode="json", exclude_none=True))
    )
    body = path_value(serialized, path) if path else serialized
    if remove:
        assert body is None or key not in body
    else:
        assert body[key] == replacement


@pytest.mark.parametrize(
    "protocol,body,path",
    [
        ("openai_chat", {"vendor": {"keep": "value", "drop": "old"}}, "vendor"),
        ("openai_responses", {"vendor": {"keep": "value", "drop": "old"}}, "vendor"),
        (
            "anthropic",
            {
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {"type": "object", "description": "old"},
                    }
                }
            },
            "output_config.format.schema",
        ),
        (
            "gemini",
            {"response_json_schema": {"type": "object", "description": "old"}},
            "response_json_schema",
        ),
        ("ollama", {"options": {"top_p": 0.9, "temperature": 0.2}}, "options"),
    ],
)
def test_nested_omission_keeps_unmodified_siblings_and_prepared_inputs(
    profile_factory, protocol, body, path
):
    key = {"anthropic": "description", "gemini": "description", "ollama": "temperature"}.get(
        protocol, "drop"
    )
    deletion = deepcopy(body)
    container = path_value(deletion, path)
    container.clear()
    container[key] = omit
    profile = profile_factory(
        protocol,
        max_tokens=4096,
        supports_reasoning_effort=True,
        reasoning_enable_parameter={"extra_body": body},
        reasoning_effort_parameter={"extra_body": deletion},
    )
    codec = MyCodec()
    options = codec.transport(protocol).validate_parameters(profile, reasoning_effort="high")
    expected = {field: value for field, value in path_value(body, path).items() if field != key}
    assert path_value(options.body, path) == expected
    marker = path_value(options.bindings[1]["extra_body"], path)[key]
    before = deepcopy(options, {id(marker): marker})
    request = codec.decoder(protocol).request_parameters(profile, options)
    request_body = (
        request["extra_body"]
        if protocol in ("openai_chat", "openai_responses")
        else request["config"]
        if protocol == "gemini"
        else request
    )
    expected_request = {**expected, "num_predict": 4096} if protocol == "ollama" else expected
    assert path_value(request_body, path) == expected_request
    assert options == before
    assert key in path_value(body, path)
    assert isinstance(path_value(deletion, path)[key], type(omit))


def test_common_binding_merge_deletes_then_allows_explicit_reintroduction():
    original = {"nested": {"keep": [1], "drop": "old"}}
    deletion = {"nested": {"drop": omit}}
    assert merge_bindings(original, deletion) == {"nested": {"keep": [1]}}
    assert merge_bindings(original, deletion, {"nested": {"drop": "new"}}) == {
        "nested": {"keep": [1], "drop": "new"}
    }
    assert merge_bindings(original, {"nested": omit}) == {}
    assert merge_bindings(original, {"nested": []}) == {"nested": []}
    assert original == {"nested": {"keep": [1], "drop": "old"}}


def test_common_binding_merge_replaces_lists_and_cleans_their_nested_dictionaries():
    result = merge_bindings(
        {"values": ["old"]},
        {"values": [[{"keep": "new", "drop": omit}]]},
    )
    assert result == {"values": [[{"keep": "new"}]]}


def test_responses_does_not_reintroduce_an_omitted_summary_from_defaults(profile_factory):
    profile = profile_factory(
        "openai_responses",
        reasoning_enable_parameter={"reasoning": {"summary": omit}},
        reasoning_effort_parameter={"reasoning": {}},
    )
    request = encode_request(profile, reasoning_effort="high")
    assert request["reasoning"] == {"effort": "high"}


def test_chat_binding_omission_survives_profile_resolution(profile_factory):
    profile = profile_factory(
        "openai_chat",
        supports_reasoning_effort=True,
        reasoning_effort_parameter={"reasoning_effort": omit},
    )
    assert isinstance(profile.get("reasoning_effort_parameter")["reasoning_effort"], type(omit))
    request = encode_request(profile, reasoning_effort="high")
    assert "reasoning_effort" not in request


def test_binding_deletes_sdk_option_fields_without_losing_siblings(profile_factory):
    from ollama import Options

    original = Options(temperature=0.2, top_p=0.9)
    profile = profile_factory(
        "ollama",
        max_tokens=4096,
        extra_body={"options": original},
        reasoning_enable_parameter={"extra_body": {"options": {"temperature": omit}}},
    )
    request = encode_request(profile, reasoning=True)
    assert request["options"] == {"top_p": 0.9, "num_predict": 4096}
    assert original.temperature == 0.2
    assert original.top_p == 0.9
