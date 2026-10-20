"""Declared defaults, presence, omission and Config precedence at SDK-root paths."""

from copy import deepcopy
from dataclasses import dataclass

import pytest

from my_llm import (
    Config,
    ModelProfile,
    Parameter,
    ProviderProfile,
    omit,
    resolve_profiles,
)
from my_llm.codec.shared.options import merge_parameters, path_value, set_path
from tests.helpers.profiles import config, parameter
from tests.helpers.requests import encode_request


@dataclass(frozen=True, kw_only=True)
class TemperatureProvider(ProviderProfile):
    temperature: Parameter[float | None] = Parameter(
        name="Temperature", path="$.temperature", default_value=0.4, phase="decode"
    )
    trace_enabled: Config[bool] = config("Trace", True)


CASES = [
    ("openai_chat", "$.extra_body.temperature", {"temperature": None}),
    ("openai_responses", "$.extra_body.temperature", {"temperature": None}),
    ("anthropic", "$.temperature", {"temperature": None}),
    ("gemini", "$.config.temperature", {"temperature": None}),
    ("ollama", "$.options.temperature", {"options": {"temperature": None}}),
]


def profile(protocol, path, value=0.4):
    return resolve_profiles(
        TemperatureProvider(
            protocol=protocol,
            temperature=Parameter(
                name="Temperature", path=path, default_value=value, phase="decode"
            ),
        ),
        ModelProfile(model="demo", max_tokens=parameter("max_tokens", 4096, protocol=protocol)),
    )


@pytest.mark.parametrize("protocol,path,body", CASES)
def test_missing_parameter_uses_inherited_declared_default(protocol, path, body):
    assert path_value(encode_request(profile(protocol, path)), path) == 0.4


@pytest.mark.parametrize("protocol,path,body", CASES)
def test_present_none_is_retained_instead_of_default(protocol, path, body):
    request = encode_request(profile(protocol, path), extra_body=body)
    assert path_value(request, path, "missing") is None


@pytest.mark.parametrize("protocol,path,body", CASES)
def test_default_can_be_explicit_none(protocol, path, body):
    assert path_value(encode_request(profile(protocol, path, None)), path, "missing") is None


@pytest.mark.parametrize("protocol,path,body", CASES)
def test_omitted_default_preserves_a_present_parameter(protocol, path, body):
    request = encode_request(profile(protocol, path, omit), extra_body=body)
    assert path_value(request, path, "missing") is None


@pytest.mark.parametrize("protocol,path,body", CASES)
def test_existing_false_or_zero_does_not_trigger_default(protocol, path, body):
    body = deepcopy(body)
    if protocol == "ollama":
        body["options"]["temperature"] = 0.0
    else:
        body["temperature"] = 0.0
    assert path_value(encode_request(profile(protocol, path), extra_body=body), path) == 0.0


def test_config_generated_effort_overrides_parameter_default(profile_factory):
    p = profile_factory(
        "openai_chat",
        supports_reasoning_effort=config("supports_reasoning_effort", True),
        reasoning_effort=parameter("reasoning_effort", "low", protocol="openai_chat"),
        reasoning_effort_mapping=config("reasoning_effort_mapping", {"high": "medium"}),
    )
    assert encode_request(p, reasoning_effort="high")["reasoning_effort"] == "medium"


def test_open_dictionary_omission_keeps_siblings_and_captures_owned_data():
    original = {"nested": {"keep": [1], "drop": "old"}}
    deletion = {"nested": {"drop": omit}}
    result = merge_parameters(original, deletion)
    assert result == {"nested": {"keep": [1]}}
    result["nested"]["keep"].append(2)
    assert original["nested"]["keep"] == [1]
    assert (
        merge_parameters(original, deletion, {"nested": {"drop": "new"}})["nested"]["drop"] == "new"
    )


def test_path_read_distinguishes_missing_from_explicit_none():
    assert path_value({"value": None}, "$.value", "missing") is None
    assert path_value({}, "$.value", "missing") == "missing"
    assert path_value({"value": None}, "$.value.child", "missing") == "missing"
    value = {"nested": {"keep": True}}
    set_path(value, "$.nested.items[1].name", "value")
    assert value == {"nested": {"keep": True, "items": [None, {"name": "value"}]}}


def test_display_metadata_does_not_control_business_parameters_or_branches(profile_factory):
    original = profile_factory(
        max_tokens=parameter("max_tokens", 120),
        supports_reasoning_effort=config("Effort", True),
        reasoning_effort_mapping=config("Mapping", {"high": "medium"}),
    )
    decorated = profile_factory(
        max_tokens=Parameter(
            name="用户可编辑的名称",
            comment="Only shown in the UI",
            path="$.max_completion_tokens",
            default_value=120,
            phase="decode",
        ),
        supports_reasoning_effort=Config(
            name="完全无关的标识",
            comment="No branch depends on this label",
            default_value=True,
        ),
        reasoning_effort_mapping=config("Different label", {"high": "medium"}),
    )
    assert encode_request(original, reasoning_effort="high") == encode_request(
        decorated, reasoning_effort="high"
    )


def test_parameter_default_and_resolved_value_share_one_independent_snapshot(profile_factory):
    payload = {"enabled": [True]}
    resolved = profile_factory(reasoning_enable=parameter("reasoning_enable", payload))
    binding = resolved.parameter("reasoning_enable")
    assert binding.default_value is resolved.get("reasoning_enable")
    assert binding.default_value is not payload
    payload["enabled"].append(False)
    assert binding.default_value == {"enabled": [True]}
