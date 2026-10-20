"""Shared request preparation runs once and preserves validation and ownership."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from my_llm import ConfigurationError, MyCodec, UnsupportedCapabilityError, omit
from my_llm.codec.shared.options import path_value, resolve_options
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS
from tests.helpers.requests import encode_request


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_request_resolves_shared_options_once(profile_factory, protocol):
    profile = profile_factory(protocol, max_tokens=parameter("max_tokens", 4096, protocol=protocol))
    with patch("my_llm.codec.base.decoder.resolve_options", wraps=resolve_options) as resolve:
        request = encode_request(profile, reasoning=True)
    assert request["model"] == profile.model
    resolve.assert_called_once_with(profile, reasoning=True)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "parameters,match",
    [
        ({"reasoning": "enabled"}, "reasoning must be a bool"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"extra_body": {"model": "override"}}, "cannot override protocol parameters"),
    ],
)
def test_invalid_options_fail_before_translation(
    monkeypatch, profile_factory, protocol, parameters, match
):
    codec = MyCodec()
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")

    def must_not_translate(*args, **kwargs):
        pytest.fail("invalid generation options reached the Decoder")

    monkeypatch.setattr(type(codec.decoder(protocol)), "_messages", must_not_translate)
    with pytest.raises(ConfigurationError, match=match):
        codec.request(
            intermediate,
            profile_factory(protocol, max_tokens=parameter("max_tokens", 4096, protocol=protocol)),
            **parameters,
        )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_protected_binding_fails_before_translation(monkeypatch, profile_factory, protocol):
    codec = MyCodec()
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 4096, protocol=protocol),
        reasoning_enable=parameter(
            "reasoning_enable", "override", protocol=protocol, path="$.model"
        ),
    )

    def must_not_translate(*args, **kwargs):
        pytest.fail("a protected reasoning binding reached the Decoder")

    monkeypatch.setattr(type(codec.decoder(protocol)), "_messages", must_not_translate)
    with pytest.raises(ConfigurationError, match="cannot override protocol parameters"):
        codec.request(intermediate, profile, reasoning=True)


def test_responses_omitted_path_disables_generated_effort(profile_factory):
    profile = profile_factory(
        "openai_responses",
        reasoning_enable=parameter(
            "reasoning_enable", {"summary": "detailed"}, protocol="openai_responses"
        ),
        reasoning_effort=omit,
    )
    request = encode_request(profile, reasoning_effort="high")
    assert request["reasoning"] == {"summary": "detailed"}


@pytest.mark.parametrize(
    "protocol,body,body_path,request_path",
    [
        ("openai_chat", {"stop": ["value"]}, "stop", "extra_body.stop"),
        ("openai_responses", {"stop": ["value"]}, "stop", "extra_body.stop"),
        ("anthropic", {"stop_sequences": ["value"]}, "stop_sequences", "stop_sequences"),
        ("gemini", {"stop_sequences": ["value"]}, "stop_sequences", "config.stop_sequences"),
        ("ollama", {"options": {"stop": ["value"]}}, "options.stop", "options.stop"),
    ],
)
def test_prepared_options_and_requests_own_their_nested_values(
    profile_factory, protocol, body, body_path, request_path
):
    codec = MyCodec()
    profile_body = deepcopy(body)
    caller_body = deepcopy(body)
    path_value(profile_body, "$." + body_path)[:] = ["profile"]
    path_value(caller_body, "$." + body_path)[:] = ["caller"]
    caller_headers = {"x-source": "caller"}
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 4096, protocol=protocol),
        extra_body=config("extra_body", profile_body),
        extra_headers=config("extra_headers", {"x-source": "profile"}),
        reasoning_enable=parameter(
            "reasoning_enable", ["default"], protocol=protocol, path="$." + request_path
        ),
    )
    profile_before = deepcopy(profile.values)
    options = codec.decoder(protocol).validate_parameters(
        profile, reasoning=True, extra_body=caller_body, extra_headers=caller_headers
    )
    options_before = deepcopy(options)
    request = codec.decoder(protocol).request_parameters(profile, options)
    assert path_value(request, "$." + request_path) == ["caller"]

    path_value(caller_body, "$." + body_path).append("caller edit")
    caller_headers["x-source"] = "caller edit"
    assert path_value(request, "$." + request_path) == ["caller"]
    assert options == options_before

    path_value(request, "$." + request_path).append("request edit")
    if "extra_headers" in request:
        request["extra_headers"]["x-source"] = "request edit"
    assert options == options_before
    assert profile.values == profile_before
    assert path_value(profile_body, "$." + body_path) == ["profile"]
    assert path_value(
        codec.decoder(protocol).request_parameters(profile, options), "$." + request_path
    ) == ["caller"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_none_effort_mapping_preserves_configuration_and_rejects_requested_effort(
    profile_factory, protocol
):
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 4096, protocol=protocol),
        supports_reasoning_effort=config("Effort capability", True),
        reasoning_effort_mapping=config("Disabled effort mapping", None),
    )
    assert profile.get("reasoning_effort_mapping", "missing") is None
    with pytest.raises(UnsupportedCapabilityError, match="effort"):
        encode_request(profile, reasoning_effort="high")
