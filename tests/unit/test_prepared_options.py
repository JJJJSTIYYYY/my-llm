"""Shared request preparation runs once and preserves validation and ownership."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from my_llm import ConfigurationError, MyCodec, omit
from my_llm.codec.shared.options import path_value, resolve_options
from tests.helpers.protocols import PROTOCOLS
from tests.helpers.requests import encode_request


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_request_resolves_shared_options_once(profile_factory, protocol):
    profile = profile_factory(protocol, max_tokens=4096)
    with patch("my_llm.codec.base.transport.resolve_options", wraps=resolve_options) as resolve:
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
def test_invalid_options_fail_before_decoder(
    monkeypatch, profile_factory, protocol, parameters, match
):
    codec = MyCodec()
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")

    def must_not_translate(*args, **kwargs):
        pytest.fail("invalid generation options reached the Decoder")

    monkeypatch.setattr(type(codec.decoder(protocol)), "request", must_not_translate)
    with pytest.raises(ConfigurationError, match=match):
        codec.request(intermediate, profile_factory(protocol, max_tokens=4096), **parameters)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_protected_binding_fails_before_decoder(monkeypatch, profile_factory, protocol):
    codec = MyCodec()
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")
    profile = profile_factory(
        protocol,
        max_tokens=4096,
        reasoning_enable_parameter={"extra_body": {"model": "override"}},
    )

    def must_not_translate(*args, **kwargs):
        pytest.fail("a protected reasoning binding reached the Decoder")

    monkeypatch.setattr(type(codec.decoder(protocol)), "request", must_not_translate)
    with pytest.raises(ConfigurationError, match="cannot override protocol parameters"):
        codec.request(intermediate, profile, reasoning=True)


def test_responses_effort_omit_removes_the_previous_binding(profile_factory):
    codec = MyCodec()
    profile = profile_factory(
        "openai_responses", reasoning_effort_parameter={"reasoning": {"effort": omit}}
    )
    options = codec.transport(profile.protocol).validate_parameters(
        profile, reasoning_effort="high"
    )
    request = codec.decoder(profile.protocol).request_parameters(profile, options)
    assert isinstance(options.bindings[1]["reasoning"]["effort"], type(omit))
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
    binding_body = deepcopy(body)
    path_value(profile_body, body_path)[:] = ["profile"]
    path_value(caller_body, body_path)[:] = ["caller"]
    path_value(binding_body, body_path)[:] = ["binding"]
    binding = {"extra_body": binding_body}
    caller_headers = {"x-source": "caller"}
    profile = profile_factory(
        protocol,
        max_tokens=4096,
        extra_body=profile_body,
        extra_headers={"x-source": "profile"},
        reasoning_enable_parameter=binding,
    )
    profile_before = deepcopy(profile.values)
    options = codec.transport(protocol).validate_parameters(
        profile, reasoning=True, extra_body=caller_body, extra_headers=caller_headers
    )
    options_before = deepcopy(options)
    request = codec.decoder(protocol).request_parameters(profile, options)
    assert path_value(request, request_path) == ["binding"]

    path_value(caller_body, body_path).append("caller edit")
    path_value(binding_body, body_path).append("binding edit")
    caller_headers["x-source"] = "caller edit"
    assert path_value(request, request_path) == ["binding"]
    assert options == options_before

    path_value(request, request_path).append("request edit")
    if "extra_headers" in request:
        request["extra_headers"]["x-source"] = "request edit"
    assert options == options_before
    assert profile.values == profile_before
    assert path_value(profile_body, body_path) == ["profile"]
    assert path_value(
        codec.decoder(protocol).request_parameters(profile, options), request_path
    ) == ["binding"]
