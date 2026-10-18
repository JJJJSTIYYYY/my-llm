"""Output-token routing, precedence and ownership across protocol requests."""

from copy import deepcopy

import pytest

from my_llm import ConfigurationError, omit
from my_llm.codec.shared.values import path_value
from tests.helpers.requests import encode_request

DEFAULT_PATHS = [
    ("openai_chat", "$.max_completion_tokens"),
    ("openai_responses", "$.max_output_tokens"),
    ("anthropic", "$.max_tokens"),
    ("gemini", "$.config.max_output_tokens"),
    ("ollama", "$.options.num_predict"),
]


@pytest.mark.parametrize("protocol,path", DEFAULT_PATHS)
def test_protocol_token_paths_and_call_override(profile_factory, protocol, path):
    profile = profile_factory(protocol, max_tokens=2048)
    assert profile.get("max_tokens_path") == path
    assert path_value(encode_request(profile), path) == 2048
    assert path_value(encode_request(profile, max_tokens=3072), path) == 3072


@pytest.mark.parametrize("protocol,path", DEFAULT_PATHS)
def test_omitted_path_disables_automatic_insertion(profile_factory, protocol, path):
    profile = profile_factory(protocol, max_tokens=2048, max_tokens_path=omit)
    assert path_value(encode_request(profile, max_tokens=3072), path) is None


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize(
    "path,body",
    [
        ("max_tokens", {"max_tokens": 12, "temperature": 0.2}),
        ("$.extra_body.max_tokens", {"max_tokens": 12, "temperature": 0.2}),
        ("extra_body.max_completion_tokens", {"temperature": 0.2}),
        ("$.extra_body.generation.limits[0].tokens", {"generation": {"mode": "custom"}}),
    ],
)
def test_custom_token_routes_override_body_and_preserve_siblings(
    profile_factory, protocol, path, body
):
    before = deepcopy(body)
    profile = profile_factory(protocol, max_tokens=100, max_tokens_path=path, extra_body=body)
    for tokens in (50, 75):
        request = encode_request(profile, max_tokens=tokens)
        assert path_value(request, path) == tokens
        if path == "max_tokens":
            assert request["extra_body"]["max_tokens"] == tokens
        for key in ("max_completion_tokens", "max_output_tokens"):
            assert key not in request
    assert body == before
    assert profile.get("extra_body") == before
    assert path_value(encode_request(profile, max_tokens=omit), path) == (
        12 if path == "$.extra_body.max_tokens" else None
    )


def test_token_route_wins_over_reasoning_bindings_without_mutating_them(profile_factory):
    binding = {"extra_body": {"limits": [{"tokens": 1, "mode": "custom"}]}}
    profile = profile_factory(
        max_tokens_path="extra_body.limits[0].tokens",
        reasoning_enable_parameter=binding,
        max_tokens=64,
    )
    request = encode_request(profile, reasoning=True)
    assert request["extra_body"]["limits"] == [{"tokens": 64, "mode": "custom"}]
    assert binding["extra_body"]["limits"][0]["tokens"] == 1


@pytest.mark.parametrize("path", ["model", "$.messages[0].content", "$.extra_body.stream"])
def test_token_routes_cannot_override_protocol_fields(profile_factory, path):
    with pytest.raises(ConfigurationError, match="cannot override protocol parameters"):
        encode_request(profile_factory(max_tokens=64, max_tokens_path=path))


def test_incompatible_existing_path_container_is_a_configuration_error(profile_factory):
    with pytest.raises(ConfigurationError, match="path requires a dictionary"):
        encode_request(
            profile_factory(
                max_tokens=64,
                max_tokens_path="extra_body.limits.tokens",
                extra_body={"limits": "scalar"},
            )
        )
