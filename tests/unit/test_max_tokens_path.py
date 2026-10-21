"""Output-token routing, precedence and ownership across protocol requests."""

from copy import deepcopy

import pytest

from my_llm import ConfigurationError, ModelProfile, ProviderProfile, omit, resolve_profiles
from my_llm.codec.shared.options import path_value
from tests.helpers.profiles import config, parameter
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
    profile = profile_factory(protocol, max_tokens=parameter("max_tokens", 2048, protocol=protocol))
    assert profile.parameter("max_tokens").path == path
    assert path_value(encode_request(profile), path) == 2048
    assert path_value(encode_request(profile, max_tokens=3072), path) == 3072


@pytest.mark.parametrize("protocol,path", DEFAULT_PATHS)
def test_omitted_path_disables_automatic_insertion(profile_factory, protocol, path):
    profile = profile_factory(protocol, max_tokens=omit)
    if protocol == "anthropic":
        with pytest.raises(ConfigurationError, match="requires max_tokens"):
            encode_request(profile, max_tokens=3072)
        assert encode_request(profile, extra_body={"max_tokens": 1024})["max_tokens"] == 1024
    else:
        assert path_value(encode_request(profile, max_tokens=3072), path) is None


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize(
    "path,body",
    [
        ("$.max_tokens", {"max_tokens": 12, "temperature": 0.2}),
        ("$.extra_body.max_tokens", {"max_tokens": 12, "temperature": 0.2}),
        ("$.extra_body.max_completion_tokens", {"temperature": 0.2}),
        ("$.extra_body.generation.limits[0].tokens", {"generation": {"mode": "custom"}}),
    ],
)
def test_custom_token_routes_override_body_and_preserve_siblings(
    profile_factory, protocol, path, body
):
    before = deepcopy(body)
    if protocol == "openai_responses" and path == "$.max_tokens":
        path = "$.max_output_tokens"
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 100, protocol=protocol, path=path),
        extra_body=config(body),
    )
    for tokens in (50, 75):
        request = encode_request(profile, max_tokens=tokens)
        assert path_value(request, path) == tokens
        if path == "$.max_tokens":
            assert request["extra_body"]["max_tokens"] == tokens
        for key in ("max_completion_tokens", "max_output_tokens"):
            if key != path.removeprefix("$."):
                assert key not in request
    assert body == before
    assert profile.get("extra_body") == before
    assert path_value(encode_request(profile, max_tokens=omit), path) is None


def test_token_route_wins_over_reasoning_bindings_without_mutating_them(profile_factory):
    binding = {"extra_body": {"limits": [{"tokens": 1, "mode": "custom"}]}}
    profile = profile_factory(
        reasoning_enable=parameter(
            "reasoning_enable",
            [{"tokens": 1, "mode": "custom"}],
            protocol="openai_chat",
            path="$.extra_body.limits",
        ),
        max_tokens=parameter(
            "max_tokens", 64, protocol="openai_chat", path="$.extra_body.limits[0].tokens"
        ),
    )
    request = encode_request(profile, reasoning=True, max_tokens=64)
    assert request["extra_body"]["limits"] == [{"tokens": 64, "mode": "custom"}]
    assert binding["extra_body"]["limits"][0]["tokens"] == 1


@pytest.mark.parametrize("path", ["$.model", "$.messages[0].content", "$.extra_body.stream"])
def test_token_routes_cannot_override_protocol_fields(profile_factory, path):
    with pytest.raises(ConfigurationError, match="cannot override protocol parameters"):
        encode_request(
            profile_factory(
                max_tokens=parameter("max_tokens", 64, protocol="openai_chat", path=path)
            )
        )


def test_incompatible_existing_path_container_is_a_configuration_error(profile_factory):
    with pytest.raises(ConfigurationError, match="path requires a dictionary"):
        encode_request(
            profile_factory(
                max_tokens=parameter(
                    "max_tokens", 64, protocol="openai_chat", path="$.extra_body.limits.tokens"
                ),
                extra_body=config({"limits": "scalar"}),
            )
        )


@pytest.mark.parametrize("protocol,path", DEFAULT_PATHS)
def test_unsupported_token_default_is_not_validated_or_inserted_into_requests(
    profile_factory, protocol, path
):
    profile = profile_factory(
        protocol,
        max_tokens=parameter(
            "max_tokens", 0, protocol=protocol, path="$.vendor.limit", supports=False
        ),
    )
    request = encode_request(
        profile, **({"extra_body": {"max_tokens": 128}} if protocol == "anthropic" else {})
    )
    assert "vendor" not in request
    assert path_value(request, path, "missing") == (128 if protocol == "anthropic" else "missing")


@pytest.mark.parametrize(
    "model_path,expected",
    [
        (None, "$.extra_body.max_tokens"),
        ("$.max_completion_tokens", "$.max_completion_tokens"),
        (omit, None),
    ],
)
def test_max_tokens_path_layering(model_path, expected):
    profile = resolve_profiles(
        ProviderProfile(
            protocol="openai_chat",
            max_tokens=parameter(
                "max_tokens", protocol="openai_chat", path="$.extra_body.max_tokens"
            ),
        ),
        ModelProfile(
            model="demo",
            max_tokens=(
                model_path
                if model_path is None or model_path is omit
                else parameter("max_tokens", path=model_path)
            ),
        ),
    )
    assert (
        profile.parameter("max_tokens").path if profile.parameter("max_tokens") else None
    ) == expected


@pytest.mark.parametrize(
    "path", ["", 1, True, [], "$", "$.extra_body..tokens", "$.limits[-1].tokens", "tokens[bad]"]
)
def test_invalid_max_tokens_path(path):
    with pytest.raises(ConfigurationError):
        resolve_profiles(
            ProviderProfile(
                protocol="openai_chat",
                max_tokens=parameter("max_tokens", protocol="openai_chat", path=path),
            ),
            ModelProfile(model="demo"),
        )


def test_max_tokens_default_is_omitted_and_explicit_none_is_a_value():
    profile = resolve_profiles(ProviderProfile(protocol="openai_chat"), ModelProfile(model="demo"))
    assert profile.get("max_tokens", omit) is omit
    assert profile.parameter("max_tokens").default_value is omit
    inherited = resolve_profiles(
        ProviderProfile(
            protocol="openai_chat", max_tokens=parameter("max_tokens", 2048, protocol="openai_chat")
        ),
        ModelProfile(model="demo"),
    )
    assert inherited.get("max_tokens") == 2048
    result = resolve_profiles(
        ProviderProfile(
            protocol="openai_chat", max_tokens=parameter("max_tokens", 2048, protocol="openai_chat")
        ),
        ModelProfile(model="demo", max_tokens=parameter("max_tokens", None)),
    )
    assert result.get("max_tokens", 100) is None
