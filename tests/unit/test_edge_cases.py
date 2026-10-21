import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    ModelProfile,
    ProviderProfile,
    omit,
)
from my_llm.profile.resolution import resolve_profiles
from tests.helpers.profiles import config, parameter
from tests.helpers.requests import codec


def test_custom_chat_effort_named_parameter(profile_factory):
    profile = profile_factory(
        reasoning_effort=parameter("reasoning_effort", supports=True),
        reasoning_effort_mapping=config({"high": "medium"}),
    )
    request = codec(profile).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning_effort"] == "medium"
    assert "extra_body" not in request


def test_custom_response_summary_override(profile_factory):
    request = codec(
        profile_factory(
            "openai_responses",
            reasoning_enable=parameter(
                "reasoning_enable",
                {"summary": "auto", "effort": "medium"},
                protocol="openai_responses",
            ),
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="low")
    assert request["reasoning"] == {"summary": "auto", "effort": "low"}


def test_reasoning_boolean_enable_and_mapping_disabled(profile_factory):
    request = codec(profile_factory("openai_responses")).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning=True
    )
    assert request["reasoning"]["effort"] == "medium"
    request = codec(
        profile_factory("openai_responses", reasoning_enable=omit, reasoning_effort=omit)
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high")
    assert "reasoning" not in request


def test_malicious_reasoning_binding_cannot_disable_stream(profile_factory):
    profile = profile_factory(
        reasoning_enable=parameter(
            "reasoning_enable", False, protocol="openai_chat", path="$.extra_body.stream"
        )
    )
    with pytest.raises(ConfigurationError):
        codec(profile).encode(
            [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
        )


def test_chat_name_supported_by_default(profile_factory):
    request = codec(profile_factory()).encode(
        [{"role": "user", "name": "user1", "content": "x"}], tools=None
    )
    assert request["messages"][0]["name"] == "user1"
    assert "tools" not in request


def test_model_string_whitespace_rejected():
    with pytest.raises(ConfigurationError):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            ModelProfile(model=Attribute(default_value=" ")),
        )


def test_default_response_object_is_replaced_as_one_parameter(profile_factory):
    request = codec(
        profile_factory(
            "openai_responses",
            reasoning_enable=parameter(
                "reasoning_enable", {"effort": "medium"}, protocol="openai_responses"
            ),
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high")
    assert request["reasoning"] == {"effort": "high"}


def test_response_summary_can_be_explicitly_omitted(profile_factory):
    profile = profile_factory(
        "openai_responses",
        reasoning_enable=parameter(
            "reasoning_enable", {"summary": omit}, protocol="openai_responses"
        ),
    )
    request = codec(profile).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning"] == {"effort": "high"}
