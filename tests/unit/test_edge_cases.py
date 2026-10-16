import pytest
from test_encoding import codec

from my_llm import ConfigurationError, ModelProfile, ProviderProfile, omit
from my_llm.profile.utils.resolution import resolve_profiles


def test_custom_chat_effort_named_parameter(profile_factory):
    profile = profile_factory(
        supports_reasoning_effort=True, reasoning_effort_mapping={"high": "medium"}
    )
    request = codec(profile).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning_effort"] == "medium"
    assert request["extra_body"]["thinking"]["type"] == "enabled"


def test_custom_response_summary_override(profile_factory):
    request = codec(
        profile_factory(
            "openai_responses",
            reasoning_effort_parameter={"reasoning": {"summary": "auto", "effort": "{effort}"}},
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="low")
    assert request["reasoning"] == {"summary": "auto", "effort": "low"}


def test_reasoning_boolean_enable_and_mapping_disabled(profile_factory):
    request = codec(profile_factory("openai_responses")).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning=True
    )
    assert request["reasoning"]["effort"] == "medium"
    request = codec(
        profile_factory(
            "openai_responses", reasoning_enable_parameter=omit, reasoning_effort_parameter=omit
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high")
    assert "reasoning" not in request


def test_malicious_reasoning_binding_cannot_disable_stream(profile_factory):
    profile = profile_factory(reasoning_enable_parameter={"extra_body": {"stream": False}})
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
        resolve_profiles(ProviderProfile(), ModelProfile(model=" "))


def test_default_response_summary_retained_with_partial_binding(profile_factory):
    request = codec(
        profile_factory(
            "openai_responses",
            reasoning_enable_parameter={"reasoning": {"effort": "medium"}},
            reasoning_effort_parameter={"reasoning": {}},
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high")
    assert request["reasoning"] == {"summary": "detailed", "effort": "high"}


def test_response_summary_can_be_explicitly_omitted(profile_factory):
    profile = profile_factory(
        "openai_responses",
        reasoning_enable_parameter={"reasoning": {"summary": omit}},
        reasoning_effort_parameter={"reasoning": {"summary": omit, "effort": "{effort}"}},
    )
    request = codec(profile).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning"] == {"effort": "high"}
