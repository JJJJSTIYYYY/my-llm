import pytest

from my_llm import ConfigurationError, ModelProfile, ProviderProfile, UnsupportedProviderError, omit
from my_llm.profile.utils.common import merge_dicts
from my_llm.profile.utils.resolution import client_parameters, resolve_profiles


def test_protocol_defaults(profile_factory):
    chat = profile_factory()
    responses = profile_factory("openai_responses")
    assert chat.get("supports_reasoning") is True
    assert chat.get("supports_function_calling") is False
    assert chat.get("supports_reasoning_effort") is False
    assert responses.get("supports_reasoning_effort") is True
    assert responses.get("reasoning_enable_parameter")["reasoning"]["summary"] == "detailed"
    assert chat.get("usage_mode") == responses.get("usage_mode") == "none"


def test_model_overrides_provider_and_dict_keys():
    provider = ProviderProfile(
        extra_body={"a": 1, "b": 2, "nested": {"a": 1}},
        extra_headers={"x": "one"},
        supports_reasoning_effort=True,
    )
    model = ModelProfile(
        model="demo",
        extra_body={"a": 3, "b": omit, "nested": {"b": 2}},
        extra_headers={"y": "two"},
        supports_reasoning_effort=False,
    )
    result = resolve_profiles(provider, model)
    assert result.get("extra_body") == {"a": 3, "nested": {"b": 2}}
    assert result.get("extra_headers") == {"x": "one", "y": "two"}
    assert result.get("supports_reasoning_effort") is False
    assert provider.extra_body["a"] == 1


def test_omit_disables_defaults_and_none_inherits(profile_factory):
    result = profile_factory(
        "openai_responses",
        reasoning_effort_parameter=omit,
        reasoning_content_path=omit,
        supports_reasoning=omit,
    )
    assert result.get("reasoning_effort_parameter", {}) == {}
    assert result.get("reasoning_content_path") is None
    assert result.get("supports_reasoning", False) is False
    assert merge_dicts({"a": 1}, omit, {"b": 2}) == {"b": 2}
    assert merge_dicts({"a": 1}, None) == {"a": 1}


def test_client_parameters_use_the_resolved_profile_and_omit_unset_values():
    provider = ProviderProfile(
        api_key="secret", base_url="https://example.test/v1", timeout=30, max_retries=2
    )
    resolved = resolve_profiles(provider, ModelProfile(model="demo"))
    assert client_parameters(resolved) == {
        "api_key": "secret",
        "base_url": "https://example.test/v1",
        "timeout": 30,
        "max_retries": 2,
    }
    without_timeout = resolve_profiles(ProviderProfile(timeout=omit), ModelProfile(model="demo"))
    assert "timeout" not in client_parameters(without_timeout)
    assert "secret" not in repr(provider)


def test_unknown_protocol():
    with pytest.raises(UnsupportedProviderError):
        resolve_profiles(ProviderProfile(protocol="unknown"), ModelProfile(model="demo"))


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_native_protocols_keep_profile_defaults(protocol):
    native = resolve_profiles(ProviderProfile(protocol=protocol), ModelProfile(model="demo"))
    original = resolve_profiles(ProviderProfile(), ModelProfile(model="demo"))
    assert native.protocol == protocol
    assert native.values == original.values


@pytest.mark.parametrize("disabled", [False, omit])
def test_upstream_stream_is_required(disabled):
    with pytest.raises(UnsupportedProviderError):
        resolve_profiles(ProviderProfile(supports_stream=disabled), ModelProfile(model="demo"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("supports_reasoning", "true"),
        ("max_tokens", 0),
        ("max_tokens", True),
        ("extra_body", []),
        ("content_path", ""),
    ],
)
def test_bad_model_configuration(field, value):
    with pytest.raises(ConfigurationError):
        resolve_profiles(ProviderProfile(), ModelProfile(model="demo", **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout", -1),
        ("max_retries", -1),
        ("api_key", ""),
        ("base_url", 4),
        ("usage_mode", "invalid"),
    ],
)
def test_bad_provider_configuration(field, value):
    with pytest.raises(ConfigurationError):
        resolve_profiles(ProviderProfile(**{field: value}), ModelProfile(model="demo"))


@pytest.mark.parametrize(
    "protocol,usage", [("openai_responses", "stream_options"), ("openai_chat", "native")]
)
def test_incompatible_usage_modes(protocol, usage):
    with pytest.raises(ConfigurationError):
        resolve_profiles(
            ProviderProfile(protocol=protocol, usage_mode=usage), ModelProfile(model="demo")
        )


def test_roles_are_fixed():
    assert ModelProfile.supported_roles == ("system", "user", "assistant", "tool")
    with pytest.raises(TypeError):
        ModelProfile(model="demo", supported_roles=["developer"])
