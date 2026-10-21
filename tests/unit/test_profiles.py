"""Protocol selection, profile layering, defaults and snapshot semantics."""

import pickle
from copy import copy, deepcopy

import pytest

from my_llm import (
    ConfigurationError,
    ModelProfile,
    Omit,
    OverridableProfile,
    ProviderProfile,
    UnsupportedProviderError,
    omit,
)
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import resolve_profiles
from tests.helpers.profiles import config, parameter


def test_protocol_defaults(profile_factory):
    chat = profile_factory()
    responses = profile_factory("openai_responses")
    assert chat.parameter("reasoning_enable").supports is True
    assert chat.get("supports_function_calling") is False
    assert chat.parameters["reasoning_effort"].supports is False
    assert responses.parameter("reasoning_effort").supports is True
    assert responses.get("reasoning_enable")["summary"] == "detailed"
    assert chat.get("usage_mode") == responses.get("usage_mode") == "none"


def test_shared_overrides_are_public():
    from my_llm.profile import OverridableProfile as ProfileExport
    from my_llm.profile.configuration import OverridableProfile as ConfigurationExport

    assert OverridableProfile is ProfileExport is ConfigurationExport
    assert isinstance(ProviderProfile(), OverridableProfile)
    assert isinstance(ModelProfile(model="demo"), OverridableProfile)


def test_model_overrides_provider_and_dict_keys():
    provider = ProviderProfile(
        protocol="openai_chat",
        extra_body=config({"a": 1, "b": 2, "nested": {"a": 1}}),
        extra_headers=config({"x": "one"}),
        reasoning_effort=parameter("reasoning_effort", supports=True, protocol="openai_chat"),
    )
    model = ModelProfile(
        model="demo",
        extra_body=config({"a": 3, "b": omit, "nested": {"b": 2}}),
        extra_headers=config({"y": "two"}),
        reasoning_effort=parameter("reasoning_effort", supports=False),
    )
    result = resolve_profiles(provider, model)
    assert result.get("extra_body") == {"a": 3, "nested": {"b": 2}}
    assert result.get("extra_headers") == {"x": "one", "y": "two"}
    assert result.parameters["reasoning_effort"].supports is False
    assert provider.extra_body.default_value["a"] == 1


def test_omit_disables_defaults_and_none_inherits(profile_factory):
    result = profile_factory(
        "openai_responses",
        reasoning_effort=omit,
        reasoning_content=omit,
        reasoning_enable=parameter("reasoning_enable", supports=False),
    )
    assert result.get("reasoning_effort", "fallback") == "fallback"
    assert result.parameter("reasoning_content") is None
    assert result.parameters["reasoning_enable"].supports is False
    assert merge_dicts({"a": 1}, omit, {"b": 2}) == {"b": 2}
    assert merge_dicts({"a": 1}, None) == {"a": 1}


@pytest.mark.parametrize("settings", [{}, {"protocol": None}, {"protocol": "unknown"}])
def test_provider_requires_a_registered_protocol(settings):
    provider = ProviderProfile(**settings)
    with pytest.raises(
        UnsupportedProviderError, match=f"unsupported protocol: {provider.protocol!r}"
    ):
        resolve_profiles(provider, ModelProfile(model="demo"))


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_native_protocols_keep_profile_defaults(protocol):
    native = resolve_profiles(ProviderProfile(protocol=protocol), ModelProfile(model="demo"))
    assert native.protocol == protocol
    assert native.parameter("reasoning_enable").path.startswith("$.")
    assert native.get("reasoning_effort", omit) is omit
    assert native.parameters["reasoning_effort"].supports is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("reasoning_enable", "true"),
        ("max_tokens", 0),
        ("max_tokens", True),
        ("extra_body", []),
        ("content", ""),
    ],
)
def test_bad_model_configuration(field, value):
    with pytest.raises(ConfigurationError):
        resolve_profiles(
            ProviderProfile(protocol="openai_chat"),
            ModelProfile(
                model="demo",
                **{
                    field: (
                        parameter(field, value)
                        if field == "max_tokens"
                        else parameter("reasoning_enable", supports=value)
                        if field == "reasoning_enable"
                        else parameter("content", path=value)
                        if field == "content"
                        else config(value)
                    )
                },
            ),
        )


def test_parameter_default_objects_replace_instead_of_merging():
    result = resolve_profiles(
        ProviderProfile(
            protocol="openai_chat",
            reasoning_enable=parameter(
                "reasoning_enable", {"type": "enabled", "budget": 1024}, protocol="openai_chat"
            ),
        ),
        ModelProfile(
            model="demo",
            reasoning_enable=parameter(
                "reasoning_enable", {"type": "adaptive"}, protocol="openai_chat"
            ),
        ),
    )
    assert result.get("reasoning_enable") == {"type": "adaptive"}


def test_omit_is_owned_by_my_llm_and_survives_snapshotting():
    from openai import Omit as OpenAIOmit

    from my_llm.profile import Omit as ProfileOmit

    assert ProfileOmit is Omit
    assert not isinstance(omit, OpenAIOmit)
    assert Omit() is omit
    assert copy(omit) is deepcopy(omit) is pickle.loads(pickle.dumps(omit)) is omit
    assert (
        resolve_profiles(
            ProviderProfile(protocol="openai_chat", max_tokens=omit), ModelProfile(model="demo")
        ).values["max_tokens"]
        is omit
    )


@pytest.mark.parametrize(
    "value", [None, omit, parameter("max_tokens", None), parameter("max_tokens", 2048)]
)
def test_profile_pickle_preserves_default_inheritance_and_explicit_values(value):
    provider = ProviderProfile(protocol="openai_chat", max_tokens=parameter("max_tokens", 4096))
    model = ModelProfile(model="demo", max_tokens=value)
    original = resolve_profiles(provider, model)
    restored_provider, restored_model = pickle.loads(pickle.dumps((provider, model)))
    assert restored_model.max_tokens == model.max_tokens
    assert resolve_profiles(restored_provider, restored_model) == original
