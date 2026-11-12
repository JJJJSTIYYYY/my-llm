"""Instance reuse identifiers, explicit resolution and cache lifecycle."""

from copy import deepcopy
from dataclasses import replace

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    InputParameter,
    ModelProfile,
    MyLLM,
    ProviderProfile,
)


def profiles():
    return ProviderProfile(protocol=Config(default_value="openai_chat")), ModelProfile(
        model=Attribute(default_value="demo")
    )


def test_resolve_profile_class_method_returns_independent_snapshot():
    provider, model = profiles()
    provider.extra_body = Config(default_value={"nested": [1]})
    provider.__post_init__()
    resolved = MyLLM.resolve_profile(provider, model)
    first = MyLLM(provider=provider, model=model, resolved_profile=resolved)
    second = MyLLM(provider=provider, model=model, resolved_profile=resolved)
    resolved.values["extra_body"]["nested"].append(2)
    first.resolved_profile.values["extra_body"]["nested"].append(3)
    assert second.resolved_profile.get("extra_body") == {"nested": [1]}
    assert provider.extra_body.default_value == {"nested": [1]}


def test_reused_snapshot_skips_resolution(monkeypatch):
    provider, model = profiles()
    resolved = MyLLM.resolve_profile(provider, model)

    def unexpected(*args, **kwargs):
        raise AssertionError("must not resolve again")

    monkeypatch.setattr("my_llm.my_llm.resolve_profiles", unexpected)
    llm = MyLLM(provider=provider, model=model, resolved_profile=resolved)
    assert llm.resolved_profile == resolved


@pytest.mark.parametrize(
    "values,expected",
    [
        (("openai_chat", "demo", "openai_chat"), True),
        (("other", "demo", "openai_chat"), False),
        (("openai_chat", "other", "openai_chat"), False),
        (("openai_chat", "demo", "anthropic"), False),
        (("OPENAI_CHAT", "demo", "openai_chat"), False),
    ],
)
def test_compare_matches_all_three_strings(values, expected):
    provider, model = profiles()
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    assert llm.compare(*values) is expected


def test_provider_identity_can_differ_from_protocol():
    provider, model = profiles()
    provider = replace(provider, provider=Attribute(default_value="vendor"))
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    assert llm.compare("vendor", "demo", "openai_chat") is True
    assert llm.compare("openai_chat", "demo", "openai_chat") is False
    assert "provider" not in llm.resolved_profile.values


@pytest.mark.parametrize("value", ["raw", Attribute(default_value=""), Attribute(default_value=1)])
def test_provider_identity_requires_a_nonempty_string_attribute(value):
    with pytest.raises(ConfigurationError, match="provider"):
        ProviderProfile(provider=value)


def test_initialization_does_not_require_serializable_configuration():
    provider, model = profiles()
    provider = replace(provider, extra_body=Config(default_value={"opaque": object()}))
    llm = MyLLM(provider=provider, model=model)
    assert "opaque" in llm.resolved_profile.get("extra_body")


def test_provider_identity_defaults_to_final_protocol_when_only_model_selects_it():
    provider = ProviderProfile()
    model = ModelProfile(
        model=Attribute(default_value="demo"), protocol=Config(default_value="anthropic")
    )
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    assert llm.provider.provider.default_value == "anthropic"
    assert llm.compare("anthropic", "demo", "openai_chat") is True


def test_compare_does_not_resolve_or_modify_state(monkeypatch):
    provider, model = profiles()
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    before = deepcopy(llm.resolved_profile)

    def unexpected(*args, **kwargs):
        raise AssertionError("compare must only compare identifiers")

    monkeypatch.setattr("my_llm.my_llm.resolve_profiles", unexpected)
    assert llm.compare("openai_chat", "demo", "openai_chat") is True
    assert llm.resolved_profile == before


def test_precomputed_snapshot_still_validates_protocol():
    provider, model = profiles()
    provider = replace(
        provider, max_tokens=InputParameter(path="$.unsupported_parameter", default_value=128)
    )
    resolved = MyLLM.resolve_profile(provider, model)
    with pytest.raises(ConfigurationError, match="max_tokens"):
        MyLLM(provider=provider, model=model, resolved_profile=resolved)


@pytest.mark.parametrize("value", [(), (None, 1), [], {}, True, object()])
def test_snapshot_requires_resolved_profile(value):
    provider, model = profiles()
    with pytest.raises(ConfigurationError, match="resolved_profile"):
        MyLLM(provider=provider, model=model, resolved_profile=value)


async def test_clear_cache_removes_only_instance_history_and_deduplication():
    provider, model = profiles()
    async with MyLLM(provider=provider, model=model, enable_message_cache=True) as llm:
        cache = llm.codec._message_cache
        cache.update(llm._message_cache_key, [{"text": "old"}], fingerprint=b"same")
        cache.update("other", [{"text": "keep"}])
        llm.clear_cache()
        llm.clear_cache()
        assert llm._message_cache_key not in cache
        assert cache.get("other") == [{"text": "keep"}]
        cache.update(llm._message_cache_key, [{"text": "new"}], fingerprint=b"same")
        assert cache.get(llm._message_cache_key) == [{"text": "new"}]
