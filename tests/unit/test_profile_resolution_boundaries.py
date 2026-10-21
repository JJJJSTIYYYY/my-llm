"""Profile snapshots use declarative defaults; Codec coordinates protocol validation."""

from dataclasses import dataclass, field, replace

import pytest

from my_llm import (
    Config,
    ConfigurationError,
    ModelProfile,
    MyCodec,
    MyLLM,
    Omit,
    Parameter,
    ProtocolDefaults,
    ProviderProfile,
    omit,
    resolve_profiles,
)
from my_llm.profile.defaults import protocol_defaults
from tests.helpers.protocols import PROTOCOLS


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_standard_defaults_are_preserved_in_independent_profile_snapshots(protocol):
    defaults = protocol_defaults(protocol)
    resolved = resolve_profiles(ProviderProfile(protocol=protocol), ModelProfile(model="demo"))
    assert all(
        (resolved.parameters[key] if isinstance(value, (Parameter, Omit)) else resolved.get(key))
        == (value if isinstance(value, (Parameter, Omit)) else value.default_value)
        for key, value in defaults.items()
    )
    defaults["reasoning_effort_mapping"].default_value["changed"] = "custom"
    assert "changed" not in resolved.get("reasoning_effort_mapping")
    assert "changed" not in protocol_defaults(protocol)["reasoning_effort_mapping"].default_value


def test_llm_uses_its_own_codec_for_protocol_validation(monkeypatch):
    calls = []
    instances = []

    class Codec(MyCodec):
        def __init__(self):
            super().__init__()
            instances.append(self)

        def validate_profile(self, profile):
            calls.append((self, profile))
            super().validate_profile(profile)

    monkeypatch.setattr("my_llm.invoke.MyCodec", Codec)
    llm = MyLLM(provider=ProviderProfile(protocol="openai_chat"), model=ModelProfile(model="demo"))
    assert instances == [llm.codec]
    assert calls == [(llm.codec, llm.resolved_profile)]


@pytest.mark.parametrize(
    "provider,model",
    [
        (None, ModelProfile(model="demo")),
        ({"protocol": "openai_chat"}, ModelProfile(model="demo")),
        (ProviderProfile(), None),
    ],
)
def test_llm_preserves_profile_type_errors_before_reading_defaults(provider, model):
    with pytest.raises(ConfigurationError, match="profile instances"):
        MyLLM(provider=provider, model=model)


@dataclass(frozen=True, kw_only=True)
class SamplingDefaults(ProtocolDefaults):
    temperature: Parameter[float | None] | Omit = field(
        default=Parameter(path="$.extra_body.temperature", default_value=0.25),
        metadata={"source": "input"},
    )
    custom_mode: Config[str] = Config(default_value="standard")


def test_inherited_protocol_defaults_are_discovered_without_a_central_table(monkeypatch):
    from my_llm.registry import codec_for

    codec = codec_for("openai_chat")
    inherited = protocol_defaults("openai_chat")
    monkeypatch.setattr(type(codec), "defaults", SamplingDefaults(**inherited))
    result = resolve_profiles(ProviderProfile(protocol="openai_chat"), ModelProfile(model="demo"))
    assert result.get("temperature") == 0.25
    assert result.parameter("temperature").path == "$.extra_body.temperature"
    assert result.get("custom_mode") == "standard"
    defaults = protocol_defaults("openai_chat")
    defaults["extra_body"].default_value["changed"] = True
    assert "changed" not in result.get("extra_body")
    assert "changed" not in protocol_defaults("openai_chat")["extra_body"].default_value


def test_inherited_protocol_none_default_and_omit_are_preserved(monkeypatch):
    from my_llm.registry import codec_for

    codec = codec_for("openai_chat")
    defaults = SamplingDefaults(
        **protocol_defaults("openai_chat"),
        temperature=Parameter(path="$.extra_body.temperature", default_value=None),
    )
    monkeypatch.setattr(type(codec), "defaults", defaults)
    result = resolve_profiles(ProviderProfile(protocol="openai_chat"), ModelProfile(model="demo"))
    assert result.get("temperature", "missing") is None
    monkeypatch.setattr(type(codec), "defaults", replace(defaults, temperature=omit))
    result = resolve_profiles(ProviderProfile(protocol="openai_chat"), ModelProfile(model="demo"))
    assert result.values["temperature"] is omit


def test_unclassified_protocol_default_is_rejected(monkeypatch):
    from my_llm.registry import codec_for

    @dataclass(frozen=True, kw_only=True)
    class BadDefaults(ProtocolDefaults):
        accidental: int = 1

    codec = codec_for("openai_chat")
    monkeypatch.setattr(type(codec), "defaults", BadDefaults(**protocol_defaults("openai_chat")))
    with pytest.raises(ConfigurationError, match="accidental"):
        resolve_profiles(ProviderProfile(protocol="openai_chat"), ModelProfile(model="demo"))
