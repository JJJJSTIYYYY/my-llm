"""Profile snapshots use declarative defaults; Codec coordinates protocol validation."""

from dataclasses import dataclass, field, replace

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    InputParameter,
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


def test_mutable_parameter_bindings_are_isolated_between_profile_snapshots():
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(
        model=Attribute(default_value="demo"),
        max_tokens=InputParameter(path="$.max_completion_tokens", default_value=128),
    )
    first = resolve_profiles(provider, model)
    second = resolve_profiles(provider, model)
    model.max_tokens.path = "$.changed"
    model.max_tokens.default_value = 256
    model.max_tokens.supports = False
    for snapshot in (first, second):
        binding = snapshot.parameter("max_tokens")
        assert binding.path == "$.max_completion_tokens"
        assert binding.default_value == snapshot.get("max_tokens") == 128
        assert dict(snapshot.request_parameters)["max_tokens"] is binding
    first.parameter("max_tokens").path = "$.first_only"
    assert second.parameter("max_tokens").path == "$.max_completion_tokens"


def test_mutable_profile_defaults_are_independent_between_instances():
    first_model = ModelProfile(model=Attribute(default_value="first"))
    second_model = ModelProfile(model=Attribute(default_value="second"))
    first_model.max_context_window.default_value = 1024
    assert second_model.max_context_window.default_value == 256000

    first_defaults = ProtocolDefaults(should_feedback_reasoning_item=Config(default_value="none"))
    second_defaults = ProtocolDefaults(should_feedback_reasoning_item=Config(default_value="none"))
    first_defaults.supports_stream.default_value = False
    first_defaults.api_key.path = "$.extra_headers.authorization"
    assert second_defaults.supports_stream.default_value is True
    assert second_defaults.api_key.path == "$.api_key"


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("model_override", [False, True])
def test_standard_defaults_are_preserved_in_independent_profile_snapshots(protocol, model_override):
    defaults = protocol_defaults(protocol)
    resolved = resolve_profiles(
        ProviderProfile(
            protocol=Config(default_value="openai_chat" if model_override else protocol)
        ),
        ModelProfile(
            model=Attribute(default_value="demo"),
            protocol=Config(default_value=protocol) if model_override else None,
        ),
    )
    assert resolved.protocol == resolved.get("protocol") == protocol
    assert all(
        (resolved.parameters[key] if isinstance(value, (Parameter, Omit)) else resolved.get(key))
        == (value if isinstance(value, (Parameter, Omit)) else value.default_value)
        for key, value in defaults.items()
    )
    defaults["reasoning_effort_mapping"].default_value["changed"] = "custom"
    assert "changed" not in resolved.get("reasoning_effort_mapping")
    assert "changed" not in protocol_defaults(protocol)["reasoning_effort_mapping"].default_value


def test_model_protocol_selects_the_instance_defaults_before_resolution():
    codec = MyCodec()
    calls = []

    def defaults_for(protocol):
        calls.append(protocol)
        return codec.defaults(protocol)

    resolved = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        ModelProfile(
            model=Attribute(default_value="demo"),
            protocol=Config(default_value="openai_responses"),
        ),
        defaults_for=defaults_for,
    )

    assert calls == ["openai_responses"]
    assert resolved.parameter("max_tokens").path == "$.max_output_tokens"


def test_llm_uses_its_own_codec_for_protocol_validation(monkeypatch):
    calls = []
    instances = []

    class Codec(MyCodec):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

        def validate_profile(self, profile):
            calls.append((self, profile))
            super().validate_profile(profile)

    monkeypatch.setattr("my_llm.my_llm.MyCodec", Codec)
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(model=Attribute(default_value="demo")),
    )
    assert instances == [llm.codec]
    assert calls == [(llm.codec, llm.resolved_profile)]


@pytest.mark.parametrize(
    "provider,model",
    [
        (None, ModelProfile(model=Attribute(default_value="demo"))),
        ({"protocol": "openai_chat"}, ModelProfile(model=Attribute(default_value="demo"))),
        (ProviderProfile(), None),
    ],
)
def test_llm_preserves_profile_type_errors_before_reading_defaults(provider, model):
    with pytest.raises(ConfigurationError, match="profile instances"):
        MyLLM(provider=provider, model=model)


@dataclass(kw_only=True)
class SamplingDefaults(ProtocolDefaults):
    temperature: InputParameter[float | None] | Omit = field(
        default_factory=lambda: InputParameter(path="$.extra_body.temperature", default_value=0.25),
    )
    custom_mode: Config[str] = field(default_factory=lambda: Config(default_value="standard"))


def test_inherited_protocol_defaults_are_discovered_without_a_central_table(monkeypatch):
    from my_llm.basic.protocols import codec_for

    codec = codec_for("openai_chat")
    inherited = protocol_defaults("openai_chat")
    monkeypatch.setattr(type(codec), "defaults", SamplingDefaults(**inherited))
    result = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        ModelProfile(model=Attribute(default_value="demo")),
    )
    assert result.get("temperature") == 0.25
    assert result.parameter("temperature").path == "$.extra_body.temperature"
    assert result.get("custom_mode") == "standard"
    defaults = protocol_defaults("openai_chat")
    defaults["extra_body"].default_value["changed"] = True
    assert "changed" not in result.get("extra_body")
    assert "changed" not in protocol_defaults("openai_chat")["extra_body"].default_value


def test_inherited_protocol_none_default_and_omit_are_preserved(monkeypatch):
    from my_llm.basic.protocols import codec_for

    codec = codec_for("openai_chat")
    defaults = SamplingDefaults(
        **protocol_defaults("openai_chat"),
        temperature=InputParameter(path="$.extra_body.temperature", default_value=None),
    )
    monkeypatch.setattr(type(codec), "defaults", defaults)
    result = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        ModelProfile(model=Attribute(default_value="demo")),
    )
    assert result.get("temperature", "missing") is None
    monkeypatch.setattr(type(codec), "defaults", replace(defaults, temperature=omit))
    result = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        ModelProfile(model=Attribute(default_value="demo")),
    )
    assert result.values["temperature"] is omit


def test_unclassified_protocol_default_is_rejected(monkeypatch):
    from my_llm.basic.protocols import codec_for

    @dataclass(kw_only=True)
    class BadDefaults(ProtocolDefaults):
        accidental: int = 1

    codec = codec_for("openai_chat")
    monkeypatch.setattr(type(codec), "defaults", BadDefaults(**protocol_defaults("openai_chat")))
    with pytest.raises(ConfigurationError, match="accidental"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            ModelProfile(model=Attribute(default_value="demo")),
        )
