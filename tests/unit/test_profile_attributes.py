"""Profile-owned attributes stay separate from settings and parameter routing."""

from dataclasses import dataclass, field

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    InputParameter,
    ModelProfile,
    MyCodec,
    ProviderProfile,
    omit,
    resolve_profiles,
)


def test_attribute_is_public_mutable_and_keyword_only():
    from my_llm.profile import Attribute as ProfileAttribute

    assert Attribute is ProfileAttribute
    attribute = Attribute(default_value="display name", comment="metadata")
    attribute.default_value = "changed"
    assert attribute.default_value == "changed"
    with pytest.raises(TypeError):
        Attribute("display name")


def test_custom_attributes_are_validated_without_copying_or_routing_payloads():
    class Metadata:
        def __deepcopy__(self, memo):
            raise AssertionError("Attribute payloads must not enter configuration snapshots")

    @dataclass(kw_only=True)
    class DescriptiveProvider(ProviderProfile):
        description: Attribute[object] = field(
            default_factory=lambda: Attribute(default_value=Metadata())
        )

    @dataclass(kw_only=True)
    class DescriptiveModel(ModelProfile):
        description: Attribute[object] = field(
            default_factory=lambda: Attribute(default_value=Metadata())
        )

    profile = resolve_profiles(
        DescriptiveProvider(protocol=Config(default_value="openai_chat")),
        DescriptiveModel(model=Attribute(default_value="demo")),
    )
    assert {"description", "model", "models", "max_context_window"}.isdisjoint(profile.values)
    assert profile.model == "demo"
    assert profile.parameter("description") is None
    request = MyCodec().request(
        MyCodec().encode([{"role": "user", "content": "hi"}], input_format="openai_chat"),
        profile,
    )
    assert request["model"] == "demo"
    assert {"description", "models", "max_context_window", "protocol"}.isdisjoint(request)


@pytest.mark.parametrize("value", ["raw", Config(default_value="text"), Attribute(default_value=3)])
def test_custom_attribute_requires_its_declared_wrapper_and_payload_type(value):
    @dataclass(kw_only=True)
    class DescriptiveModel(ModelProfile):
        description: Attribute[str] = field(default_factory=lambda: value)

    with pytest.raises(ConfigurationError, match="description"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            DescriptiveModel(model=Attribute(default_value="demo")),
        )


def test_attribute_cannot_change_to_config_between_layers():
    @dataclass(kw_only=True)
    class DescriptiveProvider(ProviderProfile):
        description: Attribute[str] = field(
            default_factory=lambda: Attribute(default_value="metadata")
        )

    @dataclass(kw_only=True)
    class ConfiguredModel(ModelProfile):
        description: Config[str] = field(default_factory=lambda: Config(default_value="setting"))

    with pytest.raises(ConfigurationError, match="description.*same profile marker"):
        resolve_profiles(
            DescriptiveProvider(protocol=Config(default_value="openai_chat")),
            ConfiguredModel(model=Attribute(default_value="demo")),
        )


@pytest.mark.parametrize(
    "annotation",
    [Attribute[str] | Config[str], Attribute[str] | InputParameter[str], Config[Attribute[str]]],
)
def test_attribute_cannot_share_a_field_with_configuration_or_routing(annotation):
    @dataclass(kw_only=True)
    class AmbiguousModel(ModelProfile):
        description: annotation = field(default_factory=lambda: Attribute(default_value="metadata"))

    with pytest.raises(ConfigurationError, match="description.*exactly one"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            AmbiguousModel(model=Attribute(default_value="demo")),
        )


@pytest.mark.parametrize("layer", ["provider", "model"])
def test_protocol_is_a_bound_config_and_resolves_to_a_plain_protocol_name(layer):
    declaration = Config(default_value="openai_chat")
    provider = ProviderProfile(protocol=declaration if layer == "provider" else None)
    model = ModelProfile(
        model=Attribute(default_value="demo"),
        protocol=declaration if layer == "model" else None,
    )
    profile = resolve_profiles(provider, model)
    configured = provider if layer == "provider" else model
    assert configured.protocol.name == "protocol"
    assert declaration.name is None
    assert profile.protocol == profile.get("protocol") == "openai_chat"
    assert profile.parameter("protocol") is None


@pytest.mark.parametrize("layer", ["provider", "model"])
@pytest.mark.parametrize("protocol", ["openai_chat", Attribute(default_value="openai_chat"), omit])
def test_protocol_rejects_non_config_declarations(layer, protocol):
    with pytest.raises(ConfigurationError, match="protocol.*Config"):
        resolve_profiles(
            ProviderProfile(
                protocol=protocol if layer == "provider" else Config(default_value="openai_chat")
            ),
            ModelProfile(
                model=Attribute(default_value="demo"),
                protocol=protocol if layer == "model" else None,
            ),
        )


@pytest.mark.parametrize("model", ["demo", None, Config(default_value="demo")])
def test_model_requires_an_attribute_declaration(model):
    with pytest.raises(ConfigurationError, match="model"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")), ModelProfile(model=model)
        )


def test_model_subclass_protocol_default_overrides_provider_before_codec_selection():
    @dataclass(kw_only=True)
    class DifferentProtocol(ModelProfile):
        protocol: Config[str] = field(
            default_factory=lambda: Config(default_value="openai_responses")
        )

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        DifferentProtocol(model=Attribute(default_value="demo")),
    )

    assert profile.protocol == profile.get("protocol") == "openai_responses"
    assert profile.parameter("max_tokens").path == "$.max_output_tokens"
