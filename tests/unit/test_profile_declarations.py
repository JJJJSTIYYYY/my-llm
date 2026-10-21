"""Typed profile extensions, declaration markers and immutable wrappers."""

from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from my_llm import (
    Config,
    ConfigurationError,
    ModelProfile,
    MyCodec,
    Omit,
    Parameter,
    ProviderProfile,
    omit,
)
from my_llm.profile.resolution import resolve_profiles
from tests.helpers.profiles import config


@dataclass(frozen=True, kw_only=True)
class SamplingProvider(ProviderProfile):
    temperature: Parameter[float | None] | Omit | None = field(
        default=Parameter(path="$.extra_body.temperature", default_value=0.5),
        metadata={"source": "input"},
    )
    sampling_enabled: Config[bool] | Omit | None = config(True)


@dataclass(frozen=True, kw_only=True)
class SamplingModel(ModelProfile):
    temperature: Parameter[float | None] | Omit | None = field(
        default=None, metadata={"source": "input"}
    )
    sampling_enabled: Config[bool] | Omit | None = None


def test_parameter_defaults_distinguish_inheritance_none_and_omit():
    provider = SamplingProvider(protocol="openai_chat")
    inherited = resolve_profiles(provider, SamplingModel(model="demo"))
    assert inherited.get("temperature") == 0.5
    assert inherited.parameter("temperature").path == "$.extra_body.temperature"
    assert inherited.get("sampling_enabled") is True
    explicit_none = resolve_profiles(
        provider,
        SamplingModel(
            model="demo",
            temperature=Parameter(
                path="$.extra_body.temperature",
                default_value=None,
            ),
        ),
    )
    assert explicit_none.get("temperature", "missing") is None
    assert explicit_none.values["temperature"] is None
    removed = resolve_profiles(provider, SamplingModel(model="demo", temperature=omit))
    assert removed.values["temperature"] is omit
    assert removed.get("temperature", "missing") == "missing"


def test_parameter_path_and_config_layering_in_custom_profiles():
    result = resolve_profiles(
        SamplingProvider(protocol="openai_chat", sampling_enabled=config(False)),
        SamplingModel(
            model="demo",
            temperature=Parameter(
                path="$.extra_body.vendor.temperature",
                default_value=None,
            ),
            sampling_enabled=None,
        ),
    )
    assert result.parameter("temperature").path == "$.extra_body.vendor.temperature"
    assert result.get("sampling_enabled") is False
    disabled = resolve_profiles(
        SamplingProvider(protocol="openai_chat"), SamplingModel(model="demo", temperature=omit)
    )
    assert disabled.parameter("temperature") is None


@pytest.mark.parametrize(
    "profile_class,value",
    [(SamplingProvider, "warm"), (SamplingProvider, True)],
)
def test_custom_default_types_are_checked(profile_class, value):
    with pytest.raises(ConfigurationError, match="temperature"):
        resolve_profiles(
            profile_class(
                protocol="openai_chat",
                temperature=Parameter(
                    path="$.extra_body.temperature",
                    default_value=value,
                ),
            ),
            ModelProfile(model="demo"),
        )


def test_custom_config_type_is_checked():
    with pytest.raises(ConfigurationError, match="sampling_enabled"):
        resolve_profiles(
            SamplingProvider(protocol="openai_chat", sampling_enabled=config("yes")),
            ModelProfile(model="demo"),
        )


def test_unclassified_extension_is_rejected():
    @dataclass(frozen=True, kw_only=True)
    class Unclassified(ProviderProfile):
        accidental: str = "not-a-declaration"

    with pytest.raises(ConfigurationError, match="accidental.*Parameter"):
        resolve_profiles(Unclassified(protocol="openai_chat"), ModelProfile(model="demo"))


def test_parameter_binds_path_default_and_source_without_a_paired_field():
    @dataclass(frozen=True, kw_only=True)
    class Extension(ModelProfile):
        arbitrary: Parameter[str] = field(
            default=Parameter(
                path="$.extra_body.vendor",
                default_value="fallback",
            ),
            metadata={"source": "input"},
        )

    resolved = resolve_profiles(ProviderProfile(protocol="openai_chat"), Extension(model="demo"))
    assert resolved.parameter("arbitrary").path == "$.extra_body.vendor"
    assert resolved.get("arbitrary") == "fallback"


def test_declarations_are_immutable_and_keyword_only():
    declaration = Parameter(path="$.value", default_value=None)
    with pytest.raises(FrozenInstanceError):
        declaration.path = "$.other"
    with pytest.raises(TypeError):
        Parameter("Value", "$.value", None, "encode")
    with pytest.raises(TypeError):
        Config("Flag", True)
    with pytest.raises(FrozenInstanceError):
        Config(default_value=True).name = "Other"


def test_profile_marker_cannot_change_between_layers():
    @dataclass(frozen=True, kw_only=True)
    class ConflictingModel(ModelProfile):
        max_tokens: Config[int] = config(2048)

    with pytest.raises(ConfigurationError, match="same profile marker"):
        resolve_profiles(ProviderProfile(protocol="openai_chat"), ConflictingModel(model="demo"))


def test_a_field_cannot_declare_multiple_responsibilities():
    @dataclass(frozen=True, kw_only=True)
    class AmbiguousProvider(ProviderProfile):
        feature: Config[bool] | Parameter[bool] = field(
            default=config(True), metadata={"source": "input"}
        )

    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_profiles(AmbiguousProvider(protocol="openai_chat"), ModelProfile(model="demo"))


def test_profile_marker_must_describe_the_field_not_a_contained_item():
    @dataclass(frozen=True, kw_only=True)
    class NestedMarker(ProviderProfile):
        feature: list[Config[bool]] = None

    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_profiles(NestedMarker(protocol="openai_chat"), ModelProfile(model="demo"))


def test_a_field_cannot_nest_different_profile_markers():
    @dataclass(frozen=True, kw_only=True)
    class NestedResponsibilities(ProviderProfile):
        feature: Config[Parameter[bool]] = field(default=config(True), metadata={"source": "input"})

    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_profiles(NestedResponsibilities(protocol="openai_chat"), ModelProfile(model="demo"))


@pytest.mark.parametrize(
    "configured,body,expected",
    [
        (0.5, {}, {"temperature": 0.5}),
        (None, {}, {"temperature": None}),
        (0.5, {"temperature": None}, {"temperature": None}),
        (None, {"temperature": 0.7}, {"temperature": 0.7}),
        (omit, {"temperature": 0.7}, {"temperature": 0.7}),
    ],
)
def test_declared_extension_defaults_reach_filtered_sdk_request(configured, body, expected):
    codec = MyCodec()
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")
    profile = resolve_profiles(
        SamplingProvider(
            protocol="openai_chat",
            temperature=Parameter(
                path="$.extra_body.temperature",
                default_value=configured,
            ),
            extra_body=config(body),
        ),
        ModelProfile(model="demo"),
    )
    request = codec.request(intermediate, profile)
    assert request.get("extra_body", {}) == expected


def test_config_wrapper_none_is_a_value_and_field_none_inherits():
    provider = SamplingProvider(protocol="openai_chat", sampling_enabled=config(True))
    inherited = resolve_profiles(provider, SamplingModel(model="demo", sampling_enabled=None))
    explicit_none = resolve_profiles(
        provider,
        SamplingModel(model="demo", sampling_enabled=Config(default_value=None)),
    )
    assert inherited.get("sampling_enabled") is True
    assert explicit_none.get("sampling_enabled", "missing") is None


def test_parameter_override_replaces_path_and_default_without_changing_source():
    replacement = Parameter(
        path="$.vendor.result",
        default_value=None,
    )
    resolved = resolve_profiles(
        SamplingProvider(protocol="openai_chat"),
        SamplingModel(model="demo", temperature=replacement),
    )
    assert resolved.parameter("temperature", "output") is None
    assert resolved.parameter("temperature", "input").path == replacement.path
    assert resolved.parameter("temperature").source == "input"
    assert resolved.get("temperature", "missing") is None
