import pickle
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, dataclass

import pytest

from my_llm import (
    Config,
    ConfigurationError,
    ModelProfile,
    MyCodec,
    MyLLM,
    Omit,
    OverridableProfile,
    Parameter,
    ProviderProfile,
    UnsupportedProviderError,
    omit,
)
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import client_parameters, resolve_profiles
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS


def test_protocol_defaults(profile_factory):
    chat = profile_factory()
    responses = profile_factory("openai_responses")
    assert chat.get("supports_reasoning") is True
    assert chat.get("supports_function_calling") is False
    assert chat.get("supports_reasoning_effort") is False
    assert responses.get("supports_reasoning_effort") is True
    assert responses.get("reasoning_enable")["summary"] == "detailed"
    assert chat.get("usage_mode") == responses.get("usage_mode") == "none"


def test_shared_overrides_are_public():
    from my_llm.profile import OverridableProfile as ProfileExport
    from my_llm.profile.configuration import OverridableProfile as ConfigurationExport

    assert OverridableProfile is ProfileExport is ConfigurationExport
    assert isinstance(ProviderProfile(), OverridableProfile)
    assert isinstance(ModelProfile(model="demo"), OverridableProfile)


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
            max_tokens=parameter(
                "max_tokens", protocol="openai_chat", path="$.extra_body.max_tokens"
            )
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
            ProviderProfile(max_tokens=parameter("max_tokens", protocol="openai_chat", path=path)),
            ModelProfile(model="demo"),
        )


def test_model_overrides_provider_and_dict_keys():
    provider = ProviderProfile(
        extra_body=config("extra_body", {"a": 1, "b": 2, "nested": {"a": 1}}),
        extra_headers=config("extra_headers", {"x": "one"}),
        supports_reasoning_effort=config("supports_reasoning_effort", True),
    )
    model = ModelProfile(
        model="demo",
        extra_body=config("extra_body", {"a": 3, "b": omit, "nested": {"b": 2}}),
        extra_headers=config("extra_headers", {"y": "two"}),
        supports_reasoning_effort=config("supports_reasoning_effort", False),
    )
    result = resolve_profiles(provider, model)
    assert result.get("extra_body") == {"a": 3, "nested": {"b": 2}}
    assert result.get("extra_headers") == {"x": "one", "y": "two"}
    assert result.get("supports_reasoning_effort") is False
    assert provider.extra_body.default_value["a"] == 1


def test_omit_disables_defaults_and_none_inherits(profile_factory):
    result = profile_factory(
        "openai_responses",
        reasoning_effort=omit,
        reasoning_content=omit,
        supports_reasoning=omit,
    )
    assert result.get("reasoning_effort", "fallback") == "fallback"
    assert result.parameter("reasoning_content") is None
    assert result.get("supports_reasoning", False) is False
    assert merge_dicts({"a": 1}, omit, {"b": 2}) == {"b": 2}
    assert merge_dicts({"a": 1}, None) == {"a": 1}


def test_client_parameters_use_the_resolved_profile_and_omit_unset_values():
    provider = ProviderProfile(
        api_key=parameter("api_key", "secret"),
        base_url=config("base_url", "https://example.test/v1"),
        timeout=config("timeout", 30),
        max_retries=config("max_retries", 2),
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


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("api_key", [omit, None, parameter("api_key", omit)])
def test_omitted_api_key_preserves_header_configuration(protocol, api_key):
    headers = {"Authorization": "Bearer local-test"}
    profile = resolve_profiles(
        ProviderProfile(
            protocol=protocol,
            api_key=api_key,
            extra_headers=config("extra_headers", headers),
        ),
        ModelProfile(model="demo"),
    )
    assert "api_key" not in client_parameters(profile)
    assert profile.get("extra_headers") == headers


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_api_key_default_value_is_omit(protocol):
    profile = resolve_profiles(ProviderProfile(protocol=protocol), ModelProfile(model="demo"))
    assert profile.parameter("api_key").default_value is omit
    assert "api_key" not in client_parameters(profile)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("header", ["Authorization", "x-api-key"])
def test_api_key_header_path_uses_the_official_sdk_header_option(monkeypatch, protocol, header):
    captured = []
    client = object()

    def factory(**kwargs):
        captured.append(kwargs)
        return client

    target = {
        "openai_chat": "my_llm.codec.openai.chat.transport.AsyncOpenAI",
        "openai_responses": "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        "anthropic": "my_llm.codec.anthropic.transport.AsyncAnthropic",
        "gemini": "my_llm.codec.gemini.transport.genai.Client",
        "ollama": "my_llm.codec.ollama.transport.AsyncOllama",
    }[protocol]
    monkeypatch.setattr(target, factory)
    llm = MyLLM(
        provider=ProviderProfile(
            protocol=protocol,
            api_key=parameter("api_key", "local-header-key", path=f"$.extra_headers.{header}"),
        ),
        model=ModelProfile(model="demo"),
    )
    assert llm.codec.transport(protocol).create_client(llm.resolved_profile) is client
    options = captured[0]
    assert "api_key" not in options
    if protocol == "gemini":
        headers = options["http_options"].headers
    else:
        headers = options["headers" if protocol == "ollama" else "default_headers"]
    assert headers == {header: "local-header-key"}


@pytest.mark.parametrize(
    "path",
    ["$.extra_body.api_key", "$.extra_headers", "$.extra_headers[0]", "$.api_key.nested"],
)
def test_api_key_rejects_unsupported_connection_paths(path):
    with pytest.raises(ConfigurationError, match="api_key path"):
        MyLLM(
            provider=ProviderProfile(api_key=parameter("api_key", "local", path=path)),
            model=ModelProfile(model="demo"),
        )


def test_api_key_rejects_encode_phase():
    with pytest.raises(ConfigurationError, match="api_key must use the decode phase"):
        MyLLM(
            provider=ProviderProfile(api_key=parameter("api_key", "local", phase="encode")),
            model=ModelProfile(model="demo"),
        )


def test_ollama_rejects_an_sdk_api_key_argument():
    with pytest.raises(ConfigurationError, match="Ollama api_key"):
        MyLLM(
            provider=ProviderProfile(protocol="ollama", api_key=parameter("api_key", "local")),
            model=ModelProfile(model="demo"),
        )


def test_unknown_protocol():
    with pytest.raises(UnsupportedProviderError):
        resolve_profiles(ProviderProfile(protocol="unknown"), ModelProfile(model="demo"))


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_native_protocols_keep_profile_defaults(protocol):
    native = resolve_profiles(ProviderProfile(protocol=protocol), ModelProfile(model="demo"))
    assert native.protocol == protocol
    assert native.parameter("reasoning_enable").path.startswith("$.")
    assert native.get("reasoning_effort", omit) is omit
    assert native.get("supports_reasoning_effort") is False


@pytest.mark.parametrize("disabled", [False, omit])
def test_upstream_stream_is_required(disabled):
    with pytest.raises(UnsupportedProviderError):
        resolve_profiles(
            ProviderProfile(supports_stream=config("supports_stream", disabled)),
            ModelProfile(model="demo"),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("supports_reasoning", "true"),
        ("max_tokens", 0),
        ("max_tokens", True),
        ("extra_body", []),
        ("content", ""),
    ],
)
def test_bad_model_configuration(field, value):
    with pytest.raises(ConfigurationError):
        resolve_profiles(
            ProviderProfile(),
            ModelProfile(
                model="demo",
                **{
                    field: (
                        parameter(field, value)
                        if field == "max_tokens"
                        else parameter("content", path=value)
                        if field == "content"
                        else config(field, value)
                    )
                },
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout", -1),
        ("max_retries", -1),
        ("api_key", ""),
        ("base_url", 4),
        ("usage_mode", ""),
        ("usage_mode", 1),
    ],
)
def test_bad_provider_configuration(field, value):
    with pytest.raises(ConfigurationError):
        resolve_profiles(
            ProviderProfile(
                **{field: parameter(field, value) if field == "api_key" else config(field, value)}
            ),
            ModelProfile(model="demo"),
        )


@pytest.mark.parametrize(
    "protocol,usage", [("openai_responses", "stream_options"), ("openai_chat", "native")]
)
def test_incompatible_usage_modes(protocol, usage):
    with pytest.raises(ConfigurationError):
        MyLLM(
            provider=ProviderProfile(protocol=protocol, usage_mode=config("usage_mode", usage)),
            model=ModelProfile(model="demo"),
        )


def test_unrecognized_usage_mode_is_rejected_by_selected_transport():
    with pytest.raises(ConfigurationError, match="usage_mode"):
        MyLLM(
            provider=ProviderProfile(usage_mode=config("usage_mode", "invalid")),
            model=ModelProfile(model="demo"),
        )


def test_roles_are_fixed():
    assert ModelProfile.supported_roles == ("system", "user", "assistant", "tool")
    with pytest.raises(TypeError):
        ModelProfile(model="demo", supported_roles=["developer"])


@dataclass(frozen=True, kw_only=True)
class SamplingProvider(ProviderProfile):
    temperature: Parameter[float | None] | Omit | None = Parameter(
        name="Temperature", path="$.extra_body.temperature", default_value=0.5, phase="decode"
    )
    sampling_enabled: Config[bool] | Omit | None = config("Sampling", True)


@dataclass(frozen=True, kw_only=True)
class SamplingModel(ModelProfile):
    temperature: Parameter[float | None] | Omit | None = None
    sampling_enabled: Config[bool] | Omit | None = None


def test_parameter_defaults_distinguish_inheritance_none_and_omit():
    provider = SamplingProvider()
    inherited = resolve_profiles(provider, SamplingModel(model="demo"))
    assert inherited.get("temperature") == 0.5
    assert inherited.parameter("temperature").path == "$.extra_body.temperature"
    assert inherited.get("sampling_enabled") is True
    explicit_none = resolve_profiles(
        provider,
        SamplingModel(
            model="demo",
            temperature=Parameter(
                name="Temperature",
                path="$.extra_body.temperature",
                default_value=None,
                phase="decode",
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
        SamplingProvider(sampling_enabled=config("sampling_enabled", False)),
        SamplingModel(
            model="demo",
            temperature=Parameter(
                name="New label",
                path="$.extra_body.vendor.temperature",
                default_value=None,
                phase="decode",
            ),
            sampling_enabled=None,
        ),
    )
    assert result.parameter("temperature").path == "$.extra_body.vendor.temperature"
    assert result.get("sampling_enabled") is False
    disabled = resolve_profiles(SamplingProvider(), SamplingModel(model="demo", temperature=omit))
    assert disabled.parameter("temperature") is None


def test_max_tokens_default_is_omitted_and_explicit_none_is_a_value():
    profile = resolve_profiles(ProviderProfile(), ModelProfile(model="demo"))
    assert profile.get("max_tokens", omit) is omit
    assert profile.parameter("max_tokens").default_value is omit
    inherited = resolve_profiles(
        ProviderProfile(max_tokens=parameter("max_tokens", 2048, protocol="openai_chat")),
        ModelProfile(model="demo"),
    )
    assert inherited.get("max_tokens") == 2048
    result = resolve_profiles(
        ProviderProfile(max_tokens=parameter("max_tokens", 2048, protocol="openai_chat")),
        ModelProfile(model="demo", max_tokens=parameter("max_tokens", None)),
    )
    assert result.get("max_tokens", 100) is None


def test_parameter_default_objects_replace_instead_of_merging():
    result = resolve_profiles(
        ProviderProfile(
            reasoning_enable=parameter(
                "reasoning_enable", {"type": "enabled", "budget": 1024}, protocol="openai_chat"
            )
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
        resolve_profiles(ProviderProfile(max_tokens=omit), ModelProfile(model="demo")).values[
            "max_tokens"
        ]
        is omit
    )


@pytest.mark.parametrize(
    "value", [None, omit, parameter("max_tokens", None), parameter("max_tokens", 2048)]
)
def test_profile_pickle_preserves_default_inheritance_and_explicit_values(value):
    provider = ProviderProfile(max_tokens=parameter("max_tokens", 4096))
    model = ModelProfile(model="demo", max_tokens=value)
    original = resolve_profiles(provider, model)
    restored_provider, restored_model = pickle.loads(pickle.dumps((provider, model)))
    assert restored_model.max_tokens == value
    assert resolve_profiles(restored_provider, restored_model) == original


@pytest.mark.parametrize(
    "profile_class,value",
    [(SamplingProvider, "warm"), (SamplingProvider, True)],
)
def test_custom_default_types_are_checked(profile_class, value):
    with pytest.raises(ConfigurationError, match="temperature"):
        resolve_profiles(
            profile_class(
                temperature=Parameter(
                    name="Temperature",
                    path="$.extra_body.temperature",
                    default_value=value,
                    phase="decode",
                )
            ),
            ModelProfile(model="demo"),
        )


def test_custom_config_type_is_checked():
    with pytest.raises(ConfigurationError, match="sampling_enabled"):
        resolve_profiles(
            SamplingProvider(sampling_enabled=config("sampling_enabled", "yes")),
            ModelProfile(model="demo"),
        )


def test_unclassified_extension_is_rejected():
    @dataclass(frozen=True, kw_only=True)
    class Unclassified(ProviderProfile):
        accidental: str = "not-a-declaration"

    with pytest.raises(ConfigurationError, match="accidental.*Parameter"):
        resolve_profiles(Unclassified(), ModelProfile(model="demo"))


def test_parameter_binds_path_default_and_phase_without_a_paired_field():
    @dataclass(frozen=True, kw_only=True)
    class Extension(ModelProfile):
        arbitrary: Parameter[str] = Parameter(
            name="Visible label",
            path="$.extra_body.vendor",
            default_value="fallback",
            phase="decode",
        )

    resolved = resolve_profiles(ProviderProfile(), Extension(model="demo"))
    assert resolved.parameter("arbitrary").path == "$.extra_body.vendor"
    assert resolved.get("arbitrary") == "fallback"


def test_declarations_are_immutable_and_keyword_only():
    declaration = Parameter(name="Value", path="$.value", default_value=None, phase="encode")
    with pytest.raises(FrozenInstanceError):
        declaration.path = "$.other"
    with pytest.raises(TypeError):
        Parameter("Value", "$.value", None, "encode")
    with pytest.raises(TypeError):
        Config("Flag", True)
    with pytest.raises(FrozenInstanceError):
        Config(name="Flag", default_value=True).name = "Other"


def test_profile_marker_cannot_change_between_layers():
    @dataclass(frozen=True, kw_only=True)
    class ConflictingModel(ModelProfile):
        max_tokens: Config[int] = config("Limit", 2048)

    with pytest.raises(ConfigurationError, match="same profile marker"):
        resolve_profiles(ProviderProfile(), ConflictingModel(model="demo"))


def test_a_field_cannot_declare_multiple_responsibilities():
    @dataclass(frozen=True, kw_only=True)
    class AmbiguousProvider(ProviderProfile):
        feature: Config[bool] | Parameter[bool] = config("Feature", True)

    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_profiles(AmbiguousProvider(), ModelProfile(model="demo"))


def test_profile_marker_must_describe_the_field_not_a_contained_item():
    @dataclass(frozen=True, kw_only=True)
    class NestedMarker(ProviderProfile):
        feature: list[Config[bool]] = None

    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_profiles(NestedMarker(), ModelProfile(model="demo"))


def test_a_field_cannot_nest_different_profile_markers():
    @dataclass(frozen=True, kw_only=True)
    class NestedResponsibilities(ProviderProfile):
        feature: Config[Parameter[bool]] = config("Feature", True)

    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_profiles(NestedResponsibilities(), ModelProfile(model="demo"))


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
            temperature=Parameter(
                name="Temperature",
                path="$.extra_body.temperature",
                default_value=configured,
                phase="decode",
            ),
            extra_body=config("extra_body", body),
        ),
        ModelProfile(model="demo"),
    )
    request = codec.request(intermediate, profile)
    assert request.get("extra_body", {}) == expected


def test_config_wrapper_none_is_a_value_and_field_none_inherits():
    provider = SamplingProvider(sampling_enabled=config("Anything", True))
    inherited = resolve_profiles(provider, SamplingModel(model="demo", sampling_enabled=None))
    explicit_none = resolve_profiles(
        provider,
        SamplingModel(model="demo", sampling_enabled=Config(name="Anything", default_value=None)),
    )
    assert inherited.get("sampling_enabled") is True
    assert explicit_none.get("sampling_enabled", "missing") is None


def test_parameter_override_replaces_path_phase_and_default_as_one_binding():
    replacement = Parameter(
        name="Different display name",
        path="$.vendor.result",
        phase="encode",
        default_value=None,
    )
    resolved = resolve_profiles(
        SamplingProvider(), SamplingModel(model="demo", temperature=replacement)
    )
    assert resolved.parameter("temperature", "decode") is None
    assert resolved.parameter("temperature", "encode") == replacement
    assert resolved.get("temperature", "missing") is None
