"""Provider connection parameters and upstream capability validation."""

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    UnsupportedProviderError,
    omit,
)
from my_llm.profile.resolution import client_parameters, resolve_profiles
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS


def test_client_parameters_use_the_resolved_profile_and_omit_unset_values():
    provider = ProviderProfile(
        protocol=Config(default_value="openai_chat"),
        api_key=parameter("api_key", "secret"),
        base_url=config("https://example.test/v1"),
        timeout=config(30),
        max_retries=config(2),
    )
    resolved = resolve_profiles(provider, ModelProfile(model=Attribute(default_value="demo")))
    assert client_parameters(resolved) == {
        "api_key": "secret",
        "base_url": "https://example.test/v1",
        "timeout": 30,
        "max_retries": 2,
    }
    without_timeout = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat"), timeout=omit),
        ModelProfile(model=Attribute(default_value="demo")),
    )
    assert "timeout" not in client_parameters(without_timeout)
    assert "secret" not in repr(provider)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("api_key", [omit, None, parameter("api_key", omit)])
def test_omitted_api_key_preserves_header_configuration(protocol, api_key):
    headers = {"Authorization": "Bearer local-test"}
    profile = resolve_profiles(
        ProviderProfile(
            protocol=Config(default_value=protocol),
            api_key=api_key,
            extra_headers=config(headers),
        ),
        ModelProfile(model=Attribute(default_value="demo")),
    )
    assert "api_key" not in client_parameters(profile)
    assert profile.get("extra_headers") == headers


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_api_key_default_value_is_omit(protocol):
    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value=protocol)),
        ModelProfile(model=Attribute(default_value="demo")),
    )
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
            protocol=Config(default_value=protocol),
            api_key=parameter("api_key", "local-header-key", path=f"$.extra_headers.{header}"),
        ),
        model=ModelProfile(model=Attribute(default_value="demo")),
    )
    assert llm.codec.transport(protocol).create_client(llm.client_parameters) is client
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
            provider=ProviderProfile(
                protocol=Config(default_value="openai_chat"),
                api_key=parameter("api_key", "local", path=path),
            ),
            model=ModelProfile(model=Attribute(default_value="demo")),
        )


def test_api_key_source_is_fixed_in_profile():
    from my_llm import Parameter

    with pytest.raises(TypeError, match="source"):
        Parameter(path="$.api_key", default_value="local", source="output")
    provider = ProviderProfile(
        protocol=Config(default_value="openai_chat"), api_key=parameter("api_key", "local")
    )
    assert provider.api_key.source == "input"


def test_ollama_rejects_an_sdk_api_key_argument():
    with pytest.raises(ConfigurationError, match="Ollama api_key"):
        MyLLM(
            provider=ProviderProfile(
                protocol=Config(default_value="ollama"), api_key=parameter("api_key", "local")
            ),
            model=ModelProfile(model=Attribute(default_value="demo")),
        )


@pytest.mark.parametrize("disabled", [False, omit])
def test_upstream_stream_is_required(disabled):
    with pytest.raises(UnsupportedProviderError):
        resolve_profiles(
            ProviderProfile(
                protocol=Config(default_value="openai_chat"), supports_stream=config(disabled)
            ),
            ModelProfile(model=Attribute(default_value="demo")),
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
                protocol=Config(default_value="openai_chat"),
                **{field: parameter(field, value) if field == "api_key" else config(value)},
            ),
            ModelProfile(model=Attribute(default_value="demo")),
        )


@pytest.mark.parametrize(
    "protocol,usage", [("openai_responses", "stream_options"), ("openai_chat", "native")]
)
def test_incompatible_usage_modes(protocol, usage):
    with pytest.raises(ConfigurationError):
        MyLLM(
            provider=ProviderProfile(
                protocol=Config(default_value=protocol), usage_mode=config(usage)
            ),
            model=ModelProfile(model=Attribute(default_value="demo")),
        )


def test_unrecognized_usage_mode_is_rejected_by_selected_transport():
    with pytest.raises(ConfigurationError, match="usage_mode"):
        MyLLM(
            provider=ProviderProfile(
                protocol=Config(default_value="openai_chat"), usage_mode=config("invalid")
            ),
            model=ModelProfile(model=Attribute(default_value="demo")),
        )
