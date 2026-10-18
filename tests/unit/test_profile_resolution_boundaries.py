"""Profile snapshots use declarative defaults; Codec coordinates protocol validation."""

import pytest

from my_llm import (
    ConfigurationError,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    resolve_profiles,
)
from my_llm.profile.defaults import protocol_defaults
from tests.helpers.protocols import PROTOCOLS


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_standard_defaults_are_preserved_in_independent_profile_snapshots(protocol):
    defaults = protocol_defaults(protocol)
    resolved = resolve_profiles(ProviderProfile(protocol=protocol), ModelProfile(model="demo"))
    assert all(resolved.values[key] == value for key, value in defaults.items())
    defaults["reasoning_enable_parameter"]["changed"] = True
    assert "changed" not in resolved.get("reasoning_enable_parameter")
    assert "changed" not in protocol_defaults(protocol)["reasoning_enable_parameter"]


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
    llm = MyLLM(provider=ProviderProfile(), model=ModelProfile(model="demo"))
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
