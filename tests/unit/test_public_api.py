import inspect

import pytest

import my_llm
from my_llm import ConfigurationError, ModelProfile, MyLLM, ProviderProfile, resolve_profiles


def test_public_contract_uses_standard_messages_and_one_protocol_field():
    for name in (
        "MyCodec",
        "BaseCodec",
        "BaseEncoder",
        "BaseDecoder",
        "BaseTransport",
        "MyLLM",
        "MessageStream",
        "ResolvedProfile",
        "OverridableProfile",
        "ProtocolDefaults",
        "PreparedOptions",
        "Parameter",
        "Config",
        "Omit",
        "omit",
    ):
        assert name in my_llm.__all__
    for name in (
        "Message",
        "MessageChunk",
        "MessageInput",
        "ToolCall",
        "FunctionCall",
        "APIType",
        "Codec",
        "Translations",
        "ParameterPath",
        "ParameterDefaultV",
        "UNSET",
    ):
        assert not hasattr(my_llm, name)
    assert "api_type" not in ProviderProfile.__dataclass_fields__
    assert "client" not in inspect.signature(MyLLM).parameters
    assert "provider" not in inspect.signature(MyLLM.generate).parameters
    assert "model" not in inspect.signature(MyLLM.generate).parameters


def test_removed_constructor_options_fail():
    with pytest.raises(TypeError):
        MyLLM(provider=ProviderProfile(), model=ModelProfile(model="demo"), client=object())
    with pytest.raises(TypeError):
        ProviderProfile(api_type="responses")


@pytest.mark.parametrize("profile_type", [ProviderProfile, ModelProfile])
@pytest.mark.parametrize("field", ["max_tokens_path", "content_path", "reasoning_content_path"])
def test_profiles_reject_separate_parameter_paths(profile_type, field):
    identity = {"model": "demo"} if profile_type is ModelProfile else {}
    with pytest.raises(TypeError, match=field):
        profile_type(**identity, **{field: "$.vendor.value"})


@pytest.mark.parametrize("layer", ["provider", "model"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 1024),
        ("reasoning_enable", {"type": "enabled"}),
        ("supports_reasoning_effort", True),
        ("extra_body", {"temperature": 0.5}),
    ],
)
def test_profiles_reject_unwrapped_configuration(layer, field, value):
    overrides = {field: value}
    provider = ProviderProfile(**overrides) if layer == "provider" else ProviderProfile()
    model = ModelProfile(model="demo", **(overrides if layer == "model" else {}))
    with pytest.raises(ConfigurationError, match=field):
        resolve_profiles(provider, model)
