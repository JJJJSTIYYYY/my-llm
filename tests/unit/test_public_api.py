import inspect

import pytest

import my_llm
from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    resolve_profiles,
)


def test_public_contract_uses_standard_messages_and_one_protocol_field():
    for name in (
        "ProtocolMiddleWare",
        "SemanticLossError",
        "SemanticLossItem",
        "SemanticLossGroup",
        "Message",
        "ToolCall",
        "Usage",
        "MyCodec",
        "MyLLM",
        "JSONGenerationError",
        "set__JSON_TOOL_PROMPT",
        "set__JSON_TEXT_PROMPT",
        "MessageStream",
        "ResolvedProfile",
        "OverridableProfile",
        "Profile",
        "ProtocolDefaults",
        "StreamBlock",
        "Parameter",
        "Config",
        "Attribute",
        "Omit",
        "omit",
    ):
        assert name in my_llm.__all__
    for name in (
        "MessageChunk",
        "MessageInput",
        "FunctionCall",
        "APIType",
        "Codec",
        "Translations",
        "ParameterPath",
        "ParameterDefaultV",
        "UNSET",
        "generate",
        "BaseCodec",
        "BaseEncoder",
        "BaseDecoder",
        "BaseTransport",
        "BaseSemanticLossAdaptor",
        "PreparedOptions",
        "StreamBatch",
    ):
        assert not hasattr(my_llm, name)
    assert "api_type" not in ProviderProfile.__dataclass_fields__
    assert "client" not in inspect.signature(MyLLM).parameters
    assert "provider" not in inspect.signature(MyLLM.generate).parameters
    assert "model" not in inspect.signature(MyLLM.generate).parameters


def test_package_facades_do_not_export_internal_operations():
    import my_llm.codec
    import my_llm.middleware
    import my_llm.runtime

    assert my_llm.codec.__all__ == ["MyCodec"]
    assert "generate" not in my_llm.runtime.__all__
    for name in (
        "StreamBatch",
        "wire_values",
        "generation_input",
        "validate_events",
        "validate_messages",
        "stream_from_wire",
        "messages_from_wire",
    ):
        assert not hasattr(my_llm.middleware, name)


def test_execution_and_translation_helpers_are_not_public_entry_points():
    for name in (
        "prepare",
        "request",
        "result",
        "decoder",
        "transport",
        "defaults",
        "validate_input",
        "validate_profile",
        "client_parameters",
        "bind_input",
        "upstream",
        "match_protocol",
    ):
        assert not hasattr(my_llm.MyCodec, name)
    for name in ("resolve_profile", "guess_stdio_protocol"):
        assert not hasattr(MyLLM, name)


def test_removed_constructor_options_fail():
    with pytest.raises(TypeError):
        MyLLM(
            provider=ProviderProfile(),
            model=ModelProfile(model=Attribute(default_value="demo")),
            client=object(),
        )
    with pytest.raises(TypeError):
        ProviderProfile(api_type="responses")


@pytest.mark.parametrize("profile_type", [ProviderProfile, ModelProfile])
@pytest.mark.parametrize("field", ["max_tokens_path", "content_path", "reasoning_content_path"])
def test_profiles_reject_separate_parameter_paths(profile_type, field):
    identity = {"model": Attribute(default_value="demo")} if profile_type is ModelProfile else {}
    with pytest.raises(TypeError, match=field):
        profile_type(**identity, **{field: "$.vendor.value"})


@pytest.mark.parametrize("layer", ["provider", "model"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 1024),
        ("reasoning_enable", {"type": "enabled"}),
        ("reasoning_effort", True),
        ("extra_body", {"temperature": 0.5}),
    ],
)
def test_profiles_reject_unwrapped_configuration(layer, field, value):
    overrides = {field: value}
    provider = (
        ProviderProfile(protocol=Config(default_value="openai_chat"), **overrides)
        if layer == "provider"
        else ProviderProfile(protocol=Config(default_value="openai_chat"))
    )
    model = ModelProfile(
        model=Attribute(default_value="demo"), **(overrides if layer == "model" else {})
    )
    with pytest.raises(ConfigurationError, match=field):
        resolve_profiles(provider, model)


def test_roles_are_fixed():
    assert ModelProfile.supported_roles == ("system", "user", "assistant", "tool")
    with pytest.raises(TypeError):
        ModelProfile(model=Attribute(default_value="demo"), supported_roles=["developer"])
