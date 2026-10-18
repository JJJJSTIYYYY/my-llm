import inspect

import pytest

import my_llm
from my_llm import ModelProfile, MyLLM, ProviderProfile


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
        "Translations",
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
