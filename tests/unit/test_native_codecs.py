import pytest
from conftest import PROTOCOLS, history, tools

from my_llm import BaseCodec, BaseDecoder, BaseEncoder, MyCodec


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_common_contract_is_stateless_and_symmetric(protocol):
    codec = MyCodec().codecs[protocol]
    assert isinstance(codec, BaseCodec)
    assert isinstance(codec.encoder, BaseEncoder)
    assert isinstance(codec.decoder, BaseDecoder)
    for obj in (codec, codec.encoder, codec.decoder):
        assert not hasattr(obj, "__dict__")
        assert not hasattr(obj, "profile")
    assert codec.decode_messages(codec.encode_messages(history(protocol))) == history(protocol)
    assert codec.decode_tools(codec.encode_tools(tools(protocol))) == tools(protocol)
    assert codec.match_protocol(history(protocol)) == protocol
    assert codec.match_protocol(tools(protocol)) == protocol


def test_abstract_contract_cannot_be_instantiated():
    for cls in (BaseCodec, BaseEncoder, BaseDecoder):
        with pytest.raises(TypeError):
            cls()


@pytest.mark.parametrize(
    "values",
    [
        [{"role": "assistant", "tool_calls": [None]}],
        [{"role": "assistant", "tool_calls": [{"function": None}]}],
        [{"role": "assistant", "tool_calls": "invalid"}],
        [{"role": "user", "content": [None]}],
    ],
)
def test_protocol_matching_rejects_malformed_nested_containers(values):
    converter = MyCodec()
    assert converter.match_protocol(values) is None
    assert all(codec.match_protocol(values) is None for codec in converter.codecs.values())


@pytest.mark.parametrize("protocol", ["openai_chat", "ollama"])
def test_nullable_native_calls_are_treated_as_absent_and_replayed_exactly(protocol):
    incoming = [{"role": "assistant", "content": None, "tool_calls": None}]
    if protocol == "ollama":
        incoming[0]["images"] = None
    converter = MyCodec()
    intermediate = converter.encode(incoming, input_format=protocol)
    assert intermediate["messages"][0]["content"] == []
    assert converter.decode(**intermediate, target_format=protocol)["messages"] == incoming
    assert converter.decode(**intermediate, target_format="anthropic")["messages"] == [
        {"role": "assistant", "content": []}
    ]
