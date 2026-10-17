from copy import deepcopy

import pytest

from my_llm import (
    BaseCodec,
    BaseDecoder,
    BaseEncoder,
    BaseTransport,
    MyCodec,
    UnsupportedCapabilityError,
)
from my_llm.codec.openai.chat.codec import ChatCompletionsCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from tests.helpers.protocols import PROTOCOLS, history, tools


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_common_contract_is_stateless_and_symmetric(protocol):
    codec = MyCodec().codecs[protocol]
    assert isinstance(codec, BaseCodec)
    assert isinstance(codec.encoder, BaseEncoder)
    assert isinstance(codec.decoder, BaseDecoder)
    assert isinstance(codec.transport, BaseTransport)
    for obj in (codec, codec.encoder, codec.decoder, codec.transport):
        assert not hasattr(obj, "__dict__")
        assert not hasattr(obj, "profile")
    assert codec.decode_messages(codec.encode_messages(history(protocol))) == history(protocol)
    assert codec.decode_tools(codec.encode_tools(tools(protocol))) == tools(protocol)
    assert codec.match_protocol(history(protocol)) == protocol
    assert codec.match_protocol(tools(protocol)) == protocol


def test_abstract_contract_cannot_be_instantiated():
    for cls in (BaseCodec, BaseEncoder, BaseDecoder, BaseTransport):
        with pytest.raises(TypeError):
            cls()


@pytest.mark.parametrize("missing", ["encoder", "decoder", "transport"])
def test_codec_requires_conversion_directions_and_transport(missing):
    bindings = {
        "protocol_or_api_style": "openai_chat",
        "encoder": ChatCompletionsEncoder(),
        "decoder": ChatCompletionsDecoder(),
        "transport": ChatCompletionsTransport(),
    }
    del bindings[missing]
    incomplete = type("IncompleteCodec", (BaseCodec,), bindings)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize(
    "base,concrete,missing",
    [
        (BaseEncoder, ChatCompletionsEncoder, "match_item"),
        (BaseEncoder, ChatCompletionsEncoder, "chunks"),
        (BaseDecoder, ChatCompletionsDecoder, "assemble_request"),
        (BaseDecoder, ChatCompletionsDecoder, "request_parameters"),
        (BaseTransport, ChatCompletionsTransport, "upstream"),
        (BaseTransport, ChatCompletionsTransport, "create_client"),
    ],
)
def test_partial_protocol_implementation_cannot_be_instantiated(base, concrete, missing):
    bindings = {
        name: getattr(concrete, name) for name in base.__abstractmethods__ if name != missing
    }
    incomplete = type("IncompleteImplementation", (base,), bindings)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_decoder_has_no_sdk_lifecycle_interface(protocol):
    converter = MyCodec()
    decoder = converter.decoder(protocol)
    transport = converter.transport(protocol)
    assert transport is converter.codecs[protocol].transport
    for name in ("create_client", "close_client", "client_key", "upstream"):
        assert not hasattr(decoder, name)
        assert callable(getattr(transport, name))


def test_codec_coordinates_conversion_and_transport_validation(profile_factory):
    profile = profile_factory()
    calls = []

    class Decoder(ChatCompletionsDecoder):
        __slots__ = ()

        def validate_profile(self, value):
            calls.append(("decoder", value))

    class Transport(ChatCompletionsTransport):
        __slots__ = ()

        def validate_profile(self, value):
            calls.append(("transport", value))

    class Codec(ChatCompletionsCodec):
        __slots__ = ()
        decoder = Decoder()
        transport = Transport()

    Codec().validate_profile(profile)
    assert calls == [("decoder", profile), ("transport", profile)]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_decoder_request_accepts_intermediate_values(profile_factory, protocol):
    converter = MyCodec()
    profile = profile_factory(protocol, max_tokens=100, supports_function_calling=True)
    intermediate = converter.encode(
        messages=[{"role": "user", "parts": [{"text": "hello"}]}],
        tools=tools("gemini"),
        input_format="gemini",
    )
    before = deepcopy(intermediate)
    request = converter.decoder(protocol).request(**intermediate, profile=profile, max_tokens=50)
    if protocol == "gemini":
        assert request["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        assert request["config"].max_output_tokens == 50
        assert request["config"].tools[0].function_declarations[0].name == "weather"
    else:
        history_key = "input" if protocol == "openai_responses" else "messages"
        expected_content = {
            "openai_chat": "hello",
            "openai_responses": [{"type": "input_text", "text": "hello"}],
            "anthropic": [{"type": "text", "text": "hello"}],
            "ollama": "hello",
        }[protocol]
        expected_message = {"role": "user", "content": expected_content}
        if protocol == "openai_responses":
            expected_message["type"] = "message"
        assert request[history_key] == [expected_message]
        assert request["tools"] == converter.decoder(protocol).tools(intermediate["tools"], profile)
    assert intermediate == before


@pytest.mark.parametrize("tool_values", [None, [], tools("openai_chat")])
def test_request_pipeline_decodes_once_before_native_assembly(profile_factory, tool_values):
    calls = []
    profile = profile_factory(supports_function_calling=True)
    intermediate = MyCodec().encode(
        messages=[{"role": "user", "content": "hello"}],
        tools=tool_values,
        input_format="openai_chat",
    )

    class RecordingDecoder(ChatCompletionsDecoder):
        __slots__ = ()

        def messages(self, messages, profile=None, **parameters):
            calls.append(("messages", profile, parameters))
            return super().messages(messages, profile, **parameters)

        def tools(self, tools, profile=None, **parameters):
            calls.append(("tools", profile, parameters))
            return super().tools(tools, profile, **parameters)

        def assemble_request(self, messages, tools, profile, **parameters):
            calls.append(("assemble", profile, parameters))
            assert messages == [{"role": "user", "content": "hello"}]
            assert tools == tool_values
            return super().assemble_request(messages, tools, profile, **parameters)

    RecordingDecoder().request(
        intermediate["messages"], intermediate.get("tools"), profile, max_tokens=42
    )
    expected = (
        ["messages", "assemble"] if tool_values is None else ["messages", "tools", "assemble"]
    )
    assert [kind for kind, _, _ in calls] == expected
    assert all(actual is profile and options == {"max_tokens": 42} for _, actual, options in calls)


def test_direct_decoder_request_checks_tool_capability_before_conversion(profile_factory):
    class RejectConversion(ChatCompletionsDecoder):
        __slots__ = ()

        def messages(self, *args, **kwargs):
            pytest.fail("unsupported tools must be rejected before converting the request")

    with pytest.raises(UnsupportedCapabilityError, match="function calling"):
        RejectConversion().request([], [{}], profile_factory())


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
