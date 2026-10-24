from copy import deepcopy

import pytest

from my_llm import (
    BaseCodec,
    BaseDecoder,
    BaseEncoder,
    BaseTransport,
    MessageValidationError,
    MyCodec,
    SemanticLossError,
)
from my_llm.codec.openai.chat.codec import ChatCompletionsCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from tests.helpers.profiles import config, parameter
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
    converted_messages = codec.decode_messages(codec.encode_messages(history(protocol)))
    converted_tools = codec.decode_tools(codec.encode_tools(tools(protocol)))
    assert type(converted_messages) is type(converted_tools) is list
    assert "Weather?" in str(converted_messages)
    assert "weather" in str(converted_messages)
    assert "Sunny" in str(converted_messages)
    assert "city" in str(converted_tools)
    assert codec.match_protocol(history(protocol)) == protocol
    assert codec.match_protocol(tools(protocol)) == protocol


def test_abstract_contract_cannot_be_instantiated():
    for cls in (BaseCodec, BaseEncoder, BaseDecoder, BaseTransport):
        with pytest.raises(TypeError):
            cls()


@pytest.mark.parametrize("missing", ["defaults", "encoder", "decoder", "transport"])
def test_codec_requires_conversion_directions_and_transport(missing):
    bindings = {
        "protocol_or_api_style": "openai_chat",
        "defaults": ChatCompletionsCodec().defaults,
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
        (BaseEncoder, ChatCompletionsEncoder, "_chunks"),
        (BaseDecoder, ChatCompletionsDecoder, "assemble_request"),
        (BaseDecoder, ChatCompletionsDecoder, "message_type"),
        (BaseDecoder, ChatCompletionsDecoder, "request_schema"),
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
    for name in ("create_client", "close_client", "upstream"):
        assert not hasattr(decoder, name)
        assert callable(getattr(transport, name))


def test_codec_delegates_configuration_validation_to_decoder(profile_factory):
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

        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
            self.transport = Transport()

    Codec().validate_profile(profile)
    assert calls == [("decoder", profile)]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_codec_request_translates_before_native_assembly(profile_factory, protocol):
    converter = MyCodec()
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 100, protocol=protocol),
        supports_function_calling=config(True),
    )
    intermediate = converter.encode(
        messages=[{"role": "user", "parts": [{"text": "hello"}]}],
        tools=tools("gemini"),
        input_format="gemini",
    )
    before = deepcopy(intermediate)
    request = converter.codecs[protocol].request(**intermediate, profile=profile, max_tokens=50)
    if protocol == "gemini":
        assert request["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        assert request["config"]["max_output_tokens"] == 50
        assert request["config"]["tools"][0]["function_declarations"][0]["name"] == "weather"
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
        assert request["tools"] == converter.decoder(protocol).filter_tools(
            converter.decoder(protocol).tools(intermediate["tools"], profile), profile
        )
    assert intermediate == before


@pytest.mark.parametrize("tool_values", [None, [], tools("openai_chat")])
def test_request_pipeline_decodes_once_before_native_assembly(profile_factory, tool_values):
    calls = []
    profile = profile_factory(supports_function_calling=config(True))
    intermediate = MyCodec().encode(
        messages=[{"role": "user", "content": "hello"}],
        tools=tool_values,
        input_format="openai_chat",
    )

    class RecordingDecoder(ChatCompletionsDecoder):
        __slots__ = ()

        def _messages(self, messages, profile=None, **parameters):
            calls.append(("messages", profile, parameters))
            return super()._messages(messages, profile, **parameters)

        def _tools(self, tools, profile=None, **parameters):
            calls.append(("tools", profile, parameters))
            return super()._tools(tools, profile, **parameters)

        def _request_parameters(self, request, profile, options):
            calls.append(("bindings", profile, {"max_tokens": options.tokens}))
            return super()._request_parameters(request, profile, options)

        def assemble_request(self, messages, tools, profile, *, request, **parameters):
            calls.append(("assemble", profile, parameters))
            assert messages == [{"role": "user", "content": "hello"}]
            assert tools == tool_values
            assert request["max_completion_tokens"] == 42
            return super().assemble_request(messages, tools, profile, request=request, **parameters)

        def validate_parameters(self, profile, **parameters):
            calls.append(("parameters", profile, parameters))
            return super().validate_parameters(profile, **parameters)

        def validate_request(self, request, profile, **parameters):
            calls.append(("validate", profile, parameters))
            return super().validate_request(request, profile, **parameters)

    class RecordingCodec(ChatCompletionsCodec):
        __slots__ = ()

        def __init__(self):
            super().__init__()
            self.decoder = RecordingDecoder()

    RecordingCodec().request(
        intermediate["messages"], intermediate.get("tools"), profile, max_tokens=42
    )
    expected = ["parameters", "messages"]
    if tool_values is not None:
        expected.append("tools")
    expected.extend(["bindings", "assemble", "validate"])
    assert [kind for kind, _, _ in calls] == expected
    assert all(actual is profile and options == {"max_tokens": 42} for _, actual, options in calls)


def test_decoder_checks_native_tool_capability(profile_factory):
    with pytest.raises(SemanticLossError, match="function calling"):
        ChatCompletionsDecoder().native_request({"messages": [], "tools": [{}]}, profile_factory())


def test_decoder_validates_filtered_native_message_structure(profile_factory):
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": "hello",
                "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "f", "arguments": 42}}
                ],
                "unknown": {"keep_in_history": True},
            }
        ]
    }
    before = deepcopy(request)
    with pytest.raises(MessageValidationError):
        ChatCompletionsDecoder().native_request(request, profile_factory())
    assert request == before


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
def test_nullable_native_calls_are_treated_as_absent(protocol):
    incoming = [{"role": "assistant", "content": None, "tool_calls": None}]
    if protocol == "ollama":
        incoming[0]["images"] = None
    converter = MyCodec()
    intermediate = converter.encode(incoming, input_format=protocol)
    assert intermediate["messages"][0]["content"] == []
    assert converter.decode(**intermediate, target_format=protocol)["messages"] == [
        {"role": "assistant", "content": "" if protocol == "ollama" else None}
    ]
    assert converter.decode(**intermediate, target_format="anthropic")["messages"] == [
        {"role": "assistant", "content": []}
    ]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_protocols_inherit_common_parameter_execution(protocol):
    codec = MyCodec().codecs[protocol]
    assert type(codec.encoder).parameter_value is BaseEncoder.parameter_value
    assert type(codec.decoder).request_parameters is BaseDecoder.request_parameters
