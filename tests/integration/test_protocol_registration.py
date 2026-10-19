"""A protocol implemented from the abstract contracts joins every request path."""

from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace

import pytest
from typing_extensions import TypedDict

from my_llm import (
    BaseCodec,
    BaseDecoder,
    BaseEncoder,
    BaseTransport,
    ConfigurationError,
    ModelProfile,
    MyCodec,
    MyLLM,
    PreparedOptions,
    ProtocolDefaults,
    ProviderProfile,
    UnsupportedProviderError,
    registry,
    resolve_profiles,
)
from my_llm.codec.shared.conversion import (
    decode_functions,
    encode_functions,
    text,
)
from my_llm.codec.shared.options import apply_max_tokens
from my_llm.codec.shared.stream import BlockRef, batch, events
from tests.helpers.protocols import PROTOCOLS, stream_events

PROTOCOL = "independent_test_protocol"


class NativeMessage(TypedDict):
    speaker: str
    body: str


class NativeTool(TypedDict):
    fn: str
    schema: dict


class Encoder(BaseEncoder):
    __slots__ = ()
    protocol = PROTOCOL

    def match_item(self, item):
        return ("speaker" in item and "body" in item) or "fn" in item

    def _messages(self, values, profile=None, **parameters):
        return [
            {
                "role": value["speaker"],
                "content": [{"type": "text", "text": value["body"]}],
                "protocol": self.protocol,
            }
            for value in values
        ]

    def _tools(self, values, profile=None, **parameters):
        return encode_functions(
            values,
            lambda value: [{"name": value["fn"], "parameters": value["schema"]}],
        )

    def _chunks(self, values, profile=None, **parameters):
        return [
            batch(
                [{"type": "finish", "reason": "stop"}]
                if value["done"]
                else [
                    {
                        "type": "block_delta",
                        "ref": BlockRef("body"),
                        "kind": "text",
                        "field": "text",
                        "value": value["body"],
                    }
                ],
                self.protocol,
            )
            for value in values
        ]

    def usage(self, value):
        return value


class Decoder(BaseDecoder):
    __slots__ = ()
    protocol = PROTOCOL

    def _messages(self, values, profile=None, **parameters):
        return [{"speaker": value["role"], "body": text(value["content"])} for value in values]

    def _tools(self, values, profile=None, **parameters):
        return decode_functions(
            values,
            lambda functions: [
                {"fn": value["name"], "schema": value["parameters"]} for value in functions
            ],
        )

    def _chunks(self, values, profile=None, **parameters):
        result = []
        for event in events(values):
            if event["type"] == "delta":
                result.append({"body": event["value"], "done": False})
            elif event["type"] == "finish":
                result.append({"body": "", "done": True})
        return result

    def usage(self, value):
        return value

    def request_parameters(self, profile, options: PreparedOptions):
        result = deepcopy(options.body)
        for binding in options.bindings:
            result.update(deepcopy(binding))
        apply_max_tokens(result, profile, options.tokens)
        return result

    def assemble_request(self, messages, tools, profile, *, options, **parameters):
        return {
            **self.request_parameters(profile, options),
            "model": profile.model,
            "dialog": messages,
            "functions": tools,
        }


class Client:
    def __init__(self):
        self.requests = []
        self.streams_closed = 0
        self.closed = False

    async def close(self):
        self.closed = True


class Transport(BaseTransport):
    __slots__ = ()
    message_path = ("dialog",)
    tool_path = ("functions",)
    message_type = NativeMessage
    tool_type = NativeTool
    usage_modes = ("independent_counter_mode",)

    def validate_request(self, request, profile, **parameters):
        return request

    def create_client(self, profile, **parameters):
        return Client()

    @asynccontextmanager
    async def upstream(self, client, request, profile, **parameters):
        client.requests.append(request)

        async def stream():
            yield {"body": "hello ", "done": False}
            yield {"body": "extension", "done": False}
            yield {"body": "", "done": True}

        try:
            yield stream()
        finally:
            client.streams_closed += 1


class Codec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = PROTOCOL
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=False,
        supports_reasoning_effort=False,
        max_tokens_path="$.limit",
        content_path="$.body",
        reasoning_content_path="$.thinking",
        reasoning_enable_parameter={"hint": {"level": 1}},
        reasoning_effort_parameter={},
    )
    encoder = Encoder()
    decoder = Decoder()
    transport = Transport()


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_CODECS", registry.registered_codecs())
    MyCodec.register(Codec())


def profiles():
    return (
        ProviderProfile(protocol=PROTOCOL, usage_mode="independent_counter_mode"),
        ModelProfile(model="independent", supports_function_calling=True),
    )


def test_builtin_fixture_protocols_cover_all_registered_protocols():
    assert set(PROTOCOLS) == set(MyCodec().codecs)


def test_registration_adds_defaults_and_preserves_existing_converter_snapshot(monkeypatch):
    monkeypatch.setattr(registry, "_CODECS", registry.registered_codecs())
    existing = MyCodec()
    MyCodec.register(Codec())
    assert PROTOCOL not in existing.codecs
    assert isinstance(MyCodec().codecs[PROTOCOL], Codec)
    provider, model = profiles()
    resolved = resolve_profiles(provider, model)
    assert resolved.get("max_tokens_path") == "$.limit"
    assert resolved.get("content_path") == "$.body"
    assert resolved.get("usage_mode") == "independent_counter_mode"
    with pytest.raises(ConfigurationError, match="registered|duplicate|already"):
        MyCodec.register(Codec())
    with pytest.raises(UnsupportedProviderError):
        existing.transport(PROTOCOL)


@pytest.mark.parametrize(
    "bindings,match",
    [
        ({"protocol_or_api_style": " "}, "non-empty"),
        ({"defaults": {}}, "ProtocolDefaults"),
        ({"encoder": object()}, "BaseEncoder"),
        ({"decoder": object()}, "BaseDecoder"),
        ({"transport": object()}, "BaseTransport"),
        ({"protocol_or_api_style": "different_name"}, "same protocol"),
    ],
)
def test_registration_rejects_invalid_external_components(bindings, match):
    invalid = type("InvalidCodec", (Codec,), {"__slots__": (), **bindings})
    with pytest.raises(ConfigurationError, match=match):
        MyCodec.register(invalid())


def test_registration_rejects_non_codecs_and_stateful_components():
    with pytest.raises(ConfigurationError, match="BaseCodec"):
        MyCodec.register(object())

    class StatefulEncoder(Encoder):
        pass

    class StatefulCodec(Codec):
        __slots__ = ()
        encoder = StatefulEncoder()

    class StatefulFacade(Codec):
        pass

    for invalid in (StatefulCodec(), StatefulFacade()):
        with pytest.raises(ConfigurationError, match="__slots__"):
            MyCodec.register(invalid)


def collected_text(protocol, result):
    if protocol is None:
        assert result["speaker"] == "assistant"
        return result["body"]
    if protocol == "openai_responses":
        assert isinstance(result, list)
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert all(block["type"] == "output_text" for block in result[0]["content"])
        return "".join(block["text"] for block in result[0]["content"])
    if protocol == "gemini":
        assert result["role"] == "model"
        return "".join(part["text"] for part in result["parts"])
    assert result["role"] == "assistant"
    if protocol == "anthropic":
        assert all(block["type"] == "text" for block in result["content"])
        return "".join(block["text"] for block in result["content"])
    return result["content"]


def streamed_text(protocol, chunks):
    if protocol is None:
        assert chunks[-1] == {"body": "", "done": True}
        return "".join(chunk["body"] for chunk in chunks)
    if protocol == "openai_chat":
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        return "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk.get("choices")
        )
    if protocol == "openai_responses":
        assert chunks[-1]["type"] == "response.completed"
        return "".join(
            chunk["delta"] for chunk in chunks if chunk["type"] == "response.output_text.delta"
        )
    if protocol == "anthropic":
        assert chunks[-1]["type"] == "message_stop"
        return "".join(
            chunk["delta"]["text"]
            for chunk in chunks
            if chunk["type"] == "content_block_delta" and chunk["delta"]["type"] == "text_delta"
        )
    if protocol == "gemini":
        assert chunks[-1]["candidates"][0]["finish_reason"] == "STOP"
        return "".join(
            part["text"]
            for chunk in chunks
            for candidate in chunk["candidates"]
            for part in candidate["content"]["parts"]
        )
    assert chunks[-1]["done"] is True
    return "".join(chunk["message"]["content"] for chunk in chunks)


@pytest.mark.parametrize("stdio", [None, *PROTOCOLS])
@pytest.mark.parametrize("streaming", [False, True])
async def test_independent_protocol_joins_conversion_and_request_lifecycle(
    registered, stdio, streaming
):
    provider, model = profiles()
    messages = [{"speaker": "user", "body": "hi"}]
    tools = [{"fn": "weather", "schema": {"type": "object"}}]
    if stdio is not None:
        converted = MyCodec().convert(
            messages=messages, tools=tools, input_format=PROTOCOL, target_format=stdio
        )
        messages, tools = converted["messages"], converted["tools"]
    async with MyLLM(provider=provider, model=model, stdio_protocol=stdio) as llm:
        result = await llm.generate(messages=messages, tools=tools, stream=streaming, max_tokens=23)
        if streaming:
            chunks = [chunk async for chunk in result]
            assert streamed_text(stdio, chunks) == "hello extension"
            result = result.result
        assert llm.stdio_protocol == (stdio or PROTOCOL)
        assert collected_text(stdio, result) == "hello extension"
        client = next(iter(llm._clients.values()))
        assert client.requests == [
            {
                "model": "independent",
                "limit": 23,
                "dialog": [{"speaker": "user", "body": "hi"}],
                "functions": [{"fn": "weather", "schema": {"type": "object"}}],
            }
        ]
        assert client.streams_closed == 1
        assert not client.closed
    assert client.closed


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_builtin_upstreams_render_registered_protocol_without_shared_changes(
    registered, monkeypatch, upstream, streaming
):
    transport = MyCodec().transport(upstream)

    @asynccontextmanager
    async def upstream_events(self, client, request, profile, **parameters):
        client.requests.append(request)

        async def stream():
            for event in stream_events(upstream, tool=False):
                yield event

        try:
            yield stream()
        finally:
            client.streams_closed += 1

    async def close_client(self, client):
        await client.close()

    monkeypatch.setattr(
        type(transport), "create_client", lambda self, profile, **parameters: Client()
    )
    monkeypatch.setattr(type(transport), "upstream", upstream_events)
    monkeypatch.setattr(type(transport), "close_client", close_client)
    provider = ProviderProfile(protocol=upstream)
    model = ModelProfile(model="builtin")
    async with MyLLM(provider=provider, model=model, stdio_protocol=PROTOCOL) as llm:
        result = await llm.generate(
            messages=[{"speaker": "user", "body": "hi"}], stream=streaming, max_tokens=23
        )
        if streaming:
            chunks = [chunk async for chunk in result]
            assert streamed_text(None, chunks) == "Hello"
            result = result.result
        assert collected_text(None, result) == "Hello"
        client = next(iter(llm._clients.values()))
        assert client.streams_closed == 1
        assert not client.closed
        request = client.requests[0]
        if upstream == "gemini":
            assert request["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
        elif upstream == "openai_responses":
            assert request["input"] == [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ]
        elif upstream == "anthropic":
            assert request["messages"] == [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ]
        else:
            assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert client.closed


async def test_protocol_default_bindings_are_isolated_between_profiles_and_requests(registered):
    provider, model = profiles()
    resolved = resolve_profiles(provider, model)
    resolved.get("reasoning_enable_parameter")["hint"]["level"] = 99
    assert resolve_profiles(provider, model).get("reasoning_enable_parameter") == {
        "hint": {"level": 1}
    }
    async with MyLLM(provider=provider, model=model, stdio_protocol=PROTOCOL) as llm:
        messages = [{"speaker": "user", "body": "hi"}]
        await llm.generate(messages=messages, reasoning=True)
        client = next(iter(llm._clients.values()))
        assert client.requests[0]["hint"] == {"level": 1}
        client.requests[0]["hint"]["level"] = 99
        await llm.generate(messages=messages, reasoning=True)
        assert client.requests[1]["hint"] == {"level": 1}
        assert llm.resolved_profile.get("reasoning_enable_parameter") == {"hint": {"level": 1}}
    assert Codec.defaults.reasoning_enable_parameter == {"hint": {"level": 1}}
    assert client.streams_closed == 2
    assert client.closed


def test_custom_usage_mode_is_owned_by_transport(registered):
    provider, model = profiles()
    with pytest.raises(ConfigurationError, match="usage_mode"):
        MyLLM(provider=replace(provider, usage_mode="native"), model=model)
