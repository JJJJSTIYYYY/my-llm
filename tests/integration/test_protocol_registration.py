"""A protocol implemented from the abstract contracts joins every request path."""

from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Annotated

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
    Parameter,
    ProtocolDefaults,
    ProviderProfile,
    UnsupportedProviderError,
    omit,
    registry,
    resolve_profiles,
)
from my_llm.codec.shared.conversion import (
    decode_functions,
    encode_functions,
    select,
    text,
)
from my_llm.codec.shared.stream import BlockRef, batch, events, metadata
from my_llm.stream._state import StreamState
from tests.helpers.profiles import config
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
                "content": [
                    {
                        "type": "text",
                        "text": self.parameter_value(value, profile, "content", "$.body"),
                    }
                ],
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
        result = []
        for value in values:
            operations = (
                [{"type": "finish", "reason": "stop"}]
                if value["done"]
                else [
                    {
                        "type": "block_delta",
                        "ref": BlockRef("body"),
                        "kind": "text",
                        "field": "text",
                        "value": deepcopy(
                            self.parameter_value(value, profile, "content", "$.body")
                        ),
                    }
                ]
            )
            if "trace" in value:
                operations.insert(
                    0,
                    metadata(select(value["trace"], "token", "phase"), path=("trace",)),
                )
            result.append(batch(operations, self.protocol))
        return result

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
                result.append({"body": deepcopy(event["value"]), "done": False})
            elif event["type"] == "finish":
                result.append(
                    {
                        "body": "",
                        "done": True,
                        **(
                            {"trace": deepcopy(select(parameters["trace"], "token", "phase"))}
                            if "trace" in parameters
                            else {}
                        ),
                    }
                )
        return result

    def usage(self, value):
        return value

    def assemble_request(self, messages, tools, profile, *, request, **parameters):
        return {
            **request,
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
    request_schema = {
        "model": str,
        "dialog": list[NativeMessage],
        "functions": list[NativeTool],
        "hint": dict,
        "limit": int,
    }

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
        replay_reasoning_content_when_tool_calls=config("Replay", False),
        supports_reasoning_effort=config("Effort", False),
        max_tokens=Parameter(name="Limit", path="$.limit", default_value=omit, phase="decode"),
        content=Parameter(name="Body", path="$.body", default_value=omit, phase="encode"),
        reasoning_content=Parameter(
            name="Thought", path="$.thinking", default_value=omit, phase="encode"
        ),
        reasoning_enable=Parameter(
            name="Hint", path="$.hint", default_value={"level": 1}, phase="decode"
        ),
        reasoning_effort=Parameter(
            name="Level", path="$.hint.level", default_value=omit, phase="decode"
        ),
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
        ProviderProfile(
            protocol=PROTOCOL, usage_mode=config("usage_mode", "independent_counter_mode")
        ),
        ModelProfile(
            model="independent", supports_function_calling=config("supports_function_calling", True)
        ),
    )


async def test_new_protocol_reuses_caches_with_unhashable_annotations(registered, monkeypatch):
    class Hint(TypedDict):
        level: int

    class AnnotatedTransport(Transport):
        __slots__ = ()
        message_type = Annotated[NativeMessage, {"vendor": "independent"}]
        request_schema = {
            **Transport.request_schema,
            "dialog": Annotated[list[NativeMessage], {"collection": "dialog"}],
            "hint": Annotated[Hint, {"vendor": "hint"}],
        }

    @dataclass(frozen=True, kw_only=True)
    class ExtendedModel(ModelProfile):
        vendor: Parameter[str] = Parameter(
            name="Vendor hint", path="$.hint.vendor", default_value="extra", phase="decode"
        )

    monkeypatch.setattr(Codec, "transport", AnnotatedTransport())
    provider, _ = profiles()
    async with MyLLM(
        provider=provider, model=ExtendedModel(model="independent"), stdio_protocol=PROTOCOL
    ) as llm:
        for body in ("first request", "second request"):
            result = await llm.generate(
                messages=[{"speaker": "user", "body": body}], reasoning=True
            )
            assert result == {"speaker": "assistant", "body": "hello extension"}
        client = next(iter(llm._clients.values()))
        assert [request["dialog"][0]["body"] for request in client.requests] == [
            "first request",
            "second request",
        ]
        assert all(
            request["hint"] == {"level": 1, "vendor": "extra"} for request in client.requests
        )
        assert client.streams_closed == 2
    assert client.closed


def test_new_protocol_queries_filter_plans_once_per_batch(registered, monkeypatch):
    from my_llm.codec.shared import transport

    calls = []
    original = transport._SchemaKey.__init__

    def counted(self, schema):
        calls.append(schema)
        original(self, schema)

    monkeypatch.setattr(transport._SchemaKey, "__init__", counted)
    provider, model = profiles()
    profile = resolve_profiles(provider, model)
    messages = [{"speaker": "user", "body": str(index), "vendor": True} for index in range(100)]
    tools = [{"fn": "fn", "schema": {"properties": {}}, "vendor": True} for _ in range(100)]
    native = Transport()
    assert native.filter_messages(messages, profile) == [
        {"speaker": "user", "body": str(index)} for index in range(100)
    ]
    filtered_tools = native.filter_tools(tools, profile)
    assert len(calls) == 2
    assert len(filtered_tools) == 100
    filtered_tools[0]["schema"]["changed"] = True
    assert "changed" not in tools[0]["schema"]


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
    assert resolved.parameter("max_tokens", "decode").path == "$.limit"
    assert resolved.parameter("content", "encode").path == "$.body"
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


@pytest.mark.parametrize("component", ["codec", "encoder", "decoder", "transport"])
@pytest.mark.parametrize("inherited", [False, True])
def test_registration_rejects_instance_state_declared_in_slots(component, inherited):
    concrete = Codec if component == "codec" else type(getattr(Codec, component))
    stateful = type("StatefulImplementation", (concrete,), {"__slots__": ("pending",)})
    if inherited:
        stateful = type("InheritedImplementation", (stateful,), {"__slots__": ()})
    implementation = stateful()
    implementation.pending = []
    assert not hasattr(implementation, "__dict__")
    invalid = (
        implementation
        if component == "codec"
        else type("StatefulCodec", (Codec,), {"__slots__": (), component: implementation})()
    )
    with pytest.raises(ConfigurationError, match="empty __slots__"):
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
    resolved.get("reasoning_enable")["level"] = 99
    assert resolve_profiles(provider, model).get("reasoning_enable") == {"level": 1}
    async with MyLLM(provider=provider, model=model, stdio_protocol=PROTOCOL) as llm:
        messages = [{"speaker": "user", "body": "hi"}]
        await llm.generate(messages=messages, reasoning=True)
        client = next(iter(llm._clients.values()))
        assert client.requests[0]["hint"] == {"level": 1}
        client.requests[0]["hint"]["level"] = 99
        await llm.generate(messages=messages, reasoning=True)
        assert client.requests[1]["hint"] == {"level": 1}
        assert llm.resolved_profile.get("reasoning_enable") == {"level": 1}
    assert Codec.defaults.reasoning_enable.default_value == {"level": 1}
    assert client.streams_closed == 2
    assert client.closed


def test_custom_usage_mode_is_owned_by_transport(registered):
    provider, model = profiles()
    with pytest.raises(ConfigurationError, match="usage_mode"):
        MyLLM(provider=replace(provider, usage_mode=config("usage_mode", "native")), model=model)


def test_registered_protocol_owns_its_metadata_projection_and_merge_paths(registered):
    class UnreadableExtension:
        def __deepcopy__(self, memo):
            raise AssertionError("copied an unconfigured native metadata field")

    codec, state = MyCodec(), StreamState("independent")
    incoming = [
        {
            "body": "hello",
            "done": False,
            "trace": {"token": "trace-1", "vendor_unconfigured": UnreadableExtension()},
        },
        {"body": "", "done": True, "trace": {"phase": "complete"}},
    ]
    encoded = codec.encode(chunks=incoming, input_format=PROTOCOL)["chunks"]
    assert "vendor_unconfigured" not in str(encoded)
    operations = state.feed(encoded)
    operations.extend(state.finish())
    assert state.metadata["trace"] == {"token": "trace-1", "phase": "complete"}
    output = codec.decode(chunks=operations, target_format=PROTOCOL, **state.metadata)["chunks"]
    assert output[-1] == {
        "body": "",
        "done": True,
        "trace": {"token": "trace-1", "phase": "complete"},
    }


def test_independent_stream_hooks_capture_their_mutable_metadata(registered):
    codec = MyCodec().codecs[PROTOCOL]
    trace = {"token": {"parts": ["original"]}}
    incoming = [{"body": "hello", "done": False, "trace": trace}]
    encoded = codec.encoder.chunks(incoming)
    expected = deepcopy(encoded)

    trace["token"]["parts"].append("caller edit")
    assert encoded == expected
    encoded[0]["events"][0]["values"]["token"]["parts"].append("output edit")
    assert trace["token"]["parts"] == ["original", "caller edit"]

    output = codec.decoder.chunks([{"type": "finish", "reason": "stop"}], trace=trace)
    expected_output = deepcopy(output)
    trace["token"]["parts"].append("later edit")
    assert output == expected_output
    output[0]["trace"]["token"]["parts"].append("output edit")
    assert trace["token"]["parts"] == ["original", "caller edit", "later edit"]


@pytest.mark.parametrize(
    "payload,expected", [({}, "fallback"), ({"body": None}, None), ({"body": "actual"}, "actual")]
)
def test_external_encoder_inherits_phase_routes_and_missing_value_semantics(
    registered, payload, expected
):
    provider, model = profiles()
    declaration = Parameter(
        name="Anything the UI needs",
        comment="Presentation only",
        path="$.vendor.body",
        default_value="fallback",
        phase="encode",
    )
    profile = resolve_profiles(provider, replace(model, content=declaration))
    codec = MyCodec()
    intermediate = codec.codecs[PROTOCOL].encoder.chunks(
        [{"body": "native", "done": False, "vendor": payload}], profile
    )
    assert intermediate[0]["events"][0]["value"] == expected
    request = codec.request(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}, profile
    )
    assert "vendor" not in request
    assert request["dialog"] == [{"speaker": "user", "body": "hello"}]


def test_external_protocol_inherits_generic_decode_parameters_and_phase_isolation(registered):
    @dataclass(frozen=True, kw_only=True)
    class ExtensionModel(ModelProfile):
        vendor_limit: Parameter[int] = Parameter(
            name="No field-name convention", path="$.hint.vendor", default_value=7, phase="decode"
        )
        encoder_only: Parameter[str] = Parameter(
            name="This is not a request",
            path="$.hint.leak",
            default_value="must not leak",
            phase="encode",
        )

    provider, _ = profiles()
    model = ExtensionModel(
        model="independent",
        content=Parameter(
            name="Decorative", path="$.hint.body", default_value="fallback", phase="decode"
        ),
    )
    profile = resolve_profiles(provider, model)
    codec = MyCodec()
    intermediate = codec.codecs[PROTOCOL].encoder.messages(
        [{"speaker": "user", "body": "native"}], profile
    )
    assert intermediate[0]["content"][0]["text"] == "native"
    request = codec.request({"messages": intermediate}, profile)
    assert request["hint"] == {"vendor": 7, "body": "fallback"}
    assert "encoder_only" not in request
    assert "body" not in request
    assert "request_parameters" not in Decoder.__dict__
