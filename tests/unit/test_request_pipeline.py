"""Complete call envelopes, preparation lifetime and destination isolation."""

import inspect
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from my_llm import ModelProfile, MyCodec, MyLLM, Parameter, ProviderProfile, resolve_profiles
from my_llm.profile import resolution
from tests.helpers.protocols import PROTOCOLS
from tests.performance.runner import measure
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.validation import validate_exchange
from tests.performance.workloads import Size, build, controls, profiles

SMALL = Size(
    rounds=1,
    tools=1,
    parallel_calls=1,
    text_chars=32,
    output_chars=64,
    fragment_chars=17,
    argument_items=1,
    schema_depth=1,
)


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("stdio", ["openai_chat", "gemini"])
async def test_complete_call_passes_through_codec_with_explicit_destinations(
    monkeypatch,
    upstream,
    stdio,
):
    workload = build(upstream, stdio, SMALL, native=False)
    provider, model = profiles(upstream)
    codec = MyCodec()
    encoder_type = type(codec.codecs[stdio].encoder)
    decoder_type = type(codec.decoder(upstream))
    encoder_request = encoder_type.request
    decoder_prepare = decoder_type.prepare
    decoder_connection = decoder_type.client_parameters
    decoder_options = decoder_type.validate_parameters
    envelopes, intermediates, outputs, client_preparations, option_preparations = [], [], [], [], []

    def encode(self, inputs, profile):
        envelopes.append(dict(inputs))
        result = encoder_request(self, inputs, profile)
        intermediates.append(result)
        return result

    def decode(self, intermediate, client):
        result = decoder_prepare(self, intermediate, client)
        outputs.append(result)
        return result

    def connection(self, profile):
        client_preparations.append(profile)
        return decoder_connection(self, profile)

    def options(self, profile, **parameters):
        option_preparations.append(profile)
        return decoder_options(self, profile, **parameters)

    monkeypatch.setattr(encoder_type, "request", encode)
    monkeypatch.setattr(decoder_type, "prepare", decode)
    monkeypatch.setattr(decoder_type, "client_parameters", connection)
    monkeypatch.setattr(decoder_type, "validate_parameters", options)
    with patch.object(
        resolution, "_connection_parameters", wraps=resolution._connection_parameters
    ) as connections:
        with LocalSDK(upstream, wire_frames(upstream, workload.events, 37)) as sdk:
            async with MyLLM(provider=provider, model=model, stdio_protocol=stdio) as llm:
                for stream in (False, True):
                    _, result = await measure(
                        llm, workload, stream=stream, parameters=controls(upstream)
                    )
                    validate_exchange(workload, stdio, result, sdk)
                assert sdk.created_clients == 1
                assert all(value["profile"] is llm.resolved_profile for value in intermediates)
                assert client_preparations == [llm.resolved_profile]
                assert option_preparations == [llm.resolved_profile, llm.resolved_profile]
            assert all(body.closed for body in sdk.bodies)
        connections.assert_called_once()
    fields = set(inspect.signature(MyLLM.generate).parameters) - {"self"}
    assert all(set(envelope) == fields for envelope in envelopes)
    assert [output["local"]["stream"] for output in outputs] == [False, True]
    for intermediate, output in zip(intermediates, outputs):
        assert intermediate["messages"][0]["protocol"] == stdio
        assert set(output) == {"client", "request", "local"}
        assert "profile" not in output["request"] and "client_key" not in output["request"]
        assert "reasoning_effort" not in output["client"]
        assert "extra_body" not in output["local"]
        assert "api_key" not in output["request"]
    assert outputs[0]["request"] is not outputs[1]["request"]


def test_client_scope_is_declared_instead_of_inferred_from_field_name():
    @dataclass(frozen=True, kw_only=True)
    class AuthProvider(ProviderProfile):
        authentication: Parameter[str] = field(
            default=Parameter(
                name="Custom authentication",
                path="$.extra_headers.x-auth",
                default_value="offline-token",
                phase="decode",
            ),
            metadata={"scope": "client"},
        )

    profile = resolve_profiles(AuthProvider(), ModelProfile(model="demo"))
    codec = MyCodec()
    codec.validate_profile(profile)
    assert codec.client_parameters(profile) == {"default_headers": {"x-auth": "offline-token"}}
    assert "authentication" not in dict(profile.request_parameters)
    intermediate = codec.encode([{"role": "user", "content": "hi"}], input_format="openai_chat")
    assert "extra_headers" not in codec.request(intermediate, profile)


@pytest.mark.parametrize("upstream", PROTOCOLS)
def test_request_projects_native_values_without_public_result_copies(monkeypatch, upstream):
    codec = MyCodec()
    provider, model = profiles(upstream)
    profile = resolve_profiles(provider, model)
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")

    def public_copy(*args, **kwargs):
        pytest.fail("request preparation copied the public conversion result before filtering it")

    monkeypatch.setattr(type(codec.decoder(upstream)), "messages", public_copy)
    request = codec.request(intermediate, profile)
    assert request["model"] == "benchmark"


def test_connection_preparation_does_not_reparse_the_snapshot(monkeypatch):
    provider, model = profiles("gemini")
    profile = resolve_profiles(provider, model)

    def reparsed(*args, **kwargs):
        pytest.fail("connection values were parsed after profile resolution")

    monkeypatch.setattr(resolution, "_connection_parameters", reparsed)
    assert (
        MyCodec().client_parameters(profile)["http_options"].base_url == "https://benchmark.invalid"
    )


def test_anthropic_open_tool_schema_is_captured_once():
    copies = []

    class Values(list):
        def __deepcopy__(self, memo):
            copies.append(True)
            return list(self)

    source = {
        "name": "choose",
        "input_schema": {
            "type": "object",
            "properties": {"choice": {"enum": Values([1, 2])}},
            "additionalProperties": False,
        },
    }
    profile = resolve_profiles(ProviderProfile(protocol="anthropic"), ModelProfile(model="demo"))
    result = MyCodec().decoder("anthropic").filter_tools([source], profile)
    assert copies == [True]
    result[0]["input_schema"]["properties"]["choice"]["enum"].append(3)
    assert source["input_schema"]["properties"]["choice"]["enum"] == [1, 2]
