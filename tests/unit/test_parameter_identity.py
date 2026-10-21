"""Profile identity, integrated capabilities and generated-control precedence."""

from dataclasses import FrozenInstanceError, dataclass, field, replace

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    ModelProfile,
    Parameter,
    ProviderProfile,
    UnsupportedCapabilityError,
    omit,
    resolve_profiles,
)
from tests.helpers.profiles import parameter
from tests.helpers.protocols import PROTOCOLS
from tests.helpers.requests import encode_request


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_profile_fixes_names_and_sources_across_every_layer(protocol):
    from my_llm.registry import codec_for

    provider = ProviderProfile(
        protocol=Config(default_value=protocol),
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        timeout=Config(default_value=30),
    )
    model = ModelProfile(
        model=Attribute(default_value="demo"), content=parameter("content", protocol=protocol)
    )
    resolved = resolve_profiles(provider, model)
    for profile in (codec_for(protocol).defaults, provider, model):
        for name, source in (("max_tokens", "input"), ("content", "output")):
            value = getattr(profile, name)
            if isinstance(value, Parameter):
                assert (value.name, value.source) == (name, source)
    assert provider.timeout.name == "timeout"
    assert resolved.parameter("max_tokens", "output") is None
    assert resolved.parameter("content", "input") is None


@pytest.mark.parametrize("attribute,value", [("name", "other"), ("source", "output")])
def test_parameter_identity_cannot_be_supplied_or_reassigned(attribute, value):
    with pytest.raises(TypeError, match=attribute):
        Parameter(path="$.limit", **{attribute: value})
    bound = ModelProfile(
        model=Attribute(default_value="demo"), max_tokens=Parameter(path="$.limit")
    ).max_tokens
    with pytest.raises(FrozenInstanceError):
        setattr(bound, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        replace(bound, **{attribute: value})


def test_config_name_cannot_be_supplied_or_rebound_to_another_field():
    with pytest.raises(TypeError, match="name"):
        Config(name="other", default_value=1)
    bound = ProviderProfile(timeout=Config(default_value=30)).timeout
    with pytest.raises(ConfigurationError, match="already been assigned"):
        ProviderProfile(max_retries=bound)
    assert (
        replace(
            ProviderProfile(timeout=bound), protocol=Config(default_value="openai_chat")
        ).timeout
        is bound
    )


def test_reusable_unbound_declaration_gets_independent_fixed_identity():
    declaration = Parameter(path="$.vendor")
    profile = ModelProfile(
        model=Attribute(default_value="demo"), content=declaration, reasoning_content=declaration
    )
    assert declaration.name is declaration.source is None
    assert profile.content.name == "content"
    assert profile.reasoning_content.name == "reasoning_content"
    with pytest.raises(ConfigurationError, match="already been assigned"):
        ModelProfile(model=Attribute(default_value="demo"), reasoning_content=profile.content)


def test_custom_parameters_require_source_in_profile_declaration():
    @dataclass(frozen=True, kw_only=True)
    class MissingSource(ModelProfile):
        vendor: Parameter[int] = Parameter(path="$.extra_body.vendor")

    with pytest.raises(ConfigurationError, match="source in Profile"):
        MissingSource(model=Attribute(default_value="demo"))

    @dataclass(frozen=True, kw_only=True)
    class Vendor(ModelProfile):
        vendor: Parameter[int] = field(
            default=Parameter(path="$.extra_body.vendor", default_value=7),
            metadata={"source": "input"},
        )

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        Vendor(model=Attribute(default_value="demo")),
    )
    assert encode_request(profile)["extra_body"]["vendor"] == 7


def test_client_parameters_must_be_inputs():
    @dataclass(frozen=True, kw_only=True)
    class InvalidClient(ProviderProfile):
        credential: Parameter[str] = field(
            default=Parameter(path="$.api_key"),
            metadata={"scope": "client", "source": "output"},
        )

    with pytest.raises(ConfigurationError, match="credential must use the input source"):
        resolve_profiles(
            InvalidClient(protocol=Config(default_value="openai_chat")),
            ModelProfile(model=Attribute(default_value="demo")),
        )


@pytest.mark.parametrize("value", [None, Parameter(path="$.max_completion_tokens")])
@pytest.mark.parametrize("source,scope", [("output", "request"), ("input", "client")])
def test_parameter_source_and_scope_cannot_change_between_profile_layers(value, source, scope):
    @dataclass(frozen=True, kw_only=True)
    class ChangedModel(ModelProfile):
        max_tokens: Parameter[int] | None = field(
            default=value, metadata={"source": source, "scope": scope}
        )

    with pytest.raises(ConfigurationError, match="max_tokens must keep the same source and scope"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            ChangedModel(model=Attribute(default_value="demo")),
        )


def test_profile_overriding_post_init_cannot_silently_skip_identity_binding():
    @dataclass(frozen=True, kw_only=True)
    class UnboundModel(ModelProfile):
        def __post_init__(self):
            pass

    with pytest.raises(ConfigurationError, match="identity must be bound"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            UnboundModel(
                model=Attribute(default_value="demo"),
                max_tokens=Parameter(path="$.max_completion_tokens"),
            ),
        )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_omitting_enable_route_preserves_independent_effort_route(protocol):
    from my_llm.codec.shared.options import path_value

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value=protocol)),
        ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
            reasoning_enable=omit,
            reasoning_effort=parameter("reasoning_effort", protocol=protocol, supports=True),
        ),
    )
    request = encode_request(profile, reasoning_effort="high")
    assert path_value(request, profile.parameter("reasoning_effort").path) == "high"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_omitting_both_reasoning_routes_only_disables_automatic_insertion(protocol):
    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value=protocol)),
        ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
            reasoning_enable=omit,
            reasoning_effort=omit,
        ),
    )
    assert encode_request(profile, reasoning_effort="high") == encode_request(profile)


@pytest.mark.parametrize("value", [None, 0, "ignored", {"enabled": True}])
def test_unsupported_parameter_forces_omit_before_profile_type_validation(value):
    declaration = Parameter(path="$.max_completion_tokens", supports=False, default_value=value)
    assert declaration.default_value is omit
    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        ModelProfile(model=Attribute(default_value="demo"), max_tokens=declaration),
    )
    assert profile.values["max_tokens"] is omit
    assert profile.parameter("max_tokens") is None
    assert "max_tokens" not in dict(profile.request_parameters)
    request = encode_request(profile, max_tokens=100, extra_body={"max_completion_tokens": 12})
    assert request["extra_body"]["max_completion_tokens"] == 12
    assert "max_completion_tokens" not in request


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_reasoning_capability_is_owned_by_parameter(protocol):
    model = ModelProfile(
        model=Attribute(default_value="demo"),
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=False),
    )
    profile = resolve_profiles(ProviderProfile(protocol=Config(default_value=protocol)), model)
    assert profile.parameters["reasoning_enable"].default_value is omit
    assert "supports_reasoning" not in profile.values
    assert "supports_reasoning_effort" not in profile.values
    with pytest.raises(UnsupportedCapabilityError, match="reasoning"):
        encode_request(profile, reasoning=True)
    enabled = replace(model, reasoning_enable=parameter("reasoning_enable", protocol=protocol))
    profile = resolve_profiles(ProviderProfile(protocol=Config(default_value=protocol)), enabled)
    assert profile.parameter("reasoning_enable").supports is True
    encode_request(profile, reasoning=True)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_generated_enable_wins_over_native_body_and_preserves_input(protocol):
    from copy import deepcopy

    from my_llm.codec.shared.options import path_value

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value=protocol)),
        ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        ),
    )
    binding = profile.parameter("reasoning_enable")
    body = {
        "openai_chat": {"reasoning_effort": "low"},
        "openai_responses": {"reasoning": {"summary": "auto"}},
        "anthropic": {"thinking": {"type": "disabled"}},
        "gemini": {"thinking_config": {"thinking_budget": 0}},
        "ollama": {"think": False},
    }[protocol]
    before = deepcopy(body)
    request = encode_request(profile, reasoning=True, extra_body=body)
    assert path_value(request, binding.path) == binding.default_value
    if protocol in ("openai_chat", "openai_responses"):
        assert path_value(request["extra_body"], binding.path) == binding.default_value
    assert body == before


@pytest.mark.parametrize("value", [None, "false", 0, 1, omit])
def test_supports_requires_a_real_boolean(value):
    with pytest.raises(ConfigurationError, match="supports must be a bool"):
        Parameter(path="$.value", supports=value)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("name,kind", [("content", "text"), ("reasoning_content", "thinking")])
def test_unsupported_output_parameter_skips_present_native_payload(protocol, name, kind):
    from my_llm import MyCodec

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value=protocol)),
        ModelProfile(
            model=Attribute(default_value="demo"),
            **{name: parameter(name, protocol=protocol, supports=False)},
        ),
    )
    codec = MyCodec().codecs[protocol]
    event = codec.decoder.chunks(
        [{"type": "delta", "field": kind, "value": "disabled"}],
        source_protocol=protocol,
    )[0]
    operations = [
        event for batch in codec.encoder.chunks([event], profile) for event in batch["events"]
    ]
    assert not any(
        event["type"] == "block_delta"
        and event.get("field") == kind
        or event["type"] == "block_append"
        and event["block"]["type"] == kind
        for event in operations
    )


@pytest.mark.parametrize("name,kind", [("content", "text"), ("reasoning_content", "thinking")])
def test_anthropic_initial_block_cannot_restore_unsupported_output(name, kind):
    from copy import deepcopy

    from my_llm import MyCodec
    from my_llm.stream._state import StreamState

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="anthropic")),
        ModelProfile(
            model=Attribute(default_value="demo"),
            **{name: parameter(name, protocol="anthropic", supports=False)},
        ),
    )
    block = {"type": kind, kind: "disabled"}
    if kind == "thinking":
        block["signature"] = "native-signature"
    incoming = [
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    before = deepcopy(incoming)
    state = StreamState("demo")
    state.feed(MyCodec().codecs["anthropic"].encoder.chunks(incoming, profile))
    state.finish()
    result = state.messages()[0]["content"][0]
    assert result.get(kind, "") == ""
    if kind == "thinking":
        assert result["signature"] == "native-signature"
    assert incoming == before


@pytest.mark.parametrize("name", ["content", "reasoning_content"])
def test_responses_final_snapshot_cannot_restore_unsupported_output(name):
    from copy import deepcopy

    from my_llm import MyCodec
    from my_llm.stream._state import StreamState
    from tests.helpers.protocols import response_completed

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_responses")),
        ModelProfile(
            model=Attribute(default_value="demo"),
            **{name: parameter(name, protocol="openai_responses", supports=False)},
        ),
    )
    part = {"type": "output_text", "text": "answer", "annotations": []}
    summary = {"type": "summary_text", "text": "thought"}
    incoming = [
        {"type": "response.content_part.done", "output_index": 0, "content_index": 0, "part": part},
        {
            "type": "response.reasoning_summary_part.done",
            "output_index": 1,
            "summary_index": 0,
            "part": summary,
        },
        response_completed(
            [
                {"type": "message", "id": "message_1", "role": "assistant", "content": [part]},
                {
                    "type": "reasoning",
                    "id": "reasoning_1",
                    "summary": [summary],
                    "encrypted_content": "opaque-reasoning",
                },
                {
                    "type": "function_call",
                    "id": "item_1",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": "{}",
                },
            ]
        ),
    ]
    before = deepcopy(incoming)
    state = StreamState("demo")
    state.feed(MyCodec().codecs["openai_responses"].encoder.chunks(incoming, profile))
    state.finish()
    blocks = [block for message in state.messages() for block in message["content"]]
    text = next(block for block in blocks if block["type"] == "text")
    reasoning = next(block for block in blocks if block["type"] == "thinking")
    assert text["text"] == ("" if name == "content" else "answer")
    assert reasoning["thinking"] == ("" if name == "reasoning_content" else "thought")
    assert reasoning["summary"][0]["text"] == ("" if name == "reasoning_content" else "thought")
    assert reasoning["encrypted_content"] == "opaque-reasoning"
    assert next(block for block in blocks if block["type"] == "tool_use")["id"] == "call_1"
    assert incoming == before
