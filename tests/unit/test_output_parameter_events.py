"""Output routes select native events independently of protocol field paths."""

from copy import deepcopy
from dataclasses import replace

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    InputParameter,
    ModelProfile,
    OutputParameter,
    Parameter,
    ProviderProfile,
    resolve_profiles,
    resolve_profiles_with_hash,
)
from my_llm.basic.protocols import codec_for
from my_llm.runtime.state import StreamState
from tests.helpers.protocols import PROTOCOLS


@pytest.mark.parametrize("marker,source", [(InputParameter, "input"), (OutputParameter, "output")])
def test_direction_is_known_before_binding(marker, source):
    parameter = marker(path="$.value")
    assert isinstance(parameter, Parameter)
    assert parameter.source == source
    assert parameter.name is None
    with pytest.raises(TypeError):
        marker(path="$.value", source=source)


@pytest.mark.parametrize("value", [None, 0, False, [], {}])
def test_output_event_type_requires_string(value):
    with pytest.raises(ConfigurationError, match="event_type must be a string"):
        OutputParameter(path="$.value", event_type=value)


def test_input_parameter_does_not_accept_event_type():
    with pytest.raises(TypeError, match="event_type"):
        InputParameter(path="$.value", event_type="vendor.delta")


@pytest.mark.parametrize(
    "name,value",
    [
        ("content", InputParameter(path="$.value")),
        ("max_tokens", OutputParameter(path="$.value")),
        ("content", Parameter(path="$.value")),
    ],
)
def test_direction_mismatch_is_rejected(name, value):
    with pytest.raises(ConfigurationError, match=f"{name} does not match"):
        resolve_profiles(
            ProviderProfile(protocol=Config(default_value="openai_chat")),
            ModelProfile(model=Attribute(default_value="demo"), **{name: value}),
        )


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("name,field", [("content", "text"), ("reasoning_content", "reasoning")])
def test_custom_event_routes_only_matching_chunks(protocol, name, field, profile_factory):
    profile = profile_factory(
        protocol,
        **{
            name: OutputParameter(
                path="$.vendor.value", event_type="vendor.delta", default_value="fallback"
            ),
        },
    )
    incoming = [
        {"type": "unrelated", "vendor": {"value": "ignored"}},
        {"type": "vendor.delta", "vendor": {"value": "first "}},
        {"type": "vendor.delta", "vendor": {"value": "second"}},
        {"type": "vendor.delta"},
        {"type": "unrelated"},
    ]
    before = deepcopy(incoming)
    encoder = codec_for(protocol).encoder
    state = StreamState("demo")
    public = []
    # Incremental calls must behave just like a single batch.
    for chunk in incoming:
        public.extend(state.feed(encoder.chunks([chunk], profile)))
    assert [
        event["value"] for event in public if event["type"] == "delta" and event["field"] == field
    ] == [
        "first ",
        "second",
        "fallback",
    ]
    blocks = [block for message in state.messages() for block in message["content"]]
    assert "".join(block.get(field) or "" for block in blocks) == "first secondfallback"
    assert incoming == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_custom_feedback_event_routes_private_value(protocol, profile_factory):
    profile = profile_factory(
        protocol,
        feedback_reasoning_item=OutputParameter(
            path="$.private",
            event_type="vendor.feedback",
        ),
    )
    encoder = codec_for(protocol).encoder
    state = StreamState("demo")
    public = state.feed(
        encoder.chunks(
            [
                {"type": "unrelated", "private": "ignored"},
                {"type": "vendor.feedback", "private": {"encrypted": "opaque"}},
            ],
            profile,
        )
    )
    assert all(event["type"] != "delta" for event in public)
    blocks = [block for message in state.messages() for block in message["content"]]
    assert [
        block["feedback_item"] for block in blocks if block.get("feedback_item") is not None
    ] == [
        {"encrypted": "opaque"},
    ]


def test_event_type_overrides_and_hashes_as_part_of_whole_output_declaration():
    provider = ProviderProfile(
        protocol=Config(default_value="openai_responses"),
        reasoning_content=OutputParameter(path="$.reasoning", event_type="vendor.reasoning"),
    )
    model = ModelProfile(model=Attribute(default_value="demo"))
    inherited, inherited_hash = resolve_profiles_with_hash(provider, model)
    override, override_hash = resolve_profiles_with_hash(
        provider,
        replace(
            model,
            reasoning_content=OutputParameter(
                path="$.delta", event_type="response.reasoning_text.delta"
            ),
        ),
    )
    assert inherited.parameter("reasoning_content").event_type == "vendor.reasoning"
    assert override.parameter("reasoning_content").event_type == "response.reasoning_text.delta"
    assert override_hash != inherited_hash
    inherited.parameter("reasoning_content").event_type = "changed"
    assert provider.reasoning_content.event_type == "vendor.reasoning"


def test_event_type_does_not_filter_history(profile_factory):
    profile = profile_factory(
        "openai_chat",
        feedback_reasoning_item=OutputParameter(
            path="$.choices[0].delta.private",
            event_type="vendor.feedback",
        ),
    )
    message = codec_for("openai_chat").encoder.messages(
        [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "visible",
                "private": "raw",
            },
        ],
        profile,
    )[0]
    assert message.content[1].feedback_item == "raw"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_custom_output_is_preserved_for_semantic_loss_reporting(
    protocol, profile_factory
):
    profile = profile_factory(
        protocol,
        content=OutputParameter(
            path="$.value",
            event_type="vendor.text",
            supports=False,
        ),
    )
    state = StreamState("demo")
    public = state.feed(
        codec_for(protocol).encoder.chunks(
            [
                {"type": "vendor.text", "value": "unsupported text"},
            ],
            profile,
        )
    )
    assert [event["value"] for event in public if event["type"] == "delta"] == ["unsupported text"]


def test_chat_usage_tail_does_not_apply_missing_text_defaults(profile_factory):
    profile = profile_factory(
        "openai_chat",
        content=OutputParameter(
            path="$.choices[0].delta.content",
            default_value="fallback",
        ),
    )
    batches = codec_for("openai_chat").encoder.chunks(
        [
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        ],
        profile,
    )
    assert all(event["type"] != "block_delta" for batch in batches for event in batch.events)


@pytest.mark.parametrize(
    "protocol,chunk",
    [
        (
            "openai_chat",
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": "native text"}},
                ],
            },
        ),
        ("openai_responses", {"type": "response.output_text.delta", "delta": "native text"}),
        (
            "anthropic",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "native text"},
            },
        ),
    ],
)
def test_explicit_event_type_excludes_native_deltas(protocol, chunk, profile_factory):
    profile = profile_factory(
        protocol,
        content=OutputParameter(
            path=codec_for(protocol).defaults.content.path,
            event_type="vendor.text",
            default_value="fallback",
        ),
    )
    state = StreamState("demo")
    public = state.feed(codec_for(protocol).encoder.chunks([chunk], profile))
    assert not [event for event in public if event["type"] == "delta"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"type": "vendor.text", "value": None}, None),
        ({"type": "vendor.text"}, "fallback"),
        ({"type": "unrelated"}, "missing"),
    ],
)
def test_event_lookup_accepts_sdk_models_and_preserves_presence(value, expected, profile_factory):
    from pydantic import BaseModel, ConfigDict

    class Event(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: str

    profile = profile_factory(
        "openai_responses",
        content=OutputParameter(
            path="$.value",
            event_type="vendor.text",
            default_value="fallback",
        ),
    )
    encoder = codec_for("openai_responses").encoder
    assert (
        encoder.parameter_value(
            Event(**value),
            profile,
            "content",
            "$.delta",
            "missing",
            event=True,
        )
        == expected
    )
