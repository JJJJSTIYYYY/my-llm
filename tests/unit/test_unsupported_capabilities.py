"""Unsupported configured controls use complete decoder drafts and loss hooks."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import pytest
from typing_extensions import TypedDict

from my_llm import (
    Attribute,
    Config,
    MessageValidationError,
    ModelProfile,
    MyCodec,
    Parameter,
    ProviderProfile,
    SemanticLossError,
    resolve_profiles,
)
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from tests.helpers.profiles import parameter
from tests.helpers.protocols import PROTOCOLS


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_unsupported_reasoning_control_reaches_policy_with_complete_request(
    profile_factory, protocol, policy
):
    observed = []

    def hook(draft, losses, context):
        assert draft["model"] == "demo"
        assert "complete tail" in str(draft)
        assert context.target_protocol == protocol
        assert context.origin_source == "input"
        assert any(
            item.origin_path == "$.reasoning"
            and item.origin_value is True
            and "support" in item.reason
            for item in losses
        )
        draft["handled_unsupported_reasoning"] = True
        observed.append(draft)

    codec = MyCodec(policy, input_semantic_loss_hook=hook)
    inputs = codec.encode(
        [{"role": "user", "content": "first"}, {"role": "user", "content": "complete tail"}],
        input_format="openai_chat",
    )
    before = deepcopy(inputs)
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 4096, protocol=protocol),
        reasoning_enable=parameter("reasoning_enable", supports=False, protocol=protocol),
    )
    if policy == "raise":
        with pytest.raises(SemanticLossError, match="support") as caught:
            codec.request(inputs, profile, reasoning=True)
        assert "complete tail" in str(caught.value.draft)
        assert caught.value.draft["model"] == "demo"
    else:
        result = codec.request(inputs, profile, reasoning=True)
        assert "complete tail" in str(result)
        if policy == "hook":
            assert observed == [result]
            assert observed[0] is result
            assert result["handled_unsupported_reasoning"] is True
        else:
            assert observed == []
    assert inputs == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_reasoning_effort_is_reported_even_if_enable_is_supported(
    profile_factory, protocol
):
    codec = MyCodec()
    inputs = codec.encode(
        [{"role": "user", "content": "complete tail"}], input_format="openai_chat"
    )
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 4096, protocol=protocol),
        reasoning_effort=parameter("reasoning_effort", supports=False, protocol=protocol),
    )
    with pytest.raises(SemanticLossError, match="support") as caught:
        codec.request(inputs, profile, reasoning_effort="high")
    assert "complete tail" in str(caught.value.draft)
    assert any(
        item.origin_path == "$.reasoning_effort" and item.origin_value == "high"
        for item in caught.value.losses
    )


def test_explicitly_omitted_unsupported_token_limit_is_not_a_loss(profile_factory):
    from my_llm import omit

    profile = profile_factory(max_tokens=parameter("max_tokens", supports=False))
    codec = MyCodec()
    inputs = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")
    request = codec.request(inputs, profile, max_tokens=omit)
    assert "max_completion_tokens" not in request


def test_unsupported_effort_cannot_use_the_native_reasoning_enable_alias(profile_factory):
    codec = MyCodec()
    inputs = codec.encode(
        [{"role": "user", "content": "complete tail"}], input_format="openai_chat"
    )
    profile = profile_factory()
    body = {"reasoning_effort": "high"}
    with pytest.raises(SemanticLossError, match="support") as caught:
        codec.request(inputs, profile, extra_body=body)
    assert "reasoning_effort" not in caught.value.draft
    assert "reasoning_effort" not in caught.value.draft.get("extra_body", {})
    assert [(loss.origin_path, loss.origin_value) for loss in caught.value.losses] == [
        ("$.extra_body.reasoning_effort", "high")
    ]
    assert "complete tail" in str(caught.value.draft)
    assert body == {"reasoning_effort": "high"}

    enabled = codec.request(inputs, profile, extra_body={"reasoning_effort": "medium"})
    assert enabled["extra_body"]["reasoning_effort"] == "medium"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_token_limit_has_a_complete_draft_for_every_protocol(profile_factory, protocol):
    from my_llm.codec.shared.options import path_value

    declaration = parameter("max_tokens", supports=False, protocol=protocol)
    profile = profile_factory(protocol, max_tokens=declaration)
    codec = MyCodec()
    inputs = codec.encode(
        [{"role": "user", "content": "complete tail"}], input_format="openai_chat"
    )
    with pytest.raises(SemanticLossError, match="support") as caught:
        codec.request(inputs, profile, max_tokens=100)
    assert "complete tail" in str(caught.value.draft)
    assert path_value(caught.value.draft, declaration.path, "absent") == "absent"
    assert any(
        loss.origin_path == "$.max_tokens" and loss.origin_value == 100
        for loss in caught.value.losses
    )


@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_unsupported_token_limit_removes_public_and_native_values(profile_factory, policy):
    observed = []

    def hook(draft, losses, context):
        assert {loss.origin_value for loss in losses} == {12, 100}
        assert all("support" in loss.reason for loss in losses)
        assert "max_completion_tokens" not in draft
        assert "max_completion_tokens" not in draft.get("extra_body", {})
        draft["handled_token_limit"] = True
        observed.append(draft)

    codec = MyCodec(policy, input_semantic_loss_hook=hook)
    profile = profile_factory(max_tokens=parameter("max_tokens", supports=False))
    inputs = codec.encode(
        [{"role": "user", "content": "complete tail"}], input_format="openai_chat"
    )
    body = {"max_completion_tokens": 12}
    if policy == "raise":
        with pytest.raises(SemanticLossError) as caught:
            codec.request(inputs, profile, max_tokens=100, extra_body=body)
        draft = caught.value.draft
        assert {loss.origin_value for loss in caught.value.losses} == {12, 100}
    else:
        draft = codec.request(inputs, profile, max_tokens=100, extra_body=body)
        if policy == "hook":
            assert observed[0] is draft
            assert draft["handled_token_limit"] is True
        else:
            assert observed == []
    assert "complete tail" in str(draft)
    assert "max_completion_tokens" not in draft
    assert "max_completion_tokens" not in draft.get("extra_body", {})
    assert body == {"max_completion_tokens": 12}


@pytest.mark.parametrize(
    "protocol,path,body",
    [
        ("openai_chat", "$.extra_body.temperature", {"temperature": 0.0}),
        ("openai_responses", "$.extra_body.temperature", {"temperature": 0.0}),
        ("anthropic", "$.temperature", {"temperature": 0.0}),
        ("gemini", "$.config.temperature", {"temperature": 0.0}),
        ("ollama", "$.options.temperature", {"options": {"temperature": 0.0}}),
    ],
)
def test_custom_unsupported_parameter_cannot_be_restored_by_native_body(protocol, path, body):
    from my_llm.codec.shared.options import path_value

    @dataclass(frozen=True, kw_only=True)
    class Model(ModelProfile):
        temperature: Parameter[float] = field(
            default=Parameter(path=path, supports=False), metadata={"source": "input"}
        )

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value=protocol)),
        Model(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 128, protocol=protocol),
        ),
    )
    codec = MyCodec()
    inputs = codec.encode(
        [{"role": "user", "content": "complete tail"}], input_format="openai_chat"
    )
    before = deepcopy(body)
    with pytest.raises(SemanticLossError, match="support") as caught:
        codec.request(inputs, profile, extra_body=body)
    assert path_value(caught.value.draft, path, "absent") == "absent"
    assert "complete tail" in str(caught.value.draft)
    assert any(loss.origin_value == 0.0 for loss in caught.value.losses)
    assert body == before


def test_decoder_sdk_union_loss_preserves_sibling_blocks_and_input(profile_factory):
    class Text(TypedDict):
        type: Literal["text"]
        text: str

    class Image(TypedDict):
        type: Literal["image"]
        url: str

    class Message(TypedDict):
        role: Literal["user"]
        content: list[Text | Image]

    class Decoder(ChatCompletionsDecoder):
        __slots__ = ()
        message_type = Message

    native = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "unsupported", "text": "lost"},
                    {"type": "text", "text": "complete tail"},
                ],
            }
        ]
    }
    before = deepcopy(native)
    decoder = Decoder()
    with pytest.raises(SemanticLossError) as caught:
        decoder.native_request(native, profile_factory())
    assert caught.value.draft["messages"][0]["content"] == [
        {"type": "text", "text": "before"},
        {"type": "text", "text": "complete tail"},
    ]
    assert [(loss.origin_path, loss.origin_value) for loss in caught.value.losses] == [
        ("$.messages[0].content[1]", native["messages"][0]["content"][1])
    ]
    assert native == before
    with pytest.raises(MessageValidationError):
        decoder.validate_values(native["messages"], decoder.message_type)


def test_encoder_rejects_invalid_protocol_structure_as_message_validation():
    codec = MyCodec("ignore")
    with pytest.raises(MessageValidationError):
        codec.validate_input("openai_chat", [{"role": "unsupported", "content": "hello"}])


def test_supported_extension_cannot_restore_an_unsupported_parent(profile_factory):
    profile = profile_factory(
        "openai_responses",
        reasoning_enable=Parameter(path="$.reasoning", supports=False),
        reasoning_effort=Parameter(path="$.reasoning.vendor_effort"),
    )
    native = {
        "input": [{"role": "user", "content": "complete tail"}],
        "reasoning": {"vendor_effort": "high"},
    }
    before = deepcopy(native)
    with pytest.raises(SemanticLossError, match="support") as caught:
        MyCodec().decoder("openai_responses").native_request(native, profile)
    assert "reasoning" not in caught.value.draft
    assert "complete tail" in str(caught.value.draft)
    assert caught.value.losses.items[0].origin_value is native["reasoning"]
    assert native == before
