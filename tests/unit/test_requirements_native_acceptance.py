"""Reverse acceptance of supported content and native-only discard boundaries."""

import json

import pytest

from my_llm import Config, MyCodec, SemanticLossError, omit
from tests.helpers.profiles import parameter
from tests.helpers.protocols import PROTOCOLS
from tests.helpers.tool_results import application_result_history, native_result_text


class Unreadable:
    def __deepcopy__(self, memo):
        raise AssertionError("an unsupported native field was traversed")


def target_profile(profile_factory, target):
    return profile_factory(
        target,
        supports_function_calling=Config(default_value=True),
        supports_vision=Config(default_value=True),
        max_tokens=parameter("max_tokens", 2048, protocol=target),
    )


async def prepare(codec, messages, source, profile):
    return (
        await codec.prepare(
            {"messages": messages, "tools": omit, "stream": False},
            profile,
            input_format=source,
            client=codec.client_parameters(profile),
        )
    )["request"]


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
async def test_application_arrays_with_type_keys_remain_tool_text(source, target, profile_factory):
    payload = [{"type": "invoice", "value": 1}, {"type": "receipt", "value": {"count": 2}}]
    request = await prepare(
        MyCodec(),
        application_result_history(source, payload),
        source,
        target_profile(profile_factory, target),
    )
    assert json.loads(native_result_text(target, request)) == payload


@pytest.mark.parametrize("source", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("target", PROTOCOLS)
async def test_unknown_native_tool_parts_drop_before_validation_and_json(
    source, target, profile_factory
):
    text_type = "input_text" if source == "openai_responses" else "text"
    file_type = (
        "input_file"
        if source == "openai_responses"
        else "document"
        if source == "anthropic"
        else "file"
    )
    messages = application_result_history(
        source,
        [
            {"type": text_type, "text": "before"},
            {"type": "input_audio", "data": Unreadable()},
            {"type": file_type, "data": Unreadable()},
            {"type": text_type, "text": "after"},
        ],
    )
    request = await prepare(MyCodec(), messages, source, target_profile(profile_factory, target))
    assert native_result_text(target, request) == "beforeafter"


@pytest.mark.parametrize("target", PROTOCOLS)
async def test_empty_tool_schema_is_preserved_or_reported_with_complete_draft(
    target, profile_factory
):
    schema = {}
    tools = [
        {"type": "function", "function": {"name": "first", "parameters": schema}},
        {"type": "function", "function": {"name": "later", "parameters": {"type": "object"}}},
    ]
    codec = MyCodec()
    profile = target_profile(profile_factory, target)
    inputs = {"messages": [{"role": "user", "content": "hi"}], "tools": tools, "stream": False}
    request = (
        await codec.prepare(
            inputs, profile, input_format="openai_chat", client=codec.client_parameters(profile)
        )
    )["request"]
    actual = (
        request["tools"][0]["function"]["parameters"]
        if target == "openai_chat"
        else request["tools"][0]["parameters"]
        if target == "openai_responses"
        else request["tools"][0]["input_schema"]
    )
    assert actual == {}


@pytest.mark.parametrize("target", PROTOCOLS)
def test_chat_custom_calls_drop_without_affecting_function_partials(target):
    codec = MyCodec()
    messages = [
        {
            "role": "assistant",
            "content": "answer",
            "tool_calls": [{"type": "custom", "id": "ignored", "custom": {"input": Unreadable()}}],
        }
    ]
    converted = codec.convert(messages, input_format="openai_chat", target_format=target)
    assert "answer" in str(converted)
    assert "ignored" not in str(converted)
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"type": "custom", "index": 0, "custom": {"input": Unreadable()}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "index": 1,
                                "id": "c",
                                "function": {"name": "f", "arguments": '{"value":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "1}"}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    result = codec.convert(chunks=events, input_format="openai_chat", target_format=target)
    assert "ignored" not in str(result)
    assert "value" in str(result)
    assert "f" in str(result)


@pytest.mark.parametrize("target", PROTOCOLS)
def test_foreign_feedback_owner_is_taken_from_the_message_protocol(target):
    from my_llm import AssistantMessage, ReasoningContent

    owner = "openai_responses" if target == "anthropic" else "anthropic"
    message = AssistantMessage(
        role="assistant",
        protocol=owner,
        content=[
            ReasoningContent(
                reasoning="reason",
                feedback_item={"type": "thinking", "thinking": "reason", "signature": "signed"},
            )
        ],
    )
    with pytest.raises(SemanticLossError):
        MyCodec().decode([message], target_format=target)


@pytest.mark.parametrize("target", ["anthropic", "openai_responses"])
@pytest.mark.parametrize("owner", ["message", "context"])
def test_native_feedback_uses_the_message_or_source_context(target, owner):
    from my_llm import AssistantMessage, ReasoningContent

    feedback = {
        "anthropic": {"type": "thinking", "thinking": "reason", "signature": "signed"},
        "openai_responses": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "reason"}],
            "encrypted_content": "opaque",
        },
    }[target]
    message = AssistantMessage(
        role="assistant",
        protocol=target if owner == "message" else None,
        content=[ReasoningContent(reasoning="reason", feedback_item=feedback)],
    )
    result = MyCodec().decode([message], target_format=target, source_protocol=target)
    assert "reason" in str(result)
    assert (
        "opaque"
        if target == "openai_responses"
        else "signed"
        if target == "anthropic"
        else "c2lnbmVk"
    ) in str(result)


@pytest.mark.parametrize("target", PROTOCOLS)
def test_foreign_feedback_owner_is_checked_on_public_response_blocks(target):
    block = {
        "type": "thinking",
        "reasoning": "reason",
        "feedback_item": {"type": "thinking", "thinking": "reason", "signature": "signed"},
    }
    with pytest.raises(SemanticLossError):
        MyCodec().decode(
            chunks=[{"type": "block_stop", "index": 0, "block": block}],
            target_format=target,
            source_protocol="openai_responses" if target == "anthropic" else "anthropic",
        )
