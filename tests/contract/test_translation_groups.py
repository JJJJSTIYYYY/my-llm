"""Extensions follow translation groups when native cardinalities differ."""

import json
from copy import deepcopy

import pytest

from my_llm import ModelProfile, MyCodec, ProviderProfile, resolve_profiles
from my_llm.codec.shared.conversion import REPLAY
from my_llm.codec.shared.replay import merge_extensions


def persisted(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize("target", ["openai_chat", "anthropic", "gemini", "ollama"])
def test_grouped_assistant_extensions_cannot_move_to_the_following_user(target):
    original = [
        {"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": "think"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
            "name": "assistant-identity",
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "next"}],
            "name": "user-identity",
        },
    ]
    before = deepcopy(original)
    codec = MyCodec()
    converted = codec.convert(original, input_format="openai_responses", target_format=target)[
        "messages"
    ]
    assert len(converted) == 2
    assert converted[0]["name"] == "assistant-identity"
    assert converted[1]["name"] == "user-identity"
    restored = codec.convert(
        persisted(converted), input_format=target, target_format="openai_responses"
    )["messages"]
    assert restored == original == before
    if target == "openai_chat":
        profile = resolve_profiles(
            ProviderProfile(protocol=target, api_key="local-test"), ModelProfile(model="demo")
        )
        request = codec.request(codec.encode(persisted(converted), input_format=target), profile)
        assert [message["name"] for message in request["messages"]] == [
            "assistant-identity",
            "user-identity",
        ]
        assert REPLAY not in str(request)


def test_expanded_assistant_extensions_cannot_move_to_another_output_item():
    original = [
        {
            "role": "assistant",
            "vendor_turn": "assistant-only",
            "content": [
                {"type": "thinking", "thinking": "summary", "signature": "signed"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "c1", "name": "f", "input": {}},
            ],
        },
        {
            "role": "user",
            "vendor_turn": "user-only",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "result"}],
        },
    ]
    codec = MyCodec()
    converted = codec.convert(original, input_format="anthropic", target_format="openai_responses")[
        "messages"
    ]
    assert [message.get("vendor_turn") for message in converted] == [
        "assistant-only",
        None,
        "user-only",
    ]
    assert [message["type"] for message in converted] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert converted[-1]["type"] == "function_call_output"
    assert (
        codec.convert(
            persisted(converted), input_format="openai_responses", target_format="anthropic"
        )["messages"]
        == original
    )


@pytest.mark.parametrize(
    "target", ["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"]
)
def test_encrypted_only_turn_keeps_a_recovery_carrier(target):
    original = [
        {
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "encrypted-only"}],
        }
    ]
    codec = MyCodec()
    converted = codec.convert(original, input_format="anthropic", target_format=target)["messages"]
    assert converted
    assert (
        codec.convert(persisted(converted), input_format=target, target_format="anthropic")[
            "messages"
        ]
        == original
    )


@pytest.mark.parametrize("target", ["openai_chat", "openai_responses", "anthropic", "ollama"])
def test_expanded_tool_bundle_extensions_stay_with_their_own_group(target):
    original = [
        {
            "vendor_bundle": "first",
            "function_declarations": [{"name": "a"}, {"name": "b"}],
        },
        {"vendor_bundle": "second", "function_declarations": [{"name": "c"}]},
    ]
    codec = MyCodec()
    converted = codec.convert(tools=original, input_format="gemini", target_format=target)["tools"]
    assert [tool.get("vendor_bundle") for tool in converted] == ["first", None, "second"]
    assert (
        codec.convert(tools=persisted(converted), input_format=target, target_format="gemini")[
            "tools"
        ]
        == original
    )


def test_undeclared_payload_is_atomic_regardless_of_its_field_name():
    extension = {
        "arbitrary_payload": {"injected": True},
        "container": {"vendor": "keep", "arbitrary_payload": {"injected": True}},
    }
    translated = {
        "arbitrary_payload": {"provided": 1},
        "container": {"arbitrary_payload": {"provided": 2}},
    }
    assert merge_extensions(extension, translated, {"container": {}}) == {
        "arbitrary_payload": {"provided": 1},
        "container": {"vendor": "keep", "arbitrary_payload": {"provided": 2}},
    }


def test_coalescing_preserves_ambiguous_block_extensions_without_moving_them():
    original = [
        {"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": "think"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer", "vendor_block": "answer-only"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "next", "vendor_block": "user-only"}],
        },
    ]
    codec = MyCodec()
    converted = codec.convert(original, input_format="openai_responses", target_format="anthropic")[
        "messages"
    ]
    assert converted[0]["content"][0]["text"] == "think"
    assert "vendor_block" not in converted[0]["content"][0]
    assert converted[1]["content"][0]["vendor_block"] == "user-only"
    assert (
        codec.convert(
            persisted(converted), input_format="anthropic", target_format="openai_responses"
        )["messages"]
        == original
    )


def test_filtering_a_block_preserves_extensions_without_reindexing_them():
    original = [
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "encrypted"},
                {"type": "text", "text": "answer", "vendor_block": "answer-only"},
            ],
        }
    ]
    codec = MyCodec()
    converted = codec.convert(original, input_format="anthropic", target_format="openai_responses")[
        "messages"
    ]
    assert converted[0]["content"][0]["type"] == "output_text"
    assert "vendor_block" not in converted[0]["content"][0]
    assert (
        codec.convert(
            persisted(converted), input_format="openai_responses", target_format="anthropic"
        )["messages"]
        == original
    )


def test_one_to_one_block_translation_preserves_extensions_at_the_same_position():
    original = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first", "vendor_block": "first-only"},
                {"type": "text", "text": "second", "vendor_block": "second-only"},
            ],
        }
    ]
    converted = MyCodec().convert(
        original, input_format="anthropic", target_format="openai_responses"
    )["messages"]
    assert [block["vendor_block"] for block in converted[0]["content"]] == [
        "first-only",
        "second-only",
    ]
