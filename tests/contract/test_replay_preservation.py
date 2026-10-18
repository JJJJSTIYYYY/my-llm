"""Preserved public conversions and filtered requests have different contracts."""

import json
from copy import deepcopy

import pytest

from my_llm import ModelProfile, MyCodec, MyLLM, ProviderProfile, resolve_profiles
from my_llm.codec.shared.conversion import REPLAY
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools
from tests.helpers.schema import validate_request
from tests.helpers.transport import FakeClient


def profile(protocol):
    return resolve_profiles(
        ProviderProfile(protocol=protocol, api_key="local-test"),
        ModelProfile(model="demo", max_tokens=4096, supports_function_calling=True),
    )


def persisted(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_extensions_survive_saved_conversions_and_are_filtered_only_in_transport(
    source, target, kind
):
    codec = MyCodec()
    original = history(source) if kind == "messages" else tools(source)
    original[0]["vendor_metadata"] = {"null": None, "nested": [{"unknown": [1, 2]}]}
    before = deepcopy(original)
    converted = codec.convert(**{kind: original}, input_format=source, target_format=target)[kind]
    assert converted[0]["vendor_metadata"] == before[0]["vendor_metadata"]
    restored = MyCodec().convert(
        **{kind: persisted(converted)}, input_format=target, target_format=source
    )[kind]
    assert restored == before
    if kind == "messages":
        intermediate = codec.encode(persisted(converted), input_format=target)
    else:
        intermediate = codec.encode(
            history(target), tools=persisted(converted), input_format=target
        )
    request = codec.request(intermediate, profile(target))
    validate_request(
        target,
        request
        if target != "gemini"
        else {**request, "config": request["config"].model_dump(mode="json", exclude_none=True)},
    )
    assert "vendor_metadata" not in str(request)
    assert REPLAY not in str(request)
    assert original == before


SIGNED = [
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "summary", "signature": "opaque-signature"},
            {"type": "redacted_thinking", "data": "encrypted-data"},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "上海"}},
        ],
    }
]


@pytest.mark.parametrize("target", [p for p in PROTOCOLS if p != "anthropic"])
def test_signed_tool_turn_can_be_restored_after_json_persistence_and_an_appended_result(target):
    codec = MyCodec()
    converted = codec.convert(SIGNED, input_format="anthropic", target_format=target)["messages"]
    restored = codec.convert(persisted(converted), input_format=target, target_format="anthropic")[
        "messages"
    ]
    assert restored == SIGNED
    result = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny"}],
        }
    ]
    converted.extend(
        codec.convert(result, input_format="anthropic", target_format=target)["messages"]
    )
    request = codec.request(
        codec.encode(persisted(converted), input_format=target), profile("anthropic")
    )
    assert request["messages"][0] == SIGNED[0]
    assert request["messages"][1]["content"][0]["tool_use_id"] == "call_1"
    assert "Sunny" in str(request["messages"][1])
    assert REPLAY not in str(request)


def test_visible_edits_do_not_restore_old_signed_blocks():
    codec = MyCodec()
    converted = codec.convert(SIGNED, input_format="anthropic", target_format="openai_chat")[
        "messages"
    ]
    converted[0]["content"] = "edited by user"
    converted[0]["tool_calls"][0]["function"]["arguments"] = '{"city":"北京"}'
    restored = codec.convert(
        persisted(converted), input_format="openai_chat", target_format="anthropic"
    )["messages"]
    assert "edited by user" in str(restored)
    assert "opaque-signature" not in str(restored)
    assert "encrypted-data" not in str(restored)
    assert any(block.get("input") == {"city": "北京"} for block in restored[0]["content"])


@pytest.mark.parametrize("summary", [[], [{"type": "summary_text", "text": "summary"}]])
def test_encrypted_responses_items_survive_chat_without_entering_chat_requests(summary):
    original = [
        {"type": "reasoning", "id": "rs_1", "summary": summary, "encrypted_content": "ciphertext"},
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
        },
    ]
    codec = MyCodec()
    converted = codec.convert(
        original, input_format="openai_responses", target_format="openai_chat"
    )["messages"]
    assert (
        codec.convert(
            persisted(converted), input_format="openai_chat", target_format="openai_responses"
        )["messages"]
        == original
    )
    encoded = codec.encode(persisted(converted), input_format="openai_chat")
    assert codec.request(encoded, profile("openai_responses"))["input"] == original
    request = codec.request(encoded, profile("openai_chat"))
    assert "ciphertext" not in str(request)
    assert REPLAY not in str(request)


async def test_stream_result_replays_original_thinking_on_the_next_request(monkeypatch):
    events = stream_events("anthropic")
    events[1]["content_block"] = {"type": "thinking", "thinking": "", "signature": ""}
    events[2]["delta"] = {"type": "thinking_delta", "thinking": "summary"}
    events.insert(
        3,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opaque-signature"},
        },
    )
    client = FakeClient(events)
    monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", lambda **kwargs: client)
    async with MyLLM(
        provider=ProviderProfile(protocol="anthropic", api_key="local-test"),
        model=ModelProfile(model="demo", max_tokens=4096),
        stdio_protocol="openai_chat",
    ) as llm:
        messages = [{"role": "user", "content": "hello"}]
        stream = await llm.generate(messages=messages, stream=True)
        async with stream:
            async for _ in stream:
                pass
        messages.extend([persisted(stream.result), {"role": "user", "content": "continue"}])
        await llm.generate(messages=messages)
    assert client.requests[-1]["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "summary", "signature": "opaque-signature"}
    ]
    assert REPLAY not in str(client.requests)


def test_translation_and_sdk_validation_have_separate_interfaces():
    for codec in MyCodec().codecs.values():
        for direction in (codec.encoder, codec.decoder):
            for name in (
                "validate_profile",
                "validate_parameters",
                "validate_request",
                "validate_input",
                "filter_messages",
                "filter_tools",
                "create_client",
                "upstream",
            ):
                assert not hasattr(direction, name)
        assert callable(codec.decoder.request_parameters)
        assert callable(codec.decoder.assemble_request)
        for name in ("request_parameters", "assemble_request", "profile_defaults"):
            assert not hasattr(codec.transport, name)
        assert callable(codec.transport.request)
        assert callable(codec.transport.validate_profile)
        assert callable(codec.transport.validate_request)


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_complete_stream_batches_preserve_extensions_and_native_events(source, target):
    original = stream_events(source, tool=True)
    original[0]["vendor_event"] = {"nested": [None, {}, {"value": 1}]}
    codec = MyCodec()
    converted = codec.convert(chunks=original, input_format=source, target_format=target)["chunks"]
    restored = MyCodec().convert(
        chunks=persisted(converted), input_format=target, target_format=source
    )["chunks"]
    assert restored == original


def test_preserved_collision_cannot_add_tool_arguments_to_public_translations():
    codec = MyCodec()
    incoming = [
        {
            "role": "model",
            "parts": [{"function_call": {"id": "c1", "name": "f", "args": {"known": 1}}}],
            "content": [
                {"type": "tool_use", "id": "wrong", "name": "wrong", "input": {"injected": True}}
            ],
        }
    ]
    converted = codec.convert(incoming, input_format="gemini", target_format="anthropic")[
        "messages"
    ]
    assert converted[0]["content"][0]["input"] == {"known": 1}


@pytest.mark.parametrize("target", PROTOCOLS)
def test_decoder_retains_extensions_even_without_an_encoder_snapshot(target):
    intermediate = [
        {
            "role": "user",
            "vendor_root": {"keep": [1, None]},
            "content": [{"type": "text", "text": "hello", "vendor_block": {"keep": True}}],
        }
    ]
    before = deepcopy(intermediate)
    converted = MyCodec().decoder(target).messages(intermediate)
    assert converted[0]["vendor_root"] == intermediate[0]["vendor_root"]
    encoded = MyCodec().encode(persisted(converted), input_format=target)["messages"]
    assert encoded == before
    assert intermediate == before


def test_transport_filters_protocol_extensions_without_pruning_application_data():
    codec = MyCodec()
    data = {
        "type": "application-data",
        "signature": "application-signature",
        REPLAY: {"user_key": [1, 2]},
    }
    messages = [
        {
            "role": "assistant",
            "vendor_root": True,
            "content": [
                {"type": "tool_use", "id": "c1", "name": "f", "input": data, "vendor_block": True}
            ],
        }
    ]
    filtered = codec.transport("anthropic").filter_messages(messages, profile("anthropic"))
    assert filtered == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c1", "name": "f", "input": data}],
        }
    ]
    schema = {
        "type": "object",
        "properties": {"signature": {"type": "string"}, REPLAY: {"type": "object"}},
        "additionalProperties": False,
    }
    tools = [
        {
            "type": "function",
            "function": {"name": "f", "parameters": schema, "vendor_function": True},
            "vendor_root": True,
        }
    ]
    assert codec.transport("openai_chat").filter_tools(tools, profile("openai_chat")) == [
        {"type": "function", "function": {"name": "f", "parameters": schema}}
    ]


@pytest.mark.parametrize("edit", ["later_item", "reorder", "truncate"])
def test_editing_any_item_invalidates_the_entire_signed_conversion_group(edit):
    codec = MyCodec()
    converted = codec.convert(SIGNED, input_format="anthropic", target_format="openai_responses")[
        "messages"
    ]
    assert len(converted) > 1
    if edit == "later_item":
        converted[-1]["arguments"] = '{"city":"北京"}'
    elif edit == "reorder":
        converted.reverse()
    else:
        converted.pop()
    encoded = codec.encode(persisted(converted), input_format="openai_responses")["messages"]
    from my_llm.codec.shared.conversion import source_protocol

    assert all(source_protocol(value) == "openai_responses" for value in encoded)
    native = codec.decoder("anthropic").messages(encoded)
    filtered = codec.transport("anthropic").filter_messages(native, profile("anthropic"))
    assert "opaque-signature" not in str(filtered)
    assert "encrypted-data" not in str(filtered)
    assert not any(
        block["type"] in ("thinking", "redacted_thinking")
        for message in filtered
        for block in message["content"]
    )
