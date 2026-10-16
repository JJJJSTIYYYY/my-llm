"""Conversion semantics and extension retention, using independent native fixtures.

Vendor/opaque extension assertions do not claim acceptance by native APIs.
Official schema conformance is checked in test_protocol_conformance.py.
"""

from copy import deepcopy

import pytest
from conftest import CHAT_HISTORY, PROTOCOLS, history, tools

from my_llm import MyCodec
from my_llm.codec.utils.validation import validate_history


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_every_message_conversion_preserves_text_and_tool_associations(source, target):
    codec = MyCodec()
    incoming = history(source)
    before = deepcopy(incoming)
    converted = codec.convert(incoming, input_format=source, target_format=target)["messages"]
    intermediate = codec.encode(converted, input_format=target)["messages"]
    validate_history(intermediate)
    chat = codec.decode(intermediate, target_format="openai_chat")["messages"]
    assert chat[0]["content"] == "Weather?"
    assistant = next(m for m in chat if m.get("tool_calls"))
    result = next(m for m in chat if m["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == result["tool_call_id"]
    assert assistant["tool_calls"][0]["function"]["name"] == "weather"
    assert "Sunny" in str(result["content"])
    assert incoming == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_every_tool_conversion_preserves_schema(source, target):
    codec = MyCodec()
    converted = codec.convert(tools=tools(source), input_format=source, target_format=target)[
        "tools"
    ]
    normalized = codec.encode(tools=converted, input_format=target)["tools"]
    chat = codec.decode(tools=normalized, target_format="openai_chat")["tools"]
    assert chat[0]["function"]["name"] == "weather"
    assert chat[0]["function"]["parameters"]["properties"]["city"] == {"type": "string"}


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_native_round_trip_preserves_arbitrary_fields_and_nested_blocks(protocol):
    codec = MyCodec()
    incoming = history(protocol)
    incoming[1]["future"] = {"nested": [{"new": True}]}
    before = deepcopy(incoming)
    result = codec.convert(incoming, input_format=protocol, target_format=protocol)["messages"]
    assert result == before
    result[1]["future"]["nested"][0]["new"] = False
    assert incoming == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unknown_tool_round_trip(protocol):
    value = [{"type": "future_tool", "configuration": {"nested": [1, 2]}}]
    assert (
        MyCodec().convert(tools=value, input_format=protocol, target_format=protocol)["tools"]
        == value
    )


def test_signed_anthropic_and_gemini_replay():
    codec = MyCodec()
    for protocol, message in (
        (
            "anthropic",
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "signed"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            },
        ),
        (
            "gemini",
            {
                "role": "model",
                "parts": [{"text": "hmm", "thought": True, "thought_signature": "c2lnbmVk"}],
            },
        ),
    ):
        assert codec.convert([message], input_format=protocol, target_format=protocol)[
            "messages"
        ] == [message]


def test_unknown_content_stays_at_its_original_level_across_protocols():
    message = {
        "role": "assistant",
        "content": [{"type": "future", "nested": {"value": [1, 2]}}],
        "extension": {"a": 1},
    }
    result = MyCodec().convert([message], input_format="openai_chat", target_format="anthropic")[
        "messages"
    ][0]
    assert result["content"] == message["content"]
    assert result["extension"] == message["extension"]


def test_converter_can_be_reused_for_unrelated_paths():
    codec = MyCodec()
    for source, target in zip(PROTOCOLS, reversed(PROTOCOLS)):
        assert codec.convert(history(source), input_format=source, target_format=target)["messages"]
    assert not hasattr(codec, "profile")
    assert CHAT_HISTORY[0]["content"] == "Weather?"


@pytest.mark.parametrize("target", PROTOCOLS)
def test_unknown_responses_item_keeps_root_hierarchy(target):
    item = {"type": "future_item", "id": "opaque_1", "configuration": {"nested": [{"value": 1}]}}
    result = MyCodec().convert([item], input_format="openai_responses", target_format=target)[
        "messages"
    ]
    assert result == [item]


@pytest.mark.parametrize("target", PROTOCOLS)
def test_unknown_tool_wrapper_extension_is_preserved(target):
    incoming = tools("openai_chat")
    incoming[0]["wrapper_extension"] = {"nested": [1, 2]}
    result = MyCodec().convert(tools=incoming, input_format="openai_chat", target_format=target)[
        "tools"
    ]
    assert result[0]["wrapper_extension"] == {"nested": [1, 2]}


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_base64_images_survive_every_conversion(source, target):
    codec = MyCodec()
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ],
        }
    ]
    native = codec.convert(chat, input_format="openai_chat", target_format=source)["messages"]
    result = codec.convert(native, input_format=source, target_format=target)["messages"]
    restored = codec.convert(result, input_format=target, target_format="openai_chat")["messages"]
    assert "data:image/png;base64,aGVsbG8=" in str(restored)


def test_chat_image_detail_moves_to_responses_field():
    incoming = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/p.png", "detail": "low"},
                }
            ],
        }
    ]
    result = MyCodec().convert(
        incoming, input_format="openai_chat", target_format="openai_responses"
    )["messages"]
    assert result[0]["content"] == [
        {"type": "input_image", "image_url": "https://example.test/p.png", "detail": "low"}
    ]


@pytest.mark.parametrize("source", ["openai_chat", "ollama"])
def test_nested_function_extensions_survive_intermediate_and_cross_protocol_conversion(source):
    arguments = {} if source == "ollama" else "{}"
    function = {"name": "weather", "arguments": arguments, "future": {"nested": [{"flag": True}]}}
    incoming = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": function}],
        }
    ]
    codec = MyCodec()
    intermediate = codec.encode(incoming, input_format=source)
    block = intermediate["messages"][0]["content"][0]
    assert block["function"] == {"future": function["future"]}
    anthropic = codec.decode(**intermediate, target_format="anthropic")["messages"]
    assert anthropic[0]["content"][0]["function"] == {"future": function["future"]}
    restored = codec.convert(anthropic, input_format="anthropic", target_format=source)["messages"]
    assert restored[0]["tool_calls"][0]["function"]["future"] == function["future"]


def test_gemini_nested_extensions_are_not_flattened():
    codec = MyCodec()
    incoming = [
        {
            "role": "model",
            "parts": [
                {
                    "part_extension": {"value": 1},
                    "function_call": {
                        "id": "call_1",
                        "name": "weather",
                        "args": {},
                        "call_extension": {"nested": [1, 2]},
                    },
                }
            ],
        }
    ]
    intermediate = codec.encode(incoming, input_format="gemini")
    block = intermediate["messages"][0]["content"][0]
    assert block["part_extension"] == {"value": 1}
    assert "call_extension" not in block
    assert block["function_call"] == {"call_extension": {"nested": [1, 2]}}
    anthropic = codec.decode(**intermediate, target_format="anthropic")["messages"]
    restored = codec.convert(anthropic, input_format="anthropic", target_format="gemini")[
        "messages"
    ]
    assert restored == incoming
