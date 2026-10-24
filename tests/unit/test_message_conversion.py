"""Recognized conversion semantics, using independent native fixtures."""

from copy import deepcopy

import pytest

from my_llm import MyCodec
from my_llm.codec.shared.conversion import validate_history
from tests.helpers.protocols import CHAT_HISTORY, PROTOCOLS, history, tools


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


def test_converter_can_be_reused_for_unrelated_paths():
    codec = MyCodec()
    for source, target in zip(PROTOCOLS, reversed(PROTOCOLS)):
        assert codec.convert(history(source), input_format=source, target_format=target)["messages"]
    assert not hasattr(codec, "profile")
    assert CHAT_HISTORY[0]["content"] == "Weather?"


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_base64_images_survive_every_conversion(source, target):
    codec = MyCodec()
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
                    },
                },
            ],
        }
    ]
    native = codec.convert(chat, input_format="openai_chat", target_format=source)["messages"]
    result = codec.convert(native, input_format=source, target_format=target)["messages"]
    restored = codec.convert(result, input_format=target, target_format="openai_chat")["messages"]
    assert (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        in str(restored)
    )


def test_chat_image_detail_is_not_part_of_the_supported_field_set():
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
        {"type": "input_image", "image_url": "https://example.test/p.png", "detail": "auto"}
    ]
