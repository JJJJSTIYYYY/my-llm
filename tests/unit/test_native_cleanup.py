"""Native adapters own supported payloads once and discard unknown output locally."""

from copy import deepcopy

import pytest

from my_llm import MyCodec


@pytest.mark.parametrize("kind", ["messages", "chunks"])
def test_ollama_function_index_and_unknown_fields_do_not_enter_middleware(kind):
    class Unreadable:
        def __deepcopy__(self, memo):
            raise AssertionError("read unsupported native tool field")

    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "lookup",
                    "arguments": {"value": 1},
                    "index": 7,
                    "vendor": Unreadable(),
                },
                "vendor": Unreadable(),
            }
        ],
    }
    values = [message] if kind == "messages" else [{"message": message}, {"done": True}]
    codec = MyCodec()
    middleware = codec.encode(**{kind: values}, input_format="ollama")
    tool = (
        middleware.messages[0].content[-1].to_wire()
        if kind == "messages"
        else middleware.stream[0].events[-1]["block"]
    )
    assert set(tool) == {"type", "id", "name", "input"} | (
        {"protocol"} if kind == "messages" else set()
    )
    assert tool["name"] == "lookup"
    assert tool["input"] == {"value": 1}

    restored = codec.convert(**{kind: values}, input_format="ollama", target_format="ollama")
    native_messages = (
        restored["messages"]
        if kind == "messages"
        else [event["message"] for event in restored["chunks"]]
    )
    calls = [call for value in native_messages for call in value.get("tool_calls", [])]
    assert calls == [{"function": {"name": "lookup", "arguments": {"value": 1}}}]


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_stream_tool_arguments_are_owned_once_at_the_middleware_boundary(protocol):
    copies = []

    class TrackedArguments(dict):
        def __deepcopy__(self, memo):
            copies.append(self)
            return deepcopy(dict(self), memo)

    arguments = TrackedArguments(items=[{"value": 1}])
    if protocol == "anthropic":
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_1",
                "name": "lookup",
                "input": arguments,
            },
        }
    elif protocol == "gemini":
        event = {
            "candidates": [
                {"content": {"parts": [{"function_call": {"name": "lookup", "args": arguments}}]}}
            ]
        }
    else:
        event = {
            "message": {"tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}]}
        }

    middleware = MyCodec().encode(chunks=[event], input_format=protocol)
    tool = middleware.stream[0].events[-1]["block"]
    assert len(copies) == 1
    assert tool["input"] == arguments
    arguments["items"][0]["value"] = 2
    assert tool["input"] == {"items": [{"value": 1}]}
    original = (
        event["content_block"]
        if protocol == "anthropic"
        else event["candidates"][0]["content"]["parts"][0]
        if protocol == "gemini"
        else event
    )
    assert tool.origin_value is original


@pytest.mark.parametrize("protocol", ["anthropic", "gemini"])
def test_native_stream_projection_discards_generated_media_before_middleware(protocol):
    from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
    from my_llm.codec.gemini.encoding.encoder import GeminiEncoder

    if protocol == "anthropic":
        encoder = AnthropicEncoder()
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
                },
            }
        ]
    else:
        encoder = GeminiEncoder()
        events = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inline_data": {"mime_type": "image/png", "data": "aW1hZ2U="}}
                            ]
                        }
                    }
                ]
            }
        ]
    projected = encoder._chunks(events)
    assert all("block" not in operation for batch in projected for operation in batch["events"])


@pytest.mark.parametrize("text", ['{"value": [1, 2]}', "ordinary text", "[1, 2]"])
def test_gemini_prepare_keeps_tool_result_text_without_guessing_json_types(text, profile_factory):
    from my_llm import Config, omit

    codec = MyCodec()
    profile = profile_factory("gemini", supports_function_calling=Config(default_value=True))
    prepared = codec.prepare(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": text},
            ],
            "tools": omit,
            "stream": False,
        },
        profile,
        input_format="openai_chat",
        client=codec.client_parameters(profile),
    )
    assert prepared["request"]["contents"][-1]["parts"][0]["function_response"]["response"] == {
        "result": text
    }


def test_gemini_native_signature_spelling_stays_at_the_codec_boundary():
    codec = MyCodec()
    messages = [
        {
            "role": "model",
            "parts": [
                {"text": "reason", "thought": True, "thought_signature": "c2ln"},
                {
                    "function_call": {"id": "call_1", "name": "lookup", "args": {}},
                    "thought_signature": "c2ln",
                },
            ],
        }
    ]
    middleware = codec.encode(messages=messages, input_format="gemini")
    blocks = middleware["messages"][0]["content"]
    assert all(block["signature"] == "c2ln" for block in blocks)
    assert all("thought_signature" not in block for block in blocks)
    restored = codec.decode(messages=middleware.messages, target_format="gemini")["messages"]
    assert all(part["thought_signature"] == "c2ln" for part in restored[0]["parts"])
    assert all("signature" not in part for part in restored[0]["parts"])
