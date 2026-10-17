"""Inspect extension paths independently of native replay snapshots.

These preservation fixtures may include vendor or deliberately partial data;
they do not assert native API acceptance. See test_protocol_conformance.py.
"""

import json
from copy import deepcopy

import pytest

from my_llm import MyCodec
from my_llm.codec.shared.conversion import residual
from tests.helpers.protocols import PROTOCOLS, tools

EXT = {"nested": [{"sentinel": "must-survive"}], "empty": {}, "null": None}


def check(protocol, kind, native):
    before = deepcopy(native)
    codec = MyCodec()
    encoded = codec.encode(input_format=protocol, **{kind: native})
    assert native == before
    encoded = json.loads(json.dumps(encoded))
    assert codec.decode(**encoded, target_format=protocol)[kind] == before
    return encoded[kind]


def test_only_recognized_leaves_are_consumed():
    source = {"a": {"b": "v1", "c": EXT}, "list": [{"b": 1}, {"b": 2, "c": EXT}]}
    result = {**residual(source, {"a": {"b": True}, "list": [{"b": True}]}), "d": "v1"}
    assert result == {"d": "v1", "a": {"c": EXT}, "list": [{}, {"c": EXT}]}


@pytest.mark.parametrize("extension", [{}, [], None, False, 0, "", EXT])
def test_empty_and_scalar_extensions_are_not_pruned(extension):
    assert residual({"a": {"b": 1, "future": extension}}, {"a": {"b": True}}) == {
        "a": {"future": extension}
    }


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_message_root_and_text_extensions(protocol):
    if protocol == "gemini":
        native = [{"role": "user", "parts": [{"text": "hello", "future": EXT}], "future": EXT}]
        path = "parts"
    else:
        native = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "future": EXT}],
                "future": EXT,
            }
        ]
        path = "content"
    intermediate = check(protocol, "messages", native)
    assert intermediate[0]["future"] == EXT
    assert intermediate[0][path][0]["future"] == EXT


@pytest.mark.parametrize("protocol", ["openai_chat", "ollama"])
def test_tool_call_extensions_stay_at_original_message_path(protocol):
    native = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {
                    "id": "c2",
                    "type": "function",
                    "future": EXT,
                    "function": {"name": "g", "arguments": {}, "future": EXT},
                },
            ],
        }
    ]
    value = check(protocol, "messages", native)[0]
    assert value["tool_calls"] == [{}, {"future": EXT, "function": {"future": EXT}}]
    assert value["content"][1]["name"] == "g"
    assert value["content"][1]["input"] == {}


@pytest.mark.parametrize(
    "container,payload",
    [
        ("inline_data", {"mime_type": "image/png", "data": "aGVsbG8="}),
        ("file_data", {"mime_type": "application/pdf", "file_uri": "https://example.com/a.pdf"}),
        ("function_call", {"id": "c1", "name": "f", "args": {"x": EXT}}),
        ("function_response", {"id": "c1", "name": "f", "response": {"x": EXT}}),
    ],
)
def test_gemini_nested_extensions_keep_native_parts_path(container, payload):
    native = [
        {
            "role": "user",
            "parts": [
                {"text": "first"},
                {container: {**payload, "future": EXT}, "future": EXT},
            ],
        }
    ]
    value = check("gemini", "messages", native)[0]
    assert value["parts"] == [{}, {container: {"future": EXT}, "future": EXT}]
    assert value["content"][1][container] == {"future": EXT}


@pytest.mark.parametrize(
    "kind,container,data",
    [
        ("image_url", "image_url", {"url": "https://example.com/a.png", "detail": "high"}),
        ("file", "file", {"file_data": "data:application/pdf;base64,YQ=="}),
    ],
)
def test_chat_media_extensions_are_not_flattened(kind, container, data):
    native = [{"role": "user", "content": [{"type": kind, container: {**data, "future": EXT}}]}]
    value = check("openai_chat", "messages", native)[0]["content"][0]
    assert value[container] == {"future": EXT}
    assert "future" not in value
    assert value["source"]["type"] in ("url", "base64")


def test_responses_does_not_merge_source_messages_or_drop_item_extensions():
    native = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"logprobs": [], "type": "output_text", "text": str(i)}],
            "future": {"item": i},
        }
        for i in range(2)
    ]
    native.append(
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}", "future": EXT}
    )
    value = check("openai_responses", "messages", native)
    assert len(value) == 3
    assert [item["future"] for item in value] == [{"item": 0}, {"item": 1}, EXT]
    # Grouping remains an output concern when a target needs a single assistant message.
    chat = MyCodec().decode(messages=value, target_format="openai_chat")["messages"]
    assert len(chat) == 1
    assert chat[0]["content"] == "01"
    assert chat[0]["tool_calls"][0]["function"]["name"] == "f"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_tool_extensions_keep_original_definition_paths(protocol):
    native = tools(protocol)
    native[0]["future"] = EXT
    definition = (
        native[0]["function"]
        if protocol in ("openai_chat", "ollama")
        else native[0]["function_declarations"][0]
        if protocol == "gemini"
        else native[0]
    )
    definition["definition_future"] = EXT
    value = check(protocol, "tools", native)[0]
    assert value["future"] == EXT
    actual = (
        value["function"]
        if protocol in ("openai_chat", "ollama")
        else value["function_declarations"][0]
        if protocol == "gemini"
        else value
    )
    assert actual["definition_future"] == EXT
    assert value["__my_llm_codec_function_definitions__"][0]["parameters"]["properties"][
        "city"
    ] == {"type": "string"}


CHUNKS = [
    (
        "openai_chat",
        {"choices": [{"index": 0, "delta": {"content": "x", "future": EXT}}]},
        ("choices", 0, "delta"),
    ),
    (
        "openai_chat",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "f", "arguments": "{}", "future": EXT},
                            }
                        ]
                    },
                }
            ]
        },
        ("choices", 0, "delta", "tool_calls", 0, "function"),
    ),
    (
        "anthropic",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "x", "future": EXT},
        },
        ("delta",),
    ),
    (
        "anthropic",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "image",
                "source": {"type": "url", "url": "u", "future": EXT},
            },
        },
        ("content_block", "source"),
    ),
    (
        "gemini",
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": "YQ==",
                                    "future": EXT,
                                }
                            }
                        ]
                    }
                }
            ]
        },
        ("candidates", 0, "content", "parts", 0, "inline_data"),
    ),
    (
        "ollama",
        {
            "message": {
                "content": "x",
                "tool_calls": [{"function": {"name": "f", "arguments": {}, "future": EXT}}],
            }
        },
        ("message", "tool_calls", 0, "function"),
    ),
    (
        "openai_responses",
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "f",
                        "arguments": "{}",
                        "future": EXT,
                    }
                ]
            },
        },
        ("response", "output", 0),
    ),
]


@pytest.mark.parametrize("protocol,event,path", CHUNKS)
def test_stream_extensions_stay_at_original_event_paths(protocol, event, path):
    native = [{**event, "future": EXT, "events": {"native_extension": EXT}}]
    value = check(protocol, "chunks", native)[0]
    assert value["future"] == EXT
    assert value["events"] == {"native_extension": EXT}
    for component in path:
        value = value[component]
    assert value["future"] == EXT


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unknown_event_is_visible_without_opening_snapshot(protocol):
    event = {"type": "response.future.event", "events": [{"x": EXT}], "future": EXT}
    value = check(protocol, "chunks", [event])[0]
    assert {key: value[key] for key in event} == event


@pytest.mark.parametrize("protocol", ["openai_chat", "ollama"])
def test_reasoning_does_not_shift_original_content_indices(protocol):
    field = "thinking" if protocol == "ollama" else "reasoning_content"
    native = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "x", "future": EXT}],
            field: "thinking",
        }
    ]
    value = check(protocol, "messages", native)[0]
    assert value["content"][0]["future"] == EXT


def test_translated_field_wins_over_preserved_extension():
    native = [{"role": "user", "parts": [{"text": "hello"}], "content": EXT}]
    value = check("gemini", "messages", native)[0]
    assert value["content"] == EXT
    assert value["__my_llm_codec_translated_fields__"]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    result = MyCodec().decode(messages=[value], target_format="openai_chat")["messages"]
    assert result[0]["content"] == "hello"


@pytest.mark.parametrize(
    "protocol,event,path",
    [
        ("openai_chat", {"usage": {"total_tokens": 1, "future": EXT}}, ("usage",)),
        (
            "anthropic",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1, "future": EXT},
            },
            ("usage",),
        ),
        (
            "gemini",
            {"usage_metadata": {"total_token_count": 1, "future": EXT}},
            ("usage_metadata",),
        ),
        (
            "openai_responses",
            {
                "type": "response.completed",
                "response": {"output": [], "usage": {"total_tokens": 1, "future": EXT}},
            },
            ("response", "usage"),
        ),
    ],
)
def test_native_usage_extensions_keep_their_paths(protocol, event, path):
    value = check(protocol, "chunks", [event])[0]
    for component in path:
        value = value[component]
    assert value["future"] == EXT


def test_responses_reasoning_summary_extensions_keep_list_indices():
    native = [
        {
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "first"},
                {"type": "summary_text", "text": "second", "future": EXT},
            ],
        }
    ]
    value = check("openai_responses", "messages", native)[0]
    assert value["summary"] == [{}, {"future": EXT}]
    assert value["content"][0]["thinking"] == "firstsecond"


def test_anthropic_tool_schema_is_atomic_but_unknown_parameter_sibling_is_not():
    native = [{"name": "f", "input_schema": {"type": "object", "future": EXT}, "parameters": EXT}]
    value = check("anthropic", "tools", native)[0]
    assert value["parameters"] == EXT
    assert (
        value["__my_llm_codec_function_definitions__"][0]["parameters"] == native[0]["input_schema"]
    )
    assert "input_schema" not in value


def test_responses_stream_snapshot_handles_native_content_key_collision():
    native = [
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "f",
                        "arguments": "{}",
                        "content": EXT,
                    },
                ]
            },
        }
    ]
    value = check("openai_responses", "chunks", native)
    assert value[0]["response"]["output"][0]["content"] == EXT
    result = MyCodec().decode(chunks=value, target_format="openai_chat")["chunks"]
    assert result[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_converted_media_source_overrides_conflicting_extension_leaves():
    native = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/a.png"},
                    "source": {"type": "wrong", "url": "wrong", "future": EXT},
                }
            ],
        }
    ]
    value = check("openai_chat", "messages", native)[0]
    assert value["content"][0]["source"] == {"type": "wrong", "url": "wrong", "future": EXT}
    assert value["__my_llm_codec_translated_fields__"]["content"][0]["source"] == {
        "type": "url",
        "url": "https://example.com/a.png",
    }
    gemini = MyCodec().decode(messages=[value], target_format="gemini")["messages"]
    assert gemini[0]["parts"][0]["file_data"]["file_uri"] == "https://example.com/a.png"


def test_function_call_extension_cannot_override_converted_assistant_role():
    native = [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "f",
            "arguments": "{}",
            "role": "user",
            "content": EXT,
        }
    ]
    value = check("openai_responses", "messages", native)[0]
    assert value["role"] == "user"
    assert value["content"] == EXT
    translated = value["__my_llm_codec_translated_fields__"]
    assert translated["role"] == "assistant"
    assert translated["content"][0]["type"] == "tool_use"
    result = MyCodec().decode(messages=[value], target_format="openai_chat")["messages"]
    assert result[0]["role"] == "assistant"
    assert result[0]["tool_calls"][0]["function"]["name"] == "f"


@pytest.mark.parametrize("target", ["openai_chat", "openai_responses", "anthropic", "ollama"])
def test_each_target_reads_namespaced_translation_instead_of_native_collision(target):
    value = check(
        "gemini", "messages", [{"role": "user", "parts": [{"text": "hello"}], "content": EXT}]
    )
    result = MyCodec().decode(messages=value, target_format=target)["messages"]
    content = result[0]["content"]
    if isinstance(content, list):
        assert content[0]["text"] == "hello"
    else:
        assert content == "hello"
    assert "__my_llm_codec_" not in json.dumps(result)


def test_history_validation_reads_converted_role_and_content():
    from my_llm.codec.shared.validation import validate_history

    native = [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "f",
            "arguments": "{}",
            "role": "user",
            "content": EXT,
        },
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ]
    value = check("openai_responses", "messages", native)
    validate_history(value)


def test_short_extension_names_are_not_treated_as_internal_control_keys():
    from my_llm.codec.shared.validation import validate_history
    from my_llm.exceptions import MessageValidationError

    tool = {"type": "function", "function": {"name": "f", "parameters": {}}, "__functions__": EXT}
    value = check("openai_chat", "tools", [tool])[0]
    assert value["__functions__"] == EXT
    assert value["__my_llm_codec_function_definitions__"][0]["name"] == "f"

    native = [
        {
            "role": "assistant",
            "content": "",
            "__opaque_message__": True,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    ]
    value = check("openai_chat", "messages", native)
    assert value[0]["__opaque_message__"] is True
    with pytest.raises(MessageValidationError, match="exactly one result"):
        validate_history(value)


@pytest.mark.parametrize(
    "extension",
    [
        [{"type": "text", "text": "wrong"}, {"type": "text", "text": "extra"}],
        {"wrong": [1, 2]},
    ],
)
def test_translated_content_is_not_extended_by_preserved_collision(extension):
    value = check(
        "gemini", "messages", [{"role": "user", "parts": [{"text": "hello"}], "content": extension}]
    )
    assert value[0]["content"] == extension
    result = MyCodec().decode(messages=value, target_format="openai_chat")["messages"]
    assert result == [{"role": "user", "content": "hello"}]


def test_preserved_content_cannot_add_arguments_to_a_translated_call():
    native = [
        {
            "role": "model",
            "parts": [{"function_call": {"id": "c1", "name": "f", "args": {"known": 1}}}],
            "content": [
                {"type": "tool_use", "id": "wrong", "name": "wrong", "input": {"injected": True}}
            ],
        }
    ]
    value = check("gemini", "messages", native)
    result = MyCodec().decode(messages=value, target_format="openai_chat")["messages"]
    call = result[0]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"]["name"] == "f"
    assert json.loads(call["function"]["arguments"]) == {"known": 1}


@pytest.mark.parametrize(
    "kind,container,data,extension_key,extension",
    [
        (
            "image_url",
            "image_url",
            {"url": "https://example.com/a.png"},
            "file",
            {"file_data": "unknown"},
        ),
        (
            "file",
            "file",
            {"file_data": "data:application/pdf;base64,YQ=="},
            "image_url",
            {"url": "unknown"},
        ),
    ],
)
def test_media_only_consumes_fields_recognized_for_its_discriminator(
    kind, container, data, extension_key, extension
):
    native = [
        {"role": "user", "content": [{"type": kind, container: data, extension_key: extension}]}
    ]
    value = check("openai_chat", "messages", native)[0]
    assert value["content"][0][extension_key] == extension


@pytest.mark.parametrize("source", ["openai_chat", "ollama"])
def test_message_call_index_is_not_consumed_as_stream_metadata(source):
    native = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "index": EXT,
                    "function": {"name": "f", "arguments": {}},
                }
            ],
        }
    ]
    value = check(source, "messages", native)[0]
    assert value["tool_calls"][0]["index"] == EXT
