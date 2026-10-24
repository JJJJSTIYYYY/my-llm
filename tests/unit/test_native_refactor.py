"""Fixed supported fields and loss behavior across the native protocol boundary."""

import json
from copy import deepcopy

import pytest

from my_llm import MyCodec
from my_llm.basic.exceptions import SemanticLossError
from tests.helpers.protocols import PROTOCOLS
from tests.helpers.tool_results import application_result_history, native_result_text


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize(
    "payload", [{"value": [1, {"metadata": "application data"}]}, [1, {"value": [2, 3]}]]
)
def test_object_and_array_tool_results_prepare_as_text_in_every_direction(
    source, target, payload, profile_factory
):
    from my_llm import Config, omit
    from tests.helpers.profiles import parameter

    messages = application_result_history(source, payload)
    before = deepcopy(messages)
    profile = profile_factory(
        target,
        supports_function_calling=Config(default_value=True),
        max_tokens=parameter("max_tokens", 2048, protocol=target),
    )
    codec = MyCodec()
    prepared = codec.prepare(
        {"messages": messages, "tools": omit, "stream": False},
        profile,
        input_format=source,
        client=codec.client_parameters(profile),
    )
    assert json.loads(native_result_text(target, prepared["request"])) == payload
    assert messages == before


@pytest.mark.parametrize("target", PROTOCOLS)
def test_assistant_media_outside_the_supported_output_set_is_discarded(target):
    incoming = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "answer"},
                            {"inline_data": {"mime_type": "image/png", "data": "aW1hZ2U="}},
                            {"inline_data": {"mime_type": "audio/wav", "data": "ignored"}},
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ]
        }
    ]
    output = MyCodec().convert(chunks=incoming, input_format="gemini", target_format=target)
    assert "answer" in str(output)
    assert "aW1hZ2U=" not in str(output)
    assert "ignored" not in str(output)


def test_unsupported_response_metadata_is_never_inspected_or_copied():
    class Unreadable:
        def __deepcopy__(self, memo):
            raise AssertionError("unsupported metadata was inspected")

    output = MyCodec().encode(
        chunks=[
            {
                "type": "response.created",
                "response": {"id": "r1", "model": "demo", "metadata": Unreadable()},
            }
        ],
        input_format="openai_responses",
    )
    assert output["chunks"][0]["events"][0]["values"] == {"response_id": "r1", "model": "demo"}


def test_loss_provenance_does_not_traverse_unsupported_fields():
    class Unreadable:
        def __deepcopy__(self, memo):
            raise AssertionError("unsupported metadata was copied during loss collection")

    image = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/a.png"},
        "metadata": Unreadable(),
    }
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c", "content": [image]}],
        },
    ]
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().convert(messages, input_format="anthropic", target_format="openai_chat")
    assert caught.value.losses.items[0].origin_value is image
    assert caught.value.losses.items[0].origin_path == "$.messages[1].content[0].content[0]"


def test_completed_response_snapshot_reports_the_original_stream_position():
    from tests.helpers.protocols import response_completed

    item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "reason"}],
        "encrypted_content": "opaque",
    }
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().convert(
            chunks=[response_completed([item])],
            input_format="openai_responses",
            target_format="openai_chat",
        )
    assert caught.value.losses.items[0].origin_path == "$.chunks[0].response.output[0]"
    assert caught.value.losses.items[0].origin_value is item


def test_snapshot_source_positions_do_not_shift_when_unsupported_blocks_are_dropped():
    from tests.helpers.protocols import response_completed

    part = {"type": "output_text", "text": "answer"}
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_file", "file_data": "ignored"}, part],
        }
    ]
    encoded = MyCodec().encode(chunks=[response_completed(output)], input_format="openai_responses")
    snapshot = next(
        event for batch in encoded.stream for event in batch.events if event.type == "snapshot"
    )
    block = snapshot.blocks[0][1]
    assert block.origin.path == "$.chunks[0].response.output[0].content[1]"
    assert block.origin.value is part


def test_loss_origin_is_the_original_sdk_object_without_copying_extensions():
    from google.genai import types

    class Unreadable:
        def __deepcopy__(self, memo):
            raise AssertionError("unsupported SDK extension was copied")

    part = types.Part.model_construct(
        text="reason", thought=True, thought_signature=b"sig", part_metadata=Unreadable()
    )
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().convert(
            [types.Content(role="model", parts=[part])],
            input_format="gemini",
            target_format="openai_chat",
        )
    assert caught.value.losses.items[0].origin_value is part
    assert caught.value.losses.items[0].origin_path == "$.messages[0].parts[0]"


UNSUPPORTED_INPUTS = {
    "openai_chat": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep"},
                {"type": "input_audio", "input_audio": "invalid"},
                {"type": "file", "file": "invalid"},
            ],
            "metadata": {"secret": "discard"},
        }
    ],
    "openai_responses": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "keep"},
                {"type": "input_audio", "data": "invalid"},
                {"type": "input_file", "file_data": 42},
            ],
            "metadata": {"secret": "discard"},
        }
    ],
    "anthropic": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep", "citations": {"invalid": True}},
                {"type": "document", "source": "invalid"},
                {"type": "audio", "source": "invalid"},
            ],
            "metadata": {"secret": "discard"},
        }
    ],
    "gemini": [
        {
            "role": "user",
            "parts": [
                {"text": "keep", "part_metadata": {"secret": "discard"}},
                {"inline_data": {"mime_type": "audio/wav", "data": "invalid"}},
                {"file_data": {"mime_type": "application/pdf", "file_uri": 42}},
            ],
            "metadata": {"secret": "discard"},
        }
    ],
    "ollama": [
        {
            "role": "user",
            "content": "keep",
            "audio": {"invalid": True},
            "file": {"invalid": True},
            "metadata": {"secret": "discard"},
        }
    ],
}


def native_reasoning(protocol, *, signed=False):
    if protocol == "openai_chat":
        return [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}]
    if protocol == "openai_responses":
        return [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "reason"}],
                **({"encrypted_content": "opaque"} if signed else {}),
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
            },
        ]
    if protocol == "anthropic":
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reason",
                        **({"signature": "signed"} if signed else {}),
                    },
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
    if protocol == "gemini":
        return [
            {
                "role": "model",
                "parts": [
                    {
                        "text": "reason",
                        "thought": True,
                        **({"thought_signature": "c2lnbmVk"} if signed else {}),
                    },
                    {"text": "answer"},
                ],
            }
        ]
    return [{"role": "assistant", "content": "answer", "thinking": "reason"}]


def image_tool_result(protocol):
    if protocol == "anthropic":
        return [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": [
                            {"type": "text", "text": "before"},
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://example.test/p.png"},
                            },
                            {"type": "text", "text": "after"},
                        ],
                    }
                ],
            },
            {"role": "user", "content": "tail"},
        ], "$.messages[1].content[0].content[1]"
    return [
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c",
            "output": [
                {"type": "input_text", "text": "before"},
                {"type": "input_image", "image_url": "https://example.test/p.png"},
                {"type": "input_text", "text": "after"},
            ],
        },
        {"role": "user", "content": "tail"},
    ], "$.messages[1].output[1]"


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_unsupported_native_fields_are_discarded_before_sdk_validation(source, target):
    codec = MyCodec()
    incoming = deepcopy(UNSUPPORTED_INPUTS[source])
    original = deepcopy(incoming)
    codec.codecs[source].encoder.validate_input(incoming)
    result = codec.convert(incoming, input_format=source, target_format=target)
    assert "keep" in str(result)
    assert not any(
        value in str(result)
        for value in ("invalid", "discard", "metadata", "citations", "audio", "document")
    )
    assert incoming == original


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_unsigned_reasoning_remains_distinct_or_reports_loss(source, target):
    codec = MyCodec()
    incoming = native_reasoning(source)
    codec.codecs[source].encoder.validate_input(incoming)
    if target in ("anthropic", "gemini"):
        with pytest.raises(SemanticLossError) as captured:
            codec.convert(incoming, input_format=source, target_format=target)
        assert len(captured.value.losses.items) == 1
        assert captured.value.losses.items[0].origin_path.startswith("$.messages[")
        draft = captured.value.draft
        assert "answer" in str(draft)
        assert "reason" not in str(draft)
    else:
        result = codec.convert(incoming, input_format=source, target_format=target)
        assert "reason" in str(result)
        assert "answer" in str(result)
        assert "reasonanswer" not in str(result)


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "openai_responses"])
def test_native_signed_reasoning_retains_opaque_fields(protocol):
    incoming = native_reasoning(protocol, signed=True)
    result = MyCodec().convert(incoming, input_format=protocol, target_format=protocol)
    assert (
        "opaque"
        if protocol == "openai_responses"
        else "signed"
        if protocol == "anthropic"
        else "c2lnbmVk"
    ) in str(result)
    assert "answer" in str(result)


@pytest.mark.parametrize("source", ["anthropic", "openai_responses"])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_tool_images_report_exact_native_source_and_complete_translation(source, target):
    incoming, origin_path = image_tool_result(source)
    codec = MyCodec()
    if target in ("anthropic", "openai_responses"):
        result = codec.convert(incoming, input_format=source, target_format=target)
        assert "https://example.test/p.png" in str(result)
    else:
        with pytest.raises(SemanticLossError) as captured:
            codec.convert(incoming, input_format=source, target_format=target)
        error = captured.value
        assert len(error.losses.items) == 1
        assert error.losses.items[0].origin_path == origin_path
        assert "https://example.test/p.png" in str(error.losses.items[0].origin_value)
        result = error.draft
        assert "https://example.test/p.png" not in str(result)
    assert all(value in str(result) for value in ("before", "after", "tail"))


@pytest.mark.parametrize("source", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_strict_is_preserved_only_by_the_three_declared_protocols(source, target):
    function = {"name": "f", "parameters": {"type": "object"}, "strict": True}
    tool = (
        {"type": "function", "function": function}
        if source == "openai_chat"
        else {"type": "function", **function}
        if source == "openai_responses"
        else {"name": "f", "input_schema": function["parameters"], "strict": True}
    )
    result = MyCodec().convert(tools=[tool], input_format=source, target_format=target)
    assert ("strict" in str(result)) == (target in ("openai_chat", "openai_responses", "anthropic"))


def test_ollama_schema_loss_preserves_later_tools_and_exact_schema_source():
    result = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "additionalProperties": False},
            },
        },
        {"type": "function", "function": {"name": "later", "parameters": {"type": "object"}}},
    ]
    with pytest.raises(SemanticLossError) as captured:
        MyCodec().convert(tools=result, input_format="openai_chat", target_format="ollama")
    assert captured.value.losses.items[0].origin_path == "$.tools[0].function.parameters"
    assert captured.value.losses.items[0].origin_value == {
        "type": "object",
        "additionalProperties": False,
    }
    assert captured.value.draft["tools"][1]["function"]["name"] == "later"


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "openai_responses"])
def test_signature_ownership_is_part_of_the_value_when_message_protocol_is_absent(protocol):
    field, signature = {
        "anthropic": ("signature", "signed"),
        "gemini": ("signature", "c2lnbmVk"),
        "openai_responses": ("encrypted_content", "opaque"),
    }[protocol]
    values = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "reason", "protocol": protocol, field: signature}
            ],
        }
    ]
    result = MyCodec().decode(values, target_format=protocol)
    assert signature in str(result)
    assert "reason" in str(result)


@pytest.mark.parametrize("target", PROTOCOLS)
def test_assistant_images_are_discarded_at_the_source_boundary(target):
    incoming = [
        {
            "role": "model",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": "aW1hZ2U="}},
                {"text": "answer"},
            ],
        }
    ]
    result = MyCodec().convert(incoming, input_format="gemini", target_format=target)
    assert "answer" in str(result)
    assert "aW1hZ2U=" not in str(result)
    assert "image" not in str(result)
