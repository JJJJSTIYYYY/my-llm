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
        max_tokens=parameter("max_tokens", 2048, protocol=target),
    )


def prepare(codec, messages, source, profile):
    return codec.prepare(
        {"messages": messages, "tools": omit, "stream": False},
        profile,
        input_format=source,
        client=codec.client_parameters(profile),
    )["request"]


def gemini_image_history(media, sdk):
    image = (
        {"inline_data": {"mime_type": "image/png", "data": b"png"}}
        if media == "inline"
        else {"file_data": {"mime_type": "image/png", "file_uri": "https://example.test/p.png"}}
    )
    parts = [
        {"inline_data": {"mime_type": "audio/wav", "data": Unreadable()}},
        image,
        {"file_data": {"mime_type": "application/pdf", "file_uri": Unreadable()}},
    ]
    messages = [
        {"role": "model", "parts": [{"function_call": {"id": "c", "name": "f", "args": {}}}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "c",
                        "name": "f",
                        "response": {"result": "ok"},
                        "parts": parts,
                    }
                }
            ],
        },
    ]
    if sdk:
        from google.genai import types

        native_parts = []
        for part in parts:
            field = next(iter(part))
            model = (
                types.FunctionResponseBlob
                if field == "inline_data"
                else types.FunctionResponseFileData
            )
            native_parts.append(
                types.FunctionResponsePart.model_construct(
                    **{field: model.model_construct(**part[field])}
                )
            )
        messages = [
            types.Content(**messages[0]),
            types.Content.model_construct(
                role="user",
                parts=[
                    types.Part.model_construct(
                        function_response=types.FunctionResponse.model_construct(
                            id="c", name="f", response={"result": "ok"}, parts=native_parts
                        )
                    )
                ],
            ),
        ]
        image = native_parts[1]
    return messages, image


@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("media", ["inline", "file"])
@pytest.mark.parametrize("sdk", [False, True])
def test_gemini_tool_images_preserve_or_report_original_values(target, media, sdk, profile_factory):
    messages, original_image = gemini_image_history(media, sdk)
    codec = MyCodec()
    profile = target_profile(profile_factory, target)
    if target in ("openai_chat", "gemini", "ollama"):
        with pytest.raises(SemanticLossError) as caught:
            prepare(codec, messages, "gemini", profile)
        assert len(caught.value.losses.items) == 1
        loss = caught.value.losses.items[0]
        assert loss.origin_value is original_image
        assert loss.origin_path == "$.messages[1].parts[0].function_response.parts[1]"
        assert "ok" in str(caught.value.draft)
        assert "image/png" not in str(caught.value.draft)
        return
    request = prepare(codec, messages, "gemini", profile)
    parts = (
        request["input"][-1]["output"]
        if target == "openai_responses"
        else request["messages"][-1]["content"][0]["content"]
    )
    assert len(parts) == 2
    assert json.loads(parts[0]["text"]) == {"result": "ok"}
    image = parts[1]
    assert image["type"] == ("input_image" if target == "openai_responses" else "image")
    assert ("cG5n" if media == "inline" else "https://example.test/p.png") in str(image)


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_application_arrays_with_type_keys_remain_tool_text(source, target, profile_factory):
    payload = [{"type": "invoice", "value": 1}, {"type": "receipt", "value": {"count": 2}}]
    request = prepare(
        MyCodec(),
        application_result_history(source, payload),
        source,
        target_profile(profile_factory, target),
    )
    assert json.loads(native_result_text(target, request)) == payload


@pytest.mark.parametrize("source", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_unknown_native_tool_parts_drop_before_validation_and_json(source, target, profile_factory):
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
    request = prepare(MyCodec(), messages, source, target_profile(profile_factory, target))
    assert native_result_text(target, request) == "beforeafter"


@pytest.mark.parametrize("target", PROTOCOLS)
def test_empty_tool_schema_is_preserved_or_reported_with_complete_draft(target, profile_factory):
    schema = {}
    tools = [
        {"type": "function", "function": {"name": "first", "parameters": schema}},
        {"type": "function", "function": {"name": "later", "parameters": {"type": "object"}}},
    ]
    codec = MyCodec()
    profile = target_profile(profile_factory, target)
    inputs = {"messages": [{"role": "user", "content": "hi"}], "tools": tools, "stream": False}
    if target == "ollama":
        with pytest.raises(SemanticLossError) as caught:
            codec.prepare(
                inputs, profile, input_format="openai_chat", client=codec.client_parameters(profile)
            )
        assert caught.value.losses.items[0].origin_value is schema
        assert caught.value.losses.items[0].origin_path == "$.tools[0].function.parameters"
        assert caught.value.draft["tools"][1]["function"]["name"] == "later"
        return
    request = codec.prepare(
        inputs, profile, input_format="openai_chat", client=codec.client_parameters(profile)
    )["request"]
    actual = (
        request["tools"][0]["function"]["parameters"]
        if target == "openai_chat"
        else request["tools"][0]["parameters"]
        if target == "openai_responses"
        else request["tools"][0]["input_schema"]
        if target == "anthropic"
        else request["config"]["tools"][0]["function_declarations"][0]["parameters_json_schema"]
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
def test_explicit_opaque_owner_cannot_be_overridden_by_the_message_protocol(target):
    from my_llm import AssistantMessage, ReasoningContent

    owner = "gemini" if target == "anthropic" else "anthropic"
    message = AssistantMessage(
        role="assistant",
        protocol=target,
        content=[ReasoningContent(reasoning="reason", signature="signed", opaque_protocol=owner)],
    )
    with pytest.raises(SemanticLossError):
        MyCodec().decode([message], target_format=target)


@pytest.mark.parametrize("target", ["anthropic", "gemini", "openai_responses"])
@pytest.mark.parametrize("owner", ["explicit", "context"])
def test_native_owner_and_missing_owner_use_the_correct_context(target, owner):
    from my_llm import AssistantMessage, ReasoningContent

    message = AssistantMessage(
        role="assistant",
        protocol=target,
        content=[
            ReasoningContent(
                reasoning="reason",
                signature="signed" if target != "openai_responses" else None,
                encrypted_content="opaque" if target == "openai_responses" else None,
                opaque_protocol=target if owner == "explicit" else None,
            )
        ],
    )
    result = MyCodec().decode([message], target_format=target)
    assert "reason" in str(result)
    assert ("opaque" if target == "openai_responses" else "signed") in str(result)


@pytest.mark.parametrize("target", PROTOCOLS)
def test_foreign_opaque_owner_is_checked_on_public_response_blocks(target):
    block = {
        "type": "thinking",
        "thinking": "reason",
        "mark": "raw",
        "signature": "signed",
        "protocol": "gemini" if target == "anthropic" else "anthropic",
    }
    with pytest.raises(SemanticLossError):
        MyCodec().decode(
            chunks=[{"type": "block_stop", "index": 0, "block": block}],
            target_format=target,
            source_protocol=target,
        )


@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("parameters", [{}, {"parameters": None}, {"parameters_json_schema": None}])
def test_gemini_missing_tool_schema_does_not_become_an_explicit_empty_schema(target, parameters):
    result = MyCodec().convert(
        tools=[{"function_declarations": [{"name": "f", **parameters}]}],
        input_format="gemini",
        target_format=target,
    )
    assert "f" in str(result)
