"""Native protocol shapes, checked independently against official SDK schemas.

Sources and scope are recorded in docs/reference/schema/README.md and
docs/design/protocol-middleware.md.
Unknown/vendor fields must be ignored; SDK parsing alone
is not proof of conformance because several SDKs allow or ignore extra keys.
"""

import json
from copy import deepcopy

import pytest
from jsonschema import ValidationError as SchemaValidationError
from pydantic import ValidationError

from my_llm import MyCodec, SemanticLossError, Usage
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools
from tests.helpers.sdk import (
    EVENT_TYPES,
    MESSAGE_TYPES,
    TOOL_TYPES,
    assert_native,
    assert_output_native,
)
from tests.helpers.streaming import render


def canonical_history(messages, protocol):
    """Normalize identity/JSON spelling only, keeping each native protocol shape."""
    result = deepcopy(messages)
    for message in result:
        for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
            block.pop("annotations", None)
            block.pop("logprobs", None)
            if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                block["content"] = [{"type": "text", "text": block["content"]}]
        if protocol == "openai_responses":
            if message.get("type") == "function_call_output" and isinstance(
                message.get("output"), str
            ):
                message["output"] = [{"type": "input_text", "text": message["output"]}]
            if message.get("type") == "message" and message.get("role") == "assistant":
                assert isinstance(message["id"], str) and message["id"]
                assert message["status"] == "completed"
                message["id"] = "msg_1"
            if message.get("type") in ("function_call", "function_call_output"):
                assert isinstance(message["call_id"], str) and message["call_id"]
                message["call_id"] = "call_1"
            if message.get("type") == "function_call":
                message["arguments"] = json.loads(message["arguments"])
        elif protocol == "openai_chat":
            for call in message.get("tool_calls", []):
                assert isinstance(call["id"], str) and call["id"]
                call["id"] = "call_1"
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            if message["role"] == "tool":
                message["tool_call_id"] = "call_1"
        elif protocol == "anthropic":
            for block in message["content"]:
                if block["type"] == "tool_use":
                    block["id"] = "call_1"
                elif block["type"] == "tool_result":
                    block["tool_use_id"] = "call_1"
        elif protocol == "gemini":
            for part in message["parts"]:
                for field in ("function_call", "function_response"):
                    if field in part:
                        assert part[field]["name"] == "weather"
                        if "id" in part[field]:
                            part[field]["id"] = "call_1"
    return result


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_histories_match_independent_native_expectations(source, target):
    converted = MyCodec().convert(history(source), input_format=source, target_format=target)[
        "messages"
    ]
    for message in converted:
        assert_native(message, MESSAGE_TYPES[target])
    expected = history(target)
    # Gemini requires an object result. Converting that object into a text-only
    # result preserves its JSON, rather than silently extracting a chosen key.
    if source == "gemini" and target != "gemini":
        field = "output" if target == "openai_responses" else "content"
        if target == "anthropic":
            expected[-1]["content"][0]["content"] = '{"result": "Sunny"}'
        else:
            expected[-1][field] = '{"result": "Sunny"}'
    if source == target == "gemini":
        expected[-1]["parts"][0]["function_response"]["response"] = {
            "result": '{"result": "Sunny"}'
        }
    assert canonical_history(converted, target) == canonical_history(expected, target)


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_tools_match_independent_native_definitions(source, target):
    converted = MyCodec().convert(tools=tools(source), input_format=source, target_format=target)[
        "tools"
    ]
    for tool in converted:
        assert_native(tool, TOOL_TYPES[target])
    expected = tools(target)
    if target in ("openai_chat", "anthropic"):
        for values in (converted, expected):
            function = values[0]["function"] if target == "openai_chat" else values[0]
            function.setdefault("strict", False)
    assert converted == expected


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("mode", ["live", "batch"])
def test_stream_events_match_official_models_and_lifecycle(source, target, mode):
    incoming = stream_events(source, tool=True)
    # Validate the fixtures as well as outputs; source==target can otherwise
    # hide an invalid native event behind an SDK model that ignores fields.
    for event in incoming:
        schema = EVENT_TYPES[source]
        assert_native(event, schema[event["type"]] if isinstance(schema, dict) else schema)

    def operation():
        if mode == "live":
            return render(source, target, incoming)[0]
        return MyCodec().convert(chunks=incoming, input_format=source, target_format=target)[
            "chunks"
        ]

    if source == "gemini" and target != "gemini":
        with pytest.raises(SemanticLossError, match="signature"):
            operation()
        return
    converted = operation()
    for event in converted:
        schema = EVENT_TYPES[target]
        assert_output_native(event, schema[event["type"]] if isinstance(schema, dict) else schema)
    assert_stream_lifecycle(source, incoming)
    assert_stream_lifecycle(target, converted, output=True)


def assert_stream_lifecycle(protocol, events, *, output=False):
    if protocol == "openai_chat":
        if output:
            assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        else:
            assert events[-1]["choices"] == []
            assert events[-1]["usage"]["total_tokens"] == 3
            assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"
    elif protocol == "openai_responses":
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        if not output:
            assert events[-1]["response"]["usage"]["total_tokens"] == 3
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        done = [event["item"] for event in events if event["type"] == "response.output_item.done"]
        assert done == events[-1]["response"]["output"]
        opened, closed = {}, set()
        for event in events:
            index = event.get("output_index")
            if event["type"] == "response.output_item.added":
                assert index not in opened
                opened[index] = event["item"]["id"]
            elif index is not None:
                assert index in opened and index not in closed
                if "item_id" in event:
                    assert event["item_id"] == opened[index]
                if event["type"] == "response.output_item.done":
                    assert event["item"]["id"] == opened[index]
                    closed.add(index)
        assert set(opened) == closed
    elif protocol == "anthropic":
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        if not output:
            assert events[-2]["usage"]["output_tokens"] == 2
        opened, closed = [], []
        for event in events:
            if event["type"] == "content_block_start":
                opened.append(event["index"])
            elif event["type"] == "content_block_delta":
                assert event["index"] in opened and event["index"] not in closed
            elif event["type"] == "content_block_stop":
                closed.append(event["index"])
        assert opened == closed
    elif protocol == "gemini":
        assert events[-1]["candidates"][0]["finish_reason"] == "STOP"
        if not output:
            assert events[-1]["usage_metadata"]["total_token_count"] == 3
    else:
        assert events[-1]["done"] is True
        if not output:
            assert events[-1]["eval_count"] == 2


# A valid one-pixel PNG; conversion checks use actual image bytes, not text
# labelled image/png. Gemini fixtures use SDK JSON names, not REST camelCase.
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="


def image_message(protocol):
    if protocol == "gemini":
        return {"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": PNG}}]}
    if protocol == "ollama":
        return {"role": "user", "content": "", "images": [PNG]}
    if protocol == "anthropic":
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG},
        }
    elif protocol == "openai_responses":
        block = {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{PNG}",
            "detail": "auto",
        }
    else:
        block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    return {
        "role": "user",
        "content": [block],
        **({"type": "message"} if protocol == "openai_responses" else {}),
    }


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_image_messages_use_the_target_native_media_fields(source, target):
    result = MyCodec().convert([image_message(source)], input_format=source, target_format=target)[
        "messages"
    ]
    assert len(result) == 1
    assert_native(result[0], MESSAGE_TYPES[target])
    if target == "gemini":
        assert result[0]["parts"][-1]["inline_data"] == {"mime_type": "image/png", "data": PNG}
    elif target == "anthropic":
        expected = image_message(target)["content"]
        if source == "ollama":
            expected.insert(0, {"type": "text", "text": ""})
        assert result[0]["content"] == expected
    elif target == "openai_responses":
        expected = image_message(target)["content"]
        if source == "ollama":
            expected.insert(0, {"type": "input_text", "text": ""})
        assert result[0]["content"] == expected
    elif target == "ollama":
        assert result[0]["images"] == [PNG]
    else:
        block = result[0]["content"][-1]
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == f"data:image/png;base64,{PNG}"


@pytest.mark.parametrize("source", ["openai_chat", "anthropic", "gemini", "ollama"])
def test_output_limit_is_a_responses_incomplete_event(source):
    incoming = stream_events(source)
    if source == "openai_chat":
        incoming[-2]["choices"][0]["finish_reason"] = "length"
    elif source == "anthropic":
        incoming[-2]["delta"]["stop_reason"] = "max_tokens"
    elif source == "gemini":
        incoming[-1]["candidates"][0]["finish_reason"] = "MAX_TOKENS"
    else:
        incoming[-1]["done_reason"] = "length"
    converted, _ = render(source, "openai_responses", incoming)
    final = converted[-1]
    assert final["type"] == "response.incomplete"
    assert final["response"]["status"] == "incomplete"
    assert final["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    assert_output_native(final, EVENT_TYPES["openai_responses"][final["type"]])


def test_filtering_is_not_reported_as_successful_responses_completion():
    incoming = stream_events("openai_chat")
    incoming[-2]["choices"][0]["finish_reason"] = "content_filter"
    converted, _ = render("openai_chat", "openai_responses", incoming)
    final = converted[-1]
    assert final["type"] == "response.incomplete"
    assert final["response"]["incomplete_details"] == {"reason": "content_filter"}
    assert_output_native(final, EVENT_TYPES["openai_responses"][final["type"]])


@pytest.mark.parametrize("target", ["openai_responses", "anthropic", "gemini", "ollama"])
def test_native_thinking_events_use_only_target_supported_fields(target):
    incoming = stream_events("anthropic")
    incoming[1]["content_block"] = {"type": "thinking", "thinking": "", "signature": ""}
    incoming[2]["delta"] = {"type": "thinking_delta", "thinking": "Considering"}
    incoming.insert(
        3,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed"},
        },
    )
    for event in incoming:
        assert_native(event, EVENT_TYPES["anthropic"][event["type"]])
    if target != "anthropic":
        with pytest.raises(SemanticLossError, match="signature"):
            render("anthropic", target, incoming)
        return
    converted, _ = render("anthropic", target, incoming)
    for event in converted:
        schema = EVENT_TYPES[target]
        assert_output_native(event, schema[event["type"]] if isinstance(schema, dict) else schema)


def test_responses_request_metadata_is_not_part_of_supported_response_content():
    incoming = stream_events("openai_responses", tool=True)
    settings = {
        "tools": tools("openai_responses"),
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    for event in incoming:
        if "response" in event:
            event["response"].update(settings)
    output, _ = render("openai_responses", "openai_responses", incoming)
    for event in (output[0], output[-1]):
        assert event["response"]["tools"] == []
        assert event["response"]["tool_choice"] == "auto"
        assert event["response"]["parallel_tool_calls"] is True
        assert_output_native(event, EVENT_TYPES["openai_responses"][event["type"]])


@pytest.mark.parametrize(
    "kind,field",
    [
        ("response.output_text.delta", "logprobs"),
        ("response.function_call_arguments.done", "name"),
    ],
)
def test_required_stream_fields_are_not_hidden_by_sdk_defaults(kind, field):
    event = next(e for e in stream_events("openai_responses", tool=True) if e["type"] == kind)
    event.pop(field)
    with pytest.raises((ValidationError, SchemaValidationError)):
        assert_native(event, EVENT_TYPES["openai_responses"][kind])


def test_sdk_extra_field_permissiveness_does_not_validate_foreign_signatures():
    message = history("anthropic")[1]
    message["content"][1]["thought_signature"] = "c2lnbmVk"
    with pytest.raises(AssertionError):
        assert_native(message, MESSAGE_TYPES["anthropic"])


def test_gemini_result_requires_a_name_even_when_sdk_allows_its_absence():
    message = history("gemini")[-1]
    message["parts"][0]["function_response"].pop("name")
    with pytest.raises(AssertionError):
        assert_native(message, MESSAGE_TYPES["gemini"])


def test_cumulative_anthropic_usage_is_overwritten_without_losing_cache_counts():
    incoming = stream_events("anthropic")
    incoming[0]["message"]["usage"].update(
        input_tokens=2, cache_read_input_tokens=3, cache_creation_input_tokens=5
    )
    incoming[-2]["usage"] = {"output_tokens": 7}
    incoming.insert(
        -2,
        {
            "type": "message_delta",
            "delta": {"stop_reason": None, "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
    )
    for event in incoming:
        assert_native(event, EVENT_TYPES["anthropic"][event["type"]])
    converted, state = render("anthropic", "openai_chat", incoming)
    assert Usage.from_record(state.usage) == Usage(input=10, output=7, cached=3)
    assert state.usage["cache_write"] == 5
    assert converted[-1]["choices"][0]["finish_reason"] == "stop"
    assert_output_native(converted[-1], EVENT_TYPES["openai_chat"])
