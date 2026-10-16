from copy import deepcopy

import pytest
from conftest import CHAT_HISTORY

from my_llm import MessageValidationError, MyCodec
from my_llm.codec.utils.validation import validate_history


def validate(messages, **parameters):
    validate_history(
        MyCodec().encode(messages, input_format="openai_chat")["messages"], **parameters
    )


def test_history_pairing_accepts_complete_round_trip():
    validate(CHAT_HISTORY)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "unknown", "duplicate_call", "out_of_order"]
)
def test_history_pairing_rejects_mismatched_results(kind):
    messages = deepcopy(CHAT_HISTORY)
    if kind == "missing":
        messages.pop()
    elif kind == "duplicate":
        messages.append(messages[-1])
    elif kind == "unknown":
        messages[-1]["tool_call_id"] = "other"
    elif kind == "duplicate_call":
        messages.insert(2, messages[1])
    else:
        messages[1], messages[2] = messages[2], messages[1]
    with pytest.raises(MessageValidationError):
        validate(messages)


def test_responses_server_history_allows_external_result_ids():
    validate(
        [{"role": "tool", "tool_call_id": "external", "content": "ok"}], allow_external_calls=True
    )


def test_non_message_input_is_rejected():
    for value in ("hi", ["hi"], [{"role": "invalid", "content": "hi"}], []):
        with pytest.raises(MessageValidationError):
            validate(value)


@pytest.mark.parametrize("protocol", ["gemini", "ollama"])
@pytest.mark.parametrize("first_result,second_result", [("first", "second"), ("second", "first")])
def test_explicit_result_ids_are_removed_from_name_matching_queue(
    protocol, first_result, second_result
):
    if protocol == "gemini":
        messages = [
            {
                "role": "model",
                "parts": [
                    {"function_call": {"id": "first", "name": "weather", "args": {}}},
                    {"function_call": {"id": "second", "name": "weather", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": first_result,
                            "name": "weather",
                            "response": {"result": first_result},
                        }
                    },
                    {
                        "function_response": {
                            "name": "weather",
                            "response": {"result": second_result},
                        }
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "first", "function": {"name": "weather", "arguments": {}}},
                    {"id": "second", "function": {"name": "weather", "arguments": {}}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first_result,
                "tool_name": "weather",
                "content": first_result,
            },
            {"role": "tool", "tool_name": "weather", "content": second_result},
        ]
    intermediate = MyCodec().encode(messages, input_format=protocol)["messages"]
    validate_history(intermediate)
    results = [b for m in intermediate for b in m["content"] if b.get("type") == "tool_result"]
    assert [b["tool_use_id"] for b in results] == [first_result, second_result]


@pytest.mark.parametrize("protocol", ["gemini", "ollama"])
def test_generated_optional_call_ids_do_not_collide_with_explicit_ids(protocol):
    if protocol == "gemini":
        messages = [
            {
                "role": "model",
                "parts": [
                    {"function_call": {"id": "call_0_1", "name": "f", "args": {}}},
                    {"function_call": {"name": "f", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"function_response": {"id": "call_0_1", "name": "f", "response": {"ok": 1}}},
                    {"function_response": {"name": "f", "response": {"ok": 2}}},
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_0_1", "function": {"name": "f", "arguments": {}}},
                    {"function": {"name": "f", "arguments": {}}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_0_1", "tool_name": "f", "content": "1"},
            {"role": "tool", "tool_name": "f", "content": "2"},
        ]
    converter = MyCodec()
    intermediate = converter.encode(messages, input_format=protocol)["messages"]
    validate_history(intermediate)
    calls = [
        block["id"]
        for message in intermediate
        for block in message["content"]
        if block.get("type") == "tool_use"
    ]
    results = [
        block["tool_use_id"]
        for message in intermediate
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert calls[0] == "call_0_1"
    assert calls[1] != calls[0]
    assert results == calls
    assert converter.decode(intermediate, target_format=protocol)["messages"] == messages
