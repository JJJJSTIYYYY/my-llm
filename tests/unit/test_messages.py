from copy import deepcopy

import pytest

from my_llm import MessageValidationError, MyCodec
from my_llm.codec.shared.conversion import validate_history
from tests.helpers.protocols import CHAT_HISTORY


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
