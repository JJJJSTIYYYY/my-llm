"""The test oracle rejects ID mismatches and interrupted Chat tool exchanges."""

import pytest

from tests.helpers.chat_history import assert_chat_tool_sequence, tool_call


@pytest.mark.parametrize("ids", [["a", "b"], ["b", "a"]])
def test_parallel_results_may_arrive_in_either_order(ids):
    assert_chat_tool_sequence(
        [
            {"role": "assistant", "tool_calls": [tool_call("a"), tool_call("b")]},
            *({"role": "tool", "tool_call_id": call_id, "content": ""} for call_id in ids),
            {"role": "user", "content": "next"},
        ]
    )


@pytest.mark.parametrize("role", ["user", "system", "developer", "assistant"])
def test_ordinary_message_cannot_interrupt_parallel_tool_results(role):
    with pytest.raises(AssertionError, match="interrupts pending tool results"):
        assert_chat_tool_sequence(
            [
                {"role": "assistant", "tool_calls": [tool_call("a"), tool_call("b")]},
                {"role": "tool", "tool_call_id": "a", "content": "first"},
                {"role": role, "content": "interruption"},
                {"role": "tool", "tool_call_id": "b", "content": "second"},
            ]
        )


@pytest.mark.parametrize(
    ("calls", "results", "reason"),
    [
        (["a", "a"], ["a"], "duplicate tool call IDs"),
        ([None], [], "missing tool call ID"),
        (["a"], [None], "missing tool_call_id"),
        ([], ["a"], "unexpected or duplicate"),
        (["a"], ["b"], "unexpected or duplicate"),
        (["a"], ["a", "a"], "unexpected or duplicate"),
        (["a", "b"], ["a"], "missing tool results"),
    ],
)
def test_id_failures_are_detected(calls, results, reason):
    with pytest.raises(AssertionError, match=reason):
        assert_chat_tool_sequence(
            [
                {"role": "assistant", "tool_calls": [tool_call(call_id) for call_id in calls]},
                *(
                    {"role": "tool", "tool_call_id": call_id, "content": "result"}
                    for call_id in results
                ),
            ]
        )
