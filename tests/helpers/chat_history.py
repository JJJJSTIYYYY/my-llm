"""Check Chat tool associations on final messages without repairing them again."""


def assert_chat_tool_sequence(messages):
    """Require one contiguous result per call, allowing parallel results in any order.

    Check tool_calls[].id against tool.tool_call_id, not message IDs. The pending
    calls must all finish before any user/system/developer/assistant message.
    Schema validation alone cannot check these cross-message constraints.
    """
    pending = set()
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "tool":
            call_id = message.get("tool_call_id")
            assert isinstance(call_id, str) and call_id, f"messages[{index}]: missing tool_call_id"
            assert call_id in pending, (
                f"messages[{index}]: unexpected or duplicate tool result {call_id!r}; "
                f"pending={sorted(pending)}"
            )
            pending.remove(call_id)
            continue
        assert not pending, (
            f"messages[{index}]: {role} interrupts pending tool results {sorted(pending)}"
        )
        calls = message.get("tool_calls") or []
        if calls:
            assert role == "assistant", f"messages[{index}]: tool_calls on {role}"
            ids = [call.get("id") for call in calls]
            assert all(isinstance(call_id, str) and call_id for call_id in ids), (
                f"messages[{index}]: missing tool call ID"
            )
            assert len(ids) == len(set(ids)), f"messages[{index}]: duplicate tool call IDs {ids}"
            pending.update(ids)
    assert not pending, f"end of messages: missing tool results {sorted(pending)}"


def tool_call(call_id):
    """Independent Chat call fixture for testing the assertion itself."""
    return {"id": call_id, "type": "function", "function": {"name": "look", "arguments": "{}"}}
