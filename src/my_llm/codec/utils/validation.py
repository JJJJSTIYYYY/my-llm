"""History validation at message boundaries, independent of the source protocol."""

from my_llm.codec.utils.conversion import dictionaries
from my_llm.codec.utils.keys import OPAQUE
from my_llm.exceptions import MessageValidationError


def validate_history(messages: list[dict], *, allow_external_calls: bool = False) -> None:
    if not messages:
        raise MessageValidationError("messages must be non-empty")
    calls = set()
    used = set()
    for message in dictionaries(messages):
        if message.get(OPAQUE):
            continue
        if message.get("role") not in ("system", "developer", "user", "assistant"):
            raise MessageValidationError("invalid message role")
        for block in message.get("content", []):
            if block.get("type") == "tool_use":
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise MessageValidationError("tool call requires an ID")
                if call_id in calls or call_id in used:
                    raise MessageValidationError("duplicate tool call ID")
                calls.add(call_id)
            elif block.get("type") == "tool_result":
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id:
                    raise MessageValidationError("tool result requires a matching call ID")
                if call_id in used:
                    raise MessageValidationError("duplicate tool result")
                # Responses may reference calls stored in server-side history.
                # Duplicate results are still invalid in that mode.
                if call_id not in calls and not allow_external_calls:
                    raise MessageValidationError("tool result refers to an unknown call ID")
                used.add(call_id)
    if calls - used:
        raise MessageValidationError(
            "tool calls require exactly one result before generating again"
        )
