"""Shared native completion-reason mappings."""

_REASONS = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}
