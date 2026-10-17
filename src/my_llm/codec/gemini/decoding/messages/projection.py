"""Allowed foreign semantics for this native request format."""

from my_llm.codec.shared.rendering import project_block

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
}


def foreign_block(block: dict) -> dict | None:
    return project_block(
        block,
        FIELDS,
        text_fields={"thinking": "thinking", "refusal": "refusal"},
        omitted=("redacted_thinking",),
    )
