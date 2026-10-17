"""Allowed foreign semantics for this native request format."""

from my_llm.codec.openai.responses.decoding.messages.content import render_content
from my_llm.codec.shared.rendering import project_block, select, text_result

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
    "refusal": ("type", "refusal"),
}


def foreign_block(block: dict) -> dict | None:
    if block.get("type") == "tool_result":
        value = block.get("content")
        if isinstance(value, list):
            parts = [
                projected for child in value if (projected := foreign_block(child)) is not None
            ]
            output = render_content(parts, role="user")
        else:
            output = text_result(value)
        return {
            **select(block, "type", "tool_use_id", "name"),
            "content": output,
        }
    return project_block(
        block, FIELDS, text_fields={"thinking": "thinking"}, omitted=("redacted_thinking",)
    )
