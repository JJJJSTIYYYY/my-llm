"""Allowed foreign semantics for this native request format."""

from my_llm.codec.shared.rendering import project_block, unsupported

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
    "thinking": ("type", "thinking"),
}


def foreign_block(block: dict) -> dict | None:
    if block.get("type") in ("image", "document"):
        source = block.get("source", {})
        if source.get("type") == "url":
            return {"type": "text", "text": source["url"]}
        if block["type"] == "document":
            return unsupported(block)
    return project_block(
        block, FIELDS, text_fields={"refusal": "refusal"}, omitted=("redacted_thinking",)
    )
