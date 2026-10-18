"""anthropic decoding: content, fields and tool helpers."""

from my_llm.codec.shared.conversion import project_block, select, text_result

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "is_error", "content"),
}


def foreign_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind in ("image", "document"):
        source = block.get("source", {})
        return {
            "type": kind,
            "source": select(source, "type", "url")
            if source.get("type") == "url"
            else select(source, "type", "media_type", "data"),
        }
    if kind == "tool_result":
        value = block.get("content")
        content = (
            [rendered for child in value if (rendered := foreign_block(child)) is not None]
            if isinstance(value, list)
            else text_result(value)
        )
        return {**select(block, "type", "tool_use_id", "is_error"), "content": content}
    return project_block(
        block,
        FIELDS,
        text_fields={"thinking": "thinking", "refusal": "refusal"},
        omitted=("redacted_thinking",),
    )
