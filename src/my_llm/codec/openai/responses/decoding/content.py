"""openai/responses decoding: content, fields and tool helpers."""

from typing import Any

from my_llm.codec.shared.conversion import extras
from my_llm.codec.shared.rendering import project_block, select, text_result

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
    "refusal": ("type", "refusal"),
}


def render_content(content: list[dict], *, role: str = "user") -> Any:
    result = []
    for block in content:
        # Gemini signatures have meaning only when replayed to Gemini.
        block = extras(block, "thought_signature")
        kind = block.get("type")
        if kind in ("thinking", "redacted_thinking", "tool_use", "tool_result"):
            continue
        value = extras(block)
        if kind == "text":
            value["type"] = "output_text" if role == "assistant" else "input_text"
            if role == "assistant":
                value.setdefault("annotations", [])
                value.setdefault("logprobs", [])
        elif kind == "refusal" and role != "assistant":
            value = {"type": "input_text", "text": block.get("refusal", "")}
        elif kind == "image":
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            detail = {"detail": block.get("detail", "auto")}
            value = {
                **extras(block, "type", "source", "detail"),
                "type": "input_image",
                "image_url": uri,
                **detail,
            }
        elif kind == "document":
            source = block.get("source", {})
            file = (
                {"file_url": source["url"]}
                if "url" in source
                else {
                    "file_data": f"data:{source.get('media_type', 'application/pdf')};base64,{source.get('data', '')}"
                }
            )
            value = {
                **extras(block, "type", "source"),
                "type": "input_file",
                **file,
            }
        result.append(value)
    return result


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
