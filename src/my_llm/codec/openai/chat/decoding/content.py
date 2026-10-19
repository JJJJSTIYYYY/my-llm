"""openai/chat decoding: content, fields and tool helpers."""

from typing import Any

from my_llm.codec.shared.conversion import (
    json_arguments,
    project_block,
    select,
    text_result,
)
from my_llm.exceptions import UnsupportedCapabilityError

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
        kind = block.get("type")
        if kind in ("image", "document") and role != "user":
            source = block.get("source", {})
            if source.get("url"):
                result.append({"type": "text", "text": source["url"]})
                continue
            raise UnsupportedCapabilityError(
                "Chat cannot represent binary media in this message role"
            )
        if kind in ("thinking", "redacted_thinking", "tool_use", "tool_result"):
            continue
        value = select(block, "type", "text", "refusal", "annotations", "logprobs")
        if kind == "text":
            value = {"type": "text", "text": block.get("text", "")}
        elif kind == "input_audio":
            value = {
                "type": "input_audio",
                "input_audio": select(block.get("input_audio", {}), "data", "format"),
            }
        elif kind == "image":
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            detail = {"detail": block.get("detail", "auto")} if "detail" in block else {}
            value = {
                "type": "image_url",
                "image_url": {"url": uri, **detail},
            }
        elif kind == "document":
            source = block.get("source", {})
            if "url" in source:
                # Chat accepts file_id/file_data, not a remote file_url. Keep
                # a textual reference without downloading content implicitly.
                result.append({"type": "text", "text": source["url"]})
                continue
            file = (
                select(source, "file_id")
                if source.get("type") == "file"
                else {
                    "file_data": f"data:{source.get('media_type', 'application/pdf')};base64,{source.get('data', '')}"
                }
            )
            file.update(select(block, "filename"))
            value = {
                "type": "file",
                "file": file,
            }
        else:
            continue
        result.append(value)
    # Bare text blocks can use the native string form.
    if all(set(b) == {"type", "text"} and b["type"] == "text" for b in result):
        return "".join(b["text"] for b in result) or None
    return result


def foreign_block(block: dict) -> dict | None:
    if block.get("type") == "tool_result":
        return {
            **select(block, "type", "tool_use_id", "name"),
            "content": text_result(block.get("content")),
        }
    return project_block(
        block, FIELDS, text_fields={"thinking": "thinking"}, omitted=("redacted_thinking",)
    )


def chat_call(block: dict) -> dict:
    return {
        "id": block["id"],
        "type": "function",
        "function": {
            "name": block["name"],
            "arguments": json_arguments(block["input"]),
        },
    }
