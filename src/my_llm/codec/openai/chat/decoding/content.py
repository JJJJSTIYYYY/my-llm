"""Render supported Chat content without changing semantic types."""

from typing import Any

from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.conversion import json_arguments, text_result


def render_content(content: list[dict], *, role: str = "user") -> Any:
    result = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            if role != "user":
                report_loss(block, "Chat can represent images only in user messages")
                continue
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            result.append({"type": "image_url", "image_url": {"url": uri}})
    if all(block["type"] == "text" for block in result):
        return "".join(block["text"] for block in result) or (None if role == "assistant" else "")
    return result


def foreign_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind == "thinking" and (
        block.get("redacted")
        or block.get("mark") == "encrypted"
        and not block.get("thinking")
        and not block.get("summary")
    ):
        report_loss(block, "Chat cannot represent opaque reasoning from another protocol")
        return None
    if kind in ("thinking", "text", "tool_use"):
        if any(block.get(key) for key in ("signature", "encrypted_content")):
            report_loss(block, "Chat cannot preserve another protocol's reasoning signature")
    return block if kind in ("text", "image", "tool_use", "tool_result", "thinking") else None


def tool_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        retained = []
        for child in content:
            if child.get("type") == "image":
                report_loss(child, "Chat tool results cannot represent images")
            elif child.get("type") == "text":
                retained.append(child)
        return text_result(retained)
    return text_result(content)


def chat_call(block: dict) -> dict:
    return {
        "id": block["id"],
        "type": "function",
        "function": {"name": block["name"], "arguments": json_arguments(block["input"])},
    }
