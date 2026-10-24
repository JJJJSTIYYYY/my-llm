"""Render the supported Responses text, images and reasoning fields."""

import json
from typing import Any

from my_llm.codec.shared.conversion import select, text_result
from my_llm.semantic_loss import originated, report_loss


def _tool_output(value: Any) -> Any:
    """Render object results accepted by the public decoder as native text."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value


def render_content(content: list[dict], *, role: str = "user") -> list[dict]:
    result = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            result.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": block.get("text", ""),
                    **({"annotations": [], "logprobs": []} if role == "assistant" else {}),
                }
            )
        elif kind == "image":
            if role == "assistant":
                report_loss(block, "Responses assistant output cannot represent images")
                continue
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            result.append({"type": "input_image", "image_url": uri, "detail": "auto"})
    return result


def foreign_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind == "tool_result":
        value = block.get("content")
        return {
            **select(block, "type", "tool_use_id", "name"),
            "content": [
                projected for child in value if (projected := foreign_block(child)) is not None
            ]
            if isinstance(value, list)
            else text_result(value),
        }
    if kind == "redacted_thinking":
        report_loss(block, "Responses cannot represent opaque reasoning from another protocol")
        return None
    if kind == "thinking" and block.get("protocol") == "openai_responses":
        return block
    if kind in ("thinking", "text", "tool_use") and any(
        block.get(key) for key in ("signature", "encrypted_content")
    ):
        report_loss(block, "Responses cannot preserve another protocol's reasoning signature")
    if kind == "thinking":
        value = select(block, "type", "thinking", "summary")
        return (
            originated(value, block.origin_path, block.origin_value)
            if hasattr(block, "origin_path")
            else value
        )
    return block if kind in ("text", "image", "thinking", "tool_use") else None
