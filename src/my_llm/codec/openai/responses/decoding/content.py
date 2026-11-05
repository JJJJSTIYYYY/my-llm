"""Render the supported Responses text, images and reasoning fields."""

import json
from typing import Any

import my_llm.codec.openai.responses.constants as constants
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.conversion import select, source_uri, text_result
from my_llm.codec.shared.input import input_content, without_feedback
from my_llm.profile.resolution import ResolvedProfile


def _tool_output(value: Any) -> Any:
    """Render object results accepted by the public decoder as native text."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value


def reasoning_summary(block: dict) -> list[dict]:
    """Project canonical reasoning text parts into native summary entries."""
    reasoning = block.get("reasoning")
    return [{"type": "summary_text", "text": reasoning}] if reasoning else []


def render_content(
    content: list[dict], *, role: str = "user", profile: ResolvedProfile | None = None
) -> list[dict]:
    """Render role-specific Responses text/images while applying image capabilities."""
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
            if not input_content(block, profile):
                continue
            if role == "assistant":
                report_loss(block, "Responses assistant output cannot represent images")
                continue
            result.append(
                {
                    "type": "input_image",
                    "image_url": source_uri(block.get("source", {})),
                    "detail": "auto",
                }
            )
    return result


def foreign_block(block: dict) -> dict | None:
    """Adapt foreign result/reasoning blocks and report unrepresentable reasoning."""
    kind = block.get("type")
    if kind == "tool_result":
        value = block.get("content")
        return {
            **select(block, *constants.MIDDLEWARE_TOOL_RESULT_IDENTITY_FIELDS),
            "content": [
                projected for child in value if (projected := foreign_block(child)) is not None
            ]
            if isinstance(value, list)
            else text_result(value),
        }
    if kind == "thinking":
        return without_feedback(block)
    return block if kind in constants.MIDDLEWARE_TEXT_IMAGE_TOOL_USE_TYPES else None
