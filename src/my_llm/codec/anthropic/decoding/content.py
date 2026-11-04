"""Project supported Anthropic content and report unrepresentable reasoning."""

import my_llm.codec.anthropic.constants as constants
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.anthropic.content import native_block
from my_llm.codec.shared.conversion import select, text_result
from my_llm.codec.shared.input import input_content, without_feedback
from my_llm.profile.resolution import ResolvedProfile


def foreign_block(block: dict) -> dict | None:
    """Adapt foreign content for Anthropic and report reasoning/signature losses."""
    kind = block.get("type")
    if kind == "thinking":
        return without_feedback(block)
    if kind in constants.MIDDLEWARE_TEXT_TOOL_USE_TYPES and block.get("signature") is not None:
        report_loss(block, "Anthropic cannot preserve another protocol's reasoning signature")
    if kind == "tool_result":
        content = block.get("content")
        return {
            **select(block, *constants.MIDDLEWARE_TOOL_RESULT_ID_FIELDS),
            "content": [value for child in content if (value := foreign_block(child)) is not None]
            if isinstance(content, list)
            else text_result(content),
        }
    return block if kind in constants.MIDDLEWARE_TEXT_IMAGE_TOOL_USE_TYPES else None


def decode_block(block: dict, *, profile: ResolvedProfile | None = None) -> dict | None:
    """Render canonical blocks with Anthropic signatures and nested result capabilities."""
    if block["type"] == "tool_result" and isinstance(block.get("content"), list):
        return {
            **select(block, *constants.MIDDLEWARE_TOOL_RESULT_ID_FIELDS),
            "content": [
                value
                for child in block["content"]
                if input_content(child, profile)
                and (value := decode_block(child, profile=profile)) is not None
            ],
        }
    if block["type"] == "thinking":
        item = block.get("feedback_item")
        if (
            not isinstance(item, dict)
            or item.get("type") not in constants.THINKING_REDACTED_THINKING_TYPES
        ):
            if item is not None:
                report_loss(block, "Anthropic reasoning requires a native feedback item")
            return (
                {"type": "thinking", "thinking": block["reasoning"], "signature": ""}
                if profile is None and block.get("reasoning") is not None
                else None
            )
        if profile is not None and item["type"] == "thinking" and not item.get("signature"):
            report_loss(block, "Anthropic reasoning requires a native signature")
            return None
        return item
    return native_block(block)
