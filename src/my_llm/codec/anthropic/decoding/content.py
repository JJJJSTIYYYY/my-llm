"""Project supported Anthropic content and report unrepresentable reasoning."""

from my_llm.codec.shared.conversion import select, text_result
from my_llm.semantic_loss import report_loss


def foreign_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind in ("thinking", "redacted_thinking"):
        if block.get("protocol") == "anthropic":
            return block
        report_loss(block, "Anthropic cannot replay reasoning without its native signature")
        return None
    if kind in ("text", "tool_use") and any(
        block.get(key) for key in ("signature", "encrypted_content")
    ):
        report_loss(block, "Anthropic cannot preserve another protocol's reasoning signature")
    if kind == "tool_result":
        content = block.get("content")
        return {
            **select(block, "type", "tool_use_id"),
            "content": [value for child in content if (value := foreign_block(child)) is not None]
            if isinstance(content, list)
            else text_result(content),
        }
    return block if kind in ("text", "image", "tool_use") else None
