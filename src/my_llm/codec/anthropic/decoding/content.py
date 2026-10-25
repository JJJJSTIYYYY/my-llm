"""Project supported Anthropic content and report unrepresentable reasoning."""

from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.anthropic.content import native_block
from my_llm.codec.shared.conversion import select, text_result


def foreign_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind == "thinking":
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


def decode_block(block: dict) -> dict | None:
    if block["type"] == "thinking":
        if block.get("redacted"):
            return {"type": "redacted_thinking", "data": block["encrypted_content"]}
        if not block.get("signature"):
            report_loss(block, "Anthropic reasoning requires a native signature")
            return None
        return {
            "type": "thinking",
            "thinking": block["thinking"] or "\n".join(block.get("summary", ())),
            "signature": block["signature"],
        }
    return native_block(block)
