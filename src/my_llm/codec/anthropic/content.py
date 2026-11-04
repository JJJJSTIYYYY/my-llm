"""Select the fixed supported Anthropic content fields."""

import my_llm.codec.anthropic.constants as constants
from my_llm.basic.semantic_loss import originated
from my_llm.codec.shared.conversion import select, tool_result_content


def native_block(block: dict, *, path: str | None = None) -> dict | None:
    """Project a supported Anthropic block, recursively restricting tool-result parts."""
    kind = block.get("type")
    fields = constants.CONTENT_BLOCK_TYPE_TO_FIELDS.get(kind)
    if fields is None:
        return None
    value = select(block, "type", *fields)
    if kind == "image":
        source = block.get("source", {})
        if source.get("type") not in constants.IMAGE_SOURCE_TYPES:
            return None
        value["source"] = select(source, *constants.IMAGE_SOURCE_FIELDS)
    elif kind == "tool_result":
        content = tool_result_content(
            block.get("content"), native_types=constants.TOOL_RESULT_NATIVE_PART_TYPES
        )
        if isinstance(content, list):
            value["content"] = [
                child
                for index, item in enumerate(content)
                if (child := native_block(item, path=f"{path}.content[{index}]" if path else None))
                is not None
                and child.get("type") in constants.MIDDLEWARE_TEXT_IMAGE_TYPES
            ]
        else:
            value["content"] = content
    return originated(value, path, block) if path is not None else value


def intermediate_block(block: dict, *, path: str | None = None) -> dict | None:
    """Normalize Anthropic reasoning and supported native blocks into canonical content."""
    value = native_block(block, path=path)
    if value is None:
        return None
    if value["type"] in constants.THINKING_REDACTED_THINKING_TYPES:
        value = {
            "type": "thinking",
            "reasoning": value.get("thinking"),
            "feedback_item": value,
        }
        if path is not None:
            value = originated(value, path, block)
    elif value["type"] == "tool_result":
        value.setdefault("tool_use_id", None)
    return value
