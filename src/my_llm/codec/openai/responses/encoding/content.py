"""Translate recognized Responses content and function fragments."""

from typing import Any

import my_llm.codec.openai.responses.constants as constants
from my_llm.basic.semantic_loss import originated
from my_llm.codec.shared.conversion import blocks, select, uri_source


def encode_content(
    content: Any, *, path: str = "$.content", allow_images: bool = True
) -> list[dict]:
    """Normalize Responses content with image-role filtering and source provenance."""
    result = []
    for index, block in enumerate(blocks(content)):
        kind = block.get("type")
        if kind in constants.INPUT_OUTPUT_PLAIN_TEXT_PART_TYPES:
            value = {"type": "text", **select(block, "text")}
        elif kind == "input_image" and allow_images and isinstance(block.get("image_url"), str):
            value = {"type": "image", "source": uri_source(block["image_url"])}
        else:
            continue
        result.append(
            originated(value, path if isinstance(content, str) else f"{path}[{index}]", block)
        )
    return result
