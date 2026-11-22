"""Translate recognized Responses content and function fragments."""

from collections.abc import Iterator
from typing import Any

import my_llm.codec.openai.responses.constants as constants
from my_llm.basic.semantic_loss import originated
from my_llm.codec.shared.conversion import blocks, select, uri_source


def content_blocks(
    content: Any, *, path: str = "$.content", allow_images: bool = True
) -> Iterator[tuple[int, dict]]:
    """Yield recognized content with its original native part position and provenance."""
    for index, block in enumerate(blocks(content)):
        kind = block.get("type")
        if kind in constants.INPUT_OUTPUT_PLAIN_TEXT_PART_TYPES:
            value = {"type": "text", **select(block, "text")}
        elif kind == "input_image" and allow_images and isinstance(block.get("image_url"), str):
            value = {"type": "image", "source": uri_source(block["image_url"])}
        else:
            continue
        yield (
            index,
            originated(value, path if isinstance(content, str) else f"{path}[{index}]", block),
        )


def encode_content(
    content: Any, *, path: str = "$.content", allow_images: bool = True
) -> list[dict]:
    """Project recognized native parts to canonical content without their stream addresses."""
    return [block for _, block in content_blocks(content, path=path, allow_images=allow_images)]
