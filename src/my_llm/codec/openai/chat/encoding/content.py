"""Translate recognized Chat content and function calls."""

from typing import Any

import my_llm.codec.openai.chat.constants as constants
from my_llm.basic.semantic_loss import originated
from my_llm.codec.shared.conversion import arguments, blocks, select, uri_source


def encode_content(
    content: Any, *, path: str = "$.content", allow_images: bool = True
) -> list[dict]:
    """Normalize supported Chat content, applying image-role policy in this visit."""
    result = []
    for index, block in enumerate(blocks(content)):
        kind = block.get("type")
        if kind == "text":
            value = select(block, *constants.TEXT_PART_FIELDS)
        elif kind == "image_url" and allow_images:
            image = block.get("image_url", {})
            uri = image.get("url") if isinstance(image, dict) else image
            if not isinstance(uri, str):
                continue
            value = {"type": "image", "source": uri_source(uri)}
        else:
            continue
        result.append(
            originated(value, path if isinstance(content, str) else f"{path}[{index}]", block)
        )
    return result


def tool_block(call: dict, *, call_id: str | None = None) -> dict:
    """Build a canonical tool call, requiring arguments to decode to a JSON object."""
    fn = call.get("function", call)
    return {
        "type": "tool_use",
        "id": call.get("id") or call_id,
        "name": fn.get("name"),
        "input": arguments(fn.get("arguments", {})),
    }
