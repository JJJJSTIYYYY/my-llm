"""Ollama content block conversion."""

from typing import Any

from my_llm.codec.utils.native import content_blocks, media_source, text_content
from my_llm.types.exceptions import MessageValidationError


def encode_content(message: dict[str, Any]) -> dict[str, Any]:
    text = []
    images = []
    for block in content_blocks(message.get("content")):
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text"):
            text.append(text_content([block]))
        elif message["role"] == "user" and kind in ("image_url", "input_image"):
            data, mime, encoding = media_source(block)
            if encoding != "base64":
                raise MessageValidationError("Ollama images require base64 data URIs")
            images.append(data)
        else:
            raise MessageValidationError(f"unsupported Ollama content block: {kind!r}")
    return {
        "role": message["role"],
        "content": "".join(text),
        **({"images": images} if images else {}),
    }
