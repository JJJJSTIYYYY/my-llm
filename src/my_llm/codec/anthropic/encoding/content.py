"""Anthropic content block conversion."""

import base64
from typing import Any

from my_llm.codec.utils.native import content_blocks, media_source, text_content
from my_llm.types.exceptions import MessageValidationError


def encode_content(content: Any, role: str) -> list[dict[str, Any]]:
    return [encode_content_block(block, role) for block in content_blocks(content)]


def encode_content_block(block: dict[str, Any], role: str) -> dict[str, Any]:
    kind = block.get("type")
    if kind in ("text", "input_text", "output_text"):
        return {"type": "text", "text": text_content([block])}
    if role == "user" and kind in ("image_url", "input_image", "file", "input_file"):
        image = kind in ("image_url", "input_image")
        return {
            "type": "image" if image else "document",
            "source": _encode_media_source(block, image),
        }
    raise MessageValidationError(f"unsupported Anthropic content block: {kind!r}")


def _encode_media_source(block: dict[str, Any], image: bool) -> dict[str, Any]:
    data, mime, encoding = media_source(block)
    if not encoding:
        if not data.startswith(("https://", "http://")):
            raise MessageValidationError("Anthropic media URL must use HTTP or HTTPS")
        return {"type": "url", "url": data}
    supported = (
        ("image/jpeg", "image/png", "image/gif", "image/webp")
        if image
        else ("application/pdf", "text/plain")
    )
    if mime not in supported:
        raise MessageValidationError("unsupported Anthropic media MIME type")
    if mime == "text/plain":
        return {"type": "text", "media_type": mime, "data": _decode_document_text(data)}
    return {"type": "base64", "media_type": mime, "data": data}


def _decode_document_text(data: str) -> str:
    try:
        return base64.b64decode(data, validate=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise MessageValidationError("invalid text document base64 or UTF-8") from exc
