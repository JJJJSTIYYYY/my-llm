"""Gemini content block conversion."""

import base64
from typing import Any

from google.genai import types

from my_llm.codec.utils.native import content_blocks, media_source, text_content
from my_llm.types.exceptions import MessageValidationError


def encode_content(content: Any, role: str) -> list[types.Part]:
    return [encode_content_block(block, role) for block in content_blocks(content)]


def encode_content_block(block: dict[str, Any], role: str) -> types.Part:
    kind = block.get("type")
    if kind in ("text", "input_text", "output_text"):
        return types.Part(text=text_content([block]))
    if role == "user" and kind in ("image_url", "input_image", "file", "input_file"):
        return _encode_media(block)
    raise MessageValidationError(f"unsupported Gemini content block: {kind!r}")


def _encode_media(block: dict[str, Any]) -> types.Part:
    data, mime, encoding = media_source(block)
    if not encoding:
        return types.Part(file_data=types.FileData(file_uri=data))
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise MessageValidationError("invalid media base64 data") from exc
    return types.Part(inline_data=types.Blob(data=raw, mime_type=mime))
