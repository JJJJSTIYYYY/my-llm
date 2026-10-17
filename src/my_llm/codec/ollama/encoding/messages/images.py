"""Identify the media type omitted from Ollama's base64 image strings."""

import base64
import binascii

from my_llm.exceptions import MessageValidationError


def media_type(value: str) -> str:
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise MessageValidationError("Ollama image must contain base64 image data") from exc
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise MessageValidationError("Ollama image format cannot be identified safely")
