"""ollama encoding: content, fields and tool helpers."""

import base64
import binascii
from pathlib import Path

from my_llm.basic.exceptions import MessageValidationError
from my_llm.codec.shared.conversion import arguments, select


def image_data(value) -> str:
    """Read an SDK Image's declared value without serializing its other fields."""
    if not isinstance(value, dict):
        return value
    value = value["value"]
    if isinstance(value, Path):
        value = value.read_bytes()
    elif isinstance(value, str):
        try:
            path = Path(value)
            if path.is_file():
                value = path.read_bytes()
        except OSError:
            pass
    return base64.b64encode(value).decode("ascii") if isinstance(value, bytes) else value


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


def tool_block(call: dict, *, call_id: str | None = None) -> dict:
    fn = call.get("function", call)
    return {
        "type": "tool_use",
        "id": call.get("id") or call_id,
        "name": fn.get("name"),
        "input": arguments(fn.get("arguments", {})),
        **select(fn, "index"),
    }
