"""Recognized native fields; residual leaves retain their source paths."""

CALL = {"id": True, "type": True, "function": {"name": True, "arguments": True}}


def content_fields(part: dict) -> dict:
    kind = part.get("type")
    if kind not in (
        "text",
        "image_url",
        "file",
    ):
        return {}
    if kind == "text":
        return {"type": True, "text": True, "annotations": True}
    if kind == "image_url":
        return {
            "type": True,
            "image_url": {"url": True, "detail": True}
            if isinstance(part.get("image_url"), dict)
            else True,
        }
    if isinstance(part.get("file"), dict):
        return {"type": True, "file": {"file_data": True, "file_url": True}}
    return {"type": True, "file_data": True, "file_url": True}
