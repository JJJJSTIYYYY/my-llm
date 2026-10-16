"""Recognized source leaves; all other fields keep their native paths.

These rules describe consumed input fields, not a closed intermediate schema.
Function arguments, result objects and schemas are atomic values: their full
contents move with them, including arbitrary application dictionaries.
"""

CALL = {"id": True, "type": True, "function": {"name": True, "arguments": True}}


def part_fields(part: dict) -> dict:
    if "function_call" in part:
        return {"function_call": {"id": True, "name": True, "args": True}}
    if "function_response" in part:
        return {"function_response": {"id": True, "name": True, "response": True}}
    if "text" in part:
        return {"text": True, "thought": True}
    if "inline_data" in part:
        return {"inline_data": {"mime_type": True, "data": True}}
    if "file_data" in part:
        return {"file_data": {"mime_type": True, "file_uri": True}}
    return {}


def response_fields(item: dict) -> dict:
    kind = item.get("type", "message")
    common = {"type": True, "id": True, "status": True}
    if kind == "function_call":
        return {**common, "call_id": True, "name": True, "arguments": True}
    if kind == "function_call_output":
        return {**common, "call_id": True, "output": True}
    if kind == "reasoning":
        return {**common, "summary": [{"type": True, "text": True}]}
    if kind == "message":
        return {**common, "role": True, "content": [content_fields]}
    return {}


def content_fields(part: dict) -> dict:
    kind = part.get("type")
    if kind not in (
        "text",
        "input_text",
        "output_text",
        "image_url",
        "input_image",
        "file",
        "input_file",
    ):
        return {}
    if kind in ("text", "input_text", "output_text"):
        return {"type": True, "text": True, "annotations": True}
    if kind in ("image_url", "input_image"):
        return {
            "type": True,
            "image_url": {"url": True, "detail": True}
            if isinstance(part.get("image_url"), dict)
            else True,
        }
    if isinstance(part.get("file"), dict):
        return {"type": True, "file": {"file_data": True, "file_url": True}}
    return {"type": True, "file_data": True, "file_url": True}
