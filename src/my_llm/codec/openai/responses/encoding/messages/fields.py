"""Recognized native fields; residual leaves retain their source paths."""


def content_fields(part: dict) -> dict:
    kind = part.get("type")
    if kind in ("input_text", "output_text", "text"):
        return {
            "type": True,
            "text": True,
            "annotations": True,
            **({"logprobs": True} if kind == "output_text" else {}),
        }
    if kind == "input_image":
        return {"type": True, "image_url": True, "detail": True}
    if kind == "input_file":
        return {"type": True, "file_data": True, "file_url": True}
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
