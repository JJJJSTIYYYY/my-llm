"""Gemini Part consumption rules."""


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
