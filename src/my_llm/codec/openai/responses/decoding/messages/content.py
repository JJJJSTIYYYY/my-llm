"""Render intermediate parts in the native OpenAI content format."""

from typing import Any

from my_llm.codec.shared.conversion import extras


def render_content(content: list[dict], *, role: str = "user") -> Any:
    result = []
    for block in content:
        # Gemini signatures have meaning only when replayed to Gemini.
        block = extras(block, "thought_signature")
        kind = block.get("type")
        if kind in ("thinking", "redacted_thinking", "tool_use", "tool_result"):
            continue
        value = extras(block)
        if kind == "text":
            value["type"] = "output_text" if role == "assistant" else "input_text"
            if role == "assistant":
                value.setdefault("annotations", [])
                value.setdefault("logprobs", [])
        elif kind == "refusal" and role != "assistant":
            value = {"type": "input_text", "text": block.get("refusal", "")}
        elif kind == "image":
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            detail = {"detail": block.get("detail", "auto")}
            value = {
                **extras(block, "type", "source", "detail"),
                "type": "input_image",
                "image_url": uri,
                **detail,
            }
        elif kind == "document":
            source = block.get("source", {})
            file = (
                {"file_url": source["url"]}
                if "url" in source
                else {
                    "file_data": f"data:{source.get('media_type', 'application/pdf')};base64,{source.get('data', '')}"
                }
            )
            value = {
                **extras(block, "type", "source"),
                "type": "input_file",
                **file,
            }
        result.append(value)
    return result
