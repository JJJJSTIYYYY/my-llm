"""Render intermediate parts in the native OpenAI content format."""

from typing import Any

from my_llm.codec.shared.conversion import extras
from my_llm.exceptions import UnsupportedCapabilityError


def render_content(content: list[dict], *, role: str = "user") -> Any:
    result = []
    for block in content:
        # Gemini signatures have meaning only when replayed to Gemini.
        block = extras(block, "thought_signature")
        kind = block.get("type")
        if kind in ("image", "document") and role != "user":
            source = block.get("source", {})
            if source.get("url"):
                result.append({"type": "text", "text": source["url"]})
                continue
            raise UnsupportedCapabilityError(
                "Chat cannot represent binary media in this message role"
            )
        if kind in ("thinking", "redacted_thinking", "tool_use", "tool_result"):
            continue
        value = extras(block)
        if kind == "text":
            value.pop("annotations", None)
            value["type"] = "text"
        elif kind == "image":
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            detail = {"detail": block.get("detail", "auto")} if "detail" in block else {}
            value = {
                **extras(block, "type", "source", "detail"),
                "type": "image_url",
                "image_url": {"url": uri, **detail},
            }
        elif kind == "document":
            source = block.get("source", {})
            if "url" in source:
                # Chat accepts file_id/file_data, not a remote file_url. Keep
                # a textual reference without downloading content implicitly.
                result.append({"type": "text", "text": source["url"]})
                continue
            file = {
                "file_data": f"data:{source.get('media_type', 'application/pdf')};base64,{source.get('data', '')}"
            }
            value = {
                **extras(block, "type", "source"),
                "type": "file",
                "file": file,
            }
        result.append(value)
    # A string cannot retain block extensions; collapse only bare text blocks.
    if all(set(b) == {"type", "text"} and b["type"] == "text" for b in result):
        return "".join(b["text"] for b in result) or None
    return result
