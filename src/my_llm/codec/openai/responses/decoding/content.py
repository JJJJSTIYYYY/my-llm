"""openai/responses decoding: content, fields and tool helpers."""

from typing import Any

from my_llm.codec.shared.conversion import project_block, select, text_result

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
    "refusal": ("type", "refusal"),
}


def render_content(content: list[dict], *, role: str = "user") -> Any:
    result = []
    for block in content:
        kind = block.get("type")
        if kind in ("thinking", "redacted_thinking", "tool_use", "tool_result"):
            continue
        value = select(block, "type", "text", "refusal", "annotations", "logprobs")
        if kind == "text":
            value = {
                "type": "output_text" if role == "assistant" else "input_text",
                "text": block.get("text", ""),
            }
            if role == "assistant":
                value["annotations"] = [
                    select(
                        item,
                        "type",
                        "file_id",
                        "index",
                        "filename",
                        "start_index",
                        "end_index",
                        "title",
                        "url",
                        "container_id",
                    )
                    for item in block.get("annotations", [])
                ]
                value["logprobs"] = [
                    select(item, "token", "logprob", "bytes")
                    | (
                        {
                            "top_logprobs": [
                                select(child, "token", "logprob", "bytes")
                                for child in item["top_logprobs"]
                            ]
                        }
                        if "top_logprobs" in item
                        else {}
                    )
                    for item in block.get("logprobs", [])
                ]
        elif kind == "refusal":
            value = (
                {"type": "refusal", "refusal": block.get("refusal", "")}
                if role == "assistant"
                else {"type": "input_text", "text": block.get("refusal", "")}
            )
        elif kind == "image":
            source = block.get("source", {})
            uri = source.get("url")
            if source.get("type") == "base64":
                uri = f"data:{source['media_type']};base64,{source['data']}"
            detail = {"detail": block.get("detail", "auto")}
            value = {
                "type": "input_image",
                **(
                    {"file_id": source["file_id"]}
                    if source.get("type") == "file"
                    else {"image_url": uri}
                ),
                **detail,
            }
        elif kind == "document":
            source = block.get("source", {})
            file = (
                {"file_id": source["file_id"]}
                if source.get("type") == "file"
                else {"file_url": source["url"]}
                if "url" in source
                else {
                    "file_data": f"data:{source.get('media_type', 'application/pdf')};base64,{source.get('data', '')}"
                }
            )
            value = {
                "type": "input_file",
                **select(block, "filename"),
                **file,
            }
        else:
            continue
        result.append(value)
    return result


def foreign_block(block: dict) -> dict | None:
    if block.get("type") == "tool_result":
        value = block.get("content")
        if isinstance(value, list):
            parts = [
                projected for child in value if (projected := foreign_block(child)) is not None
            ]
            output = parts
        else:
            output = text_result(value)
        return {
            **select(block, "type", "tool_use_id", "name"),
            "content": output,
        }
    return project_block(
        block, FIELDS, text_fields={"thinking": "thinking"}, omitted=("redacted_thinking",)
    )
