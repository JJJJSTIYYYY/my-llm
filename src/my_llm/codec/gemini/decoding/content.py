"""Render recognized Gemini parts and open function payloads."""

from my_llm.codec.shared.conversion import project_block, select, unsupported

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
}


def decode_part(block: dict, *, foreign: bool = False) -> dict:
    kind = block.get("type")
    common = {} if foreign else select(block, "thought_signature", "part_metadata")
    if kind == "tool_use":
        return {
            **common,
            **({"thought_signature": "skip_thought_signature_validator"} if foreign else {}),
            "function_call": {"id": block["id"], "name": block["name"], "args": block["input"]},
        }
    if kind == "tool_result":
        response = block.get("content")
        return {
            **common,
            "function_response": {
                "id": block["tool_use_id"],
                "name": block.get("name"),
                "response": response if isinstance(response, dict) else {"result": response},
            },
        }
    if kind in ("text", "thinking"):
        return {
            **common,
            "text": block.get("thinking" if kind == "thinking" else "text", ""),
            **({"thought": True} if kind == "thinking" else {}),
        }
    if kind in ("image", "document"):
        source = block.get("source", {})
        data = (
            {"data": source.get("data"), "mime_type": source.get("media_type")}
            if source.get("type") == "base64"
            else {
                "file_uri": source.get("url"),
                **({"mime_type": source["media_type"]} if source.get("media_type") else {}),
            }
        )
        return {
            **common,
            **(
                {
                    "video_metadata": select(
                        block["video_metadata"], "start_offset", "end_offset", "fps"
                    )
                }
                if "video_metadata" in block
                else {}
            ),
            "inline_data" if source.get("type") == "base64" else "file_data": data,
        }
    if kind == "executable_code":
        return {**common, kind: select(block, "language", "code")}
    if kind == "code_execution_result":
        return {**common, kind: select(block, "outcome", "output")}
    return unsupported(block)


def foreign_block(block: dict) -> dict | None:
    return project_block(
        block,
        FIELDS,
        text_fields={"thinking": "thinking", "refusal": "refusal"},
        omitted=("redacted_thinking",),
    )
