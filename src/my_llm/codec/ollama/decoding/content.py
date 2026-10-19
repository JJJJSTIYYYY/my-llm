"""ollama decoding: content, fields and tool helpers."""

from my_llm.codec.shared.conversion import project_block, select, select_present, unsupported

FIELDS = {
    "text": ("type", "text"),
    "image": ("type", "source", "detail"),
    "document": ("type", "source"),
    "tool_use": ("type", "id", "name", "input"),
    "tool_result": ("type", "tool_use_id", "name", "content"),
    "thinking": ("type", "thinking"),
}


def foreign_block(block: dict) -> dict | None:
    if block.get("type") in ("image", "document"):
        source = block.get("source", {})
        if source.get("type") == "url":
            return {"type": "text", "text": source["url"]}
        if block["type"] == "document":
            return unsupported(block)
    return project_block(
        block, FIELDS, text_fields={"refusal": "refusal"}, omitted=("redacted_thinking",)
    )


def chat_call(block: dict) -> dict:
    return {
        "function": {
            **select(block, "index"),
            "name": block["name"],
            "arguments": block["input"],
        }
    }


def function_tool(function: dict) -> dict:
    return {
        "type": "function",
        "function": {
            **select_present(function, "name", "description"),
            "parameters": function.get("parameters") or {"type": "object"},
        },
    }
