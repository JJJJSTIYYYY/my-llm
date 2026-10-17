"""Render function calls for this native protocol."""

from my_llm.codec.shared.conversion import extras, json_arguments


def chat_call(block: dict) -> dict:
    return {
        **extras(block, "type", "id", "name", "input", "function", "thought_signature"),
        "id": block["id"],
        "type": "function",
        "function": {
            **block.get("function", {}),
            "name": block["name"],
            "arguments": json_arguments(block["input"]),
        },
    }
