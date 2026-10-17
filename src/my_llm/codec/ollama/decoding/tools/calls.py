"""Render function calls for this native protocol."""

from my_llm.codec.shared.conversion import extras


def chat_call(block: dict) -> dict:
    return {
        "function": {
            **extras(block.get("function", {}), "index"),
            "name": block["name"],
            "arguments": block["input"],
        }
    }
