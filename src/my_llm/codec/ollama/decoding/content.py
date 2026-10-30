"""Render supported Ollama content and report media and opaque reasoning loss."""

from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.conversion import select_present, text_result
from my_llm.codec.shared.input import input_content
from my_llm.profile.resolution import ResolvedProfile


def foreign_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind == "image" and block.get("source", {}).get("type") != "base64":
        report_loss(block, "Ollama images require base64 data")
        return None
    if kind == "thinking" and (
        block.get("redacted")
        or block.get("mark") == "encrypted"
        and not block.get("thinking")
        and not block.get("summary")
    ):
        report_loss(block, "Ollama cannot represent opaque reasoning from another protocol")
        return None
    if kind in ("thinking", "text", "tool_use") and any(
        block.get(key) for key in ("signature", "encrypted_content")
    ):
        report_loss(block, "Ollama cannot preserve another protocol's reasoning signature")
    return block if kind in ("text", "image", "tool_use", "tool_result", "thinking") else None


def tool_text(block: dict, profile: ResolvedProfile | None = None) -> str:
    content = block.get("content")
    if isinstance(content, list):
        retained = []
        for child in content:
            if not input_content(child, profile):
                continue
            if child.get("type") == "image":
                report_loss(child, "Ollama tool results cannot represent images")
            elif child.get("type") == "text":
                retained.append(child)
        return text_result(retained)
    return text_result(content)


def chat_call(block: dict) -> dict:
    return {"function": {"name": block["name"], "arguments": block["input"]}}


def function_tool(function: dict) -> dict:
    return {
        "type": "function",
        "function": {
            **select_present(function, "name", "description"),
            "parameters": function["parameters"]
            if function.get("parameters") is not None
            else {"type": "object"},
        },
    }
