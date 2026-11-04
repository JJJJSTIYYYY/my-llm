"""Render supported Ollama content and report media and opaque reasoning loss."""

import my_llm.codec.ollama.constants as constants
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.conversion import select_present, text_result
from my_llm.codec.shared.input import input_content, without_feedback
from my_llm.profile.resolution import ResolvedProfile


def foreign_block(block: dict) -> dict | None:
    """Adapt foreign Ollama input, reporting remote images and opaque reasoning losses."""
    kind = block.get("type")
    if kind == "image" and block.get("source", {}).get("type") != "base64":
        report_loss(block, "Ollama images require base64 data")
        return None
    if kind == "thinking":
        block = without_feedback(block)
    elif kind in constants.MIDDLEWARE_TEXT_TOOL_USE_TYPES and block.get("signature") is not None:
        report_loss(block, "Ollama cannot preserve another protocol's thought signature")
    return (
        block
        if kind in constants.MIDDLEWARE_TEXT_IMAGE_THINKING_TOOL_USE_TOOL_RESULT_TYPES
        else None
    )


def tool_text(block: dict, profile: ResolvedProfile | None = None) -> str:
    """Render tool-result text and report images that Ollama tool messages cannot carry."""
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
    """Render an Ollama tool call with its native argument object."""
    return {"function": {"name": block["name"], "arguments": block["input"]}}


def function_tool(function: dict) -> dict:
    """Render a native function declaration with an object schema when none was supplied."""
    return {
        "type": "function",
        "function": {
            **select_present(function, *constants.MIDDLEWARE_FUNCTION_METADATA_FIELDS),
            "parameters": function["parameters"]
            if function.get("parameters") is not None
            else {"type": "object"},
        },
    }
