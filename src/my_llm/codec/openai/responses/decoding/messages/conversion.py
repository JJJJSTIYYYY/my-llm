"""openai/responses messages: decoding conversion."""

from typing import Any

from my_llm.codec.openai.responses.decoding.messages.content import render_content
from my_llm.codec.openai.responses.decoding.messages.projection import foreign_block
from my_llm.codec.shared.conversion import (
    dictionaries,
    extras,
    json_arguments,
    original,
    source_protocol,
    tool_output,
)
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.rendering import project_message
from my_llm.exceptions import UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_responses"


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in dictionaries(messages):
        if message.get(OPAQUE):
            result.extend(original(message, PROTOCOL))
            continue
        native = original(message, PROTOCOL)
        if native is not None:
            result.extend(native)
            continue
        if source_protocol(message) != PROTOCOL:
            message = project_message(message, foreign_block)
        content = message.get("content", [])
        pending = []

        def flush():
            if pending:
                if message["role"] == "assistant" and profile is None:
                    pending[:] = [
                        {"type": "text", "text": block["source"]["url"]}
                        if block.get("type") in ("image", "document")
                        and block.get("source", {}).get("url")
                        else block
                        for block in pending
                    ]
                media = any(block.get("type") in ("image", "document") for block in pending)
                input_media = media and profile is not None
                if media and message["role"] == "assistant" and profile is None:
                    raise UnsupportedCapabilityError(
                        "Responses output messages cannot represent foreign binary media"
                    )
                result.append(
                    {
                        **extras(message, "role", "content"),
                        "type": "message",
                        **(
                            {"id": f"msg_{len(result)}", "status": "completed"}
                            if message["role"] == "assistant" and not input_media
                            else {}
                        ),
                        "role": message["role"],
                        "content": render_content(
                            pending, role="user" if input_media else message["role"]
                        ),
                    }
                )
                pending.clear()

        for block in content:
            kind = block.get("type")
            if kind not in ("tool_use", "tool_result", "thinking"):
                pending.append(block)
                continue
            flush()
            if kind == "tool_use":
                result.append(
                    {
                        **extras(block, "type", "id", "input", "thought_signature"),
                        "type": "function_call",
                        "call_id": block["id"],
                        "arguments": json_arguments(block["input"]),
                    }
                )
            elif kind == "tool_result":
                result.append(
                    {
                        **extras(block, "type", "tool_use_id", "content", "name"),
                        "type": "function_call_output",
                        "call_id": block["tool_use_id"],
                        "output": tool_output(block.get("content", "")),
                    }
                )
            else:
                result.append(
                    {
                        **extras(block, "type", "thinking", "signature", "thought_signature"),
                        "type": "reasoning",
                        "id": block.get("id", f"rs_{len(result)}"),
                        "summary": block.get(
                            "summary", [{"type": "summary_text", "text": block.get("thinking", "")}]
                        ),
                    }
                )
        flush()
    return result
