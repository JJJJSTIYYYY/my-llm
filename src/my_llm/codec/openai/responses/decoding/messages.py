"""openai/responses messages: decoding conversion."""

from typing import Any

from my_llm.codec.utils.conversion import (
    chat_content,
    dictionaries,
    extras,
    json_arguments,
    original,
    tool_output,
)
from my_llm.codec.utils.keys import OPAQUE
from my_llm.profile import ResolvedProfile

PROTOCOL = "openai_responses"


def decode(messages: list, profile: ResolvedProfile | None = None, **parameters: Any) -> list[dict]:
    result = []
    for message in dictionaries(messages):
        if message.get(OPAQUE):
            result.append(extras(message, OPAQUE))
            continue
        native = original(message, PROTOCOL)
        if native is not None:
            result.extend(native)
            continue
        content = message.get("content", [])
        pending = []

        def flush():
            if pending:
                result.append(
                    {
                        **extras(message, "role", "content"),
                        "type": "message",
                        **(
                            {"id": f"msg_{len(result)}", "status": "completed"}
                            if message["role"] == "assistant"
                            else {}
                        ),
                        "role": message["role"],
                        "content": chat_content(pending, responses=True, role=message["role"]),
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
