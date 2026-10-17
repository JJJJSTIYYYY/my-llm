"""Translate native function calls into intermediate blocks."""

from my_llm.codec.shared.conversion import arguments, extras


def tool_block(call: dict, *, call_id: str | None = None) -> dict:
    fn = call.get("function", call)
    extensions = extras(fn, "name", "arguments", "index")
    return {
        **extras(call, "id", "type", "function", "index"),
        "type": "tool_use",
        "id": call.get("id") or call_id,
        "name": fn.get("name"),
        "input": arguments(fn.get("arguments", {})),
        **({"function": extensions} if extensions else {}),
    }
