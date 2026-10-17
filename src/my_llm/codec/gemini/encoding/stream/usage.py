"""Normalize native counters without losing their original values."""

from my_llm.codec.shared.usage import record


def encode(value: dict | None) -> dict | None:
    if value is None:
        return None
    thoughts = value.get("thoughts_token_count")
    return record(
        value,
        "gemini",
        input_base=value.get("prompt_token_count"),
        input_add={"tools": value.get("tool_use_prompt_token_count")},
        output_base=value.get("candidates_token_count"),
        output_add={"thinking": thoughts},
        total=value.get("total_token_count"),
        cached=value.get("cached_content_token_count"),
        reasoning=thoughts,
    )
