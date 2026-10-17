"""Normalize native counters without losing their original values."""

from my_llm.codec.shared.usage import record


def encode(value: dict | None) -> dict | None:
    if value is None:
        return None
    return record(
        value,
        "openai_chat",
        input_base=value.get("prompt_tokens"),
        output_base=value.get("completion_tokens"),
        total=value.get("total_tokens"),
        cached=(value.get("prompt_tokens_details") or {}).get("cached_tokens"),
        reasoning=(value.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    )
