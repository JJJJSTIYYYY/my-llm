"""Normalize native counters without losing their original values."""

from my_llm.codec.shared.usage import record


def encode(value: dict | None) -> dict | None:
    if value is None:
        return None
    return record(
        value,
        "openai_responses",
        input_base=value.get("input_tokens"),
        output_base=value.get("output_tokens"),
        total=value.get("total_tokens"),
        cached=(value.get("input_tokens_details") or {}).get("cached_tokens"),
        cache_write=(value.get("input_tokens_details") or {}).get("cache_write_tokens"),
        reasoning=(value.get("output_tokens_details") or {}).get("reasoning_tokens"),
    )
