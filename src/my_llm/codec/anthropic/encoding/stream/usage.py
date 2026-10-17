"""Normalize native counters without losing their original values."""

from my_llm.codec.shared.usage import record


def encode(value: dict | None) -> dict | None:
    if value is None:
        return None
    cached = value.get("cache_read_input_tokens")
    written = value.get("cache_creation_input_tokens")
    return record(
        value,
        "anthropic",
        input_base=value.get("input_tokens"),
        input_add={"cache_read": cached, "cache_write": written},
        output_base=value.get("output_tokens"),
        cached=cached,
        cache_write=written,
    )
