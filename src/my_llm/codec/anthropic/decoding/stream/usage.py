"""Render token accounting in this native protocol."""

from my_llm.codec.shared.conversion import original
from my_llm.codec.shared.usage import counts


def decode(value: dict | None) -> dict | None:
    if value is None:
        return None
    native = original(value, "anthropic")
    if native is not None:
        return native if {"output_tokens"} <= native.keys() else None
    input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
    if output_tokens is None:
        return None
    result = {"output_tokens": output_tokens}
    if input_tokens is not None:
        result["input_tokens"] = input_tokens - (cached or 0) - (cache_write or 0)
    if cached is not None:
        result["cache_read_input_tokens"] = cached
    if cache_write is not None:
        result["cache_creation_input_tokens"] = cache_write
    return result
