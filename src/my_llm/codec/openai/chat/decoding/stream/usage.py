"""Render token accounting in this native protocol."""

from my_llm.codec.shared.conversion import original
from my_llm.codec.shared.usage import counts


def decode(value: dict | None) -> dict | None:
    if value is None:
        return None
    native = original(value, "openai_chat")
    if native is not None:
        return (
            native
            if {"completion_tokens", "total_tokens", "prompt_tokens"} <= native.keys()
            else None
        )
    input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
    if input_tokens is None or output_tokens is None:
        return None
    result = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total,
    }
    if cached is not None:
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    if reasoning is not None:
        result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return result
