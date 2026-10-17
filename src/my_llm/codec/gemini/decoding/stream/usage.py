"""Render token accounting in this native protocol."""

from my_llm.codec.shared.conversion import original
from my_llm.codec.shared.usage import counts


def decode(value: dict | None) -> dict | None:
    if value is None:
        return None
    native = original(value, "gemini")
    if native is not None:
        return native if set() <= native.keys() else None
    input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
    return {
        key: child
        for key, child in {
            "prompt_token_count": input_tokens,
            "total_token_count": total,
            "candidates_token_count": output_tokens - (reasoning or 0)
            if output_tokens is not None
            else None,
            "thoughts_token_count": reasoning,
            "cached_content_token_count": cached,
        }.items()
        if child is not None
    } or None
