"""Render token accounting in this native protocol."""

from my_llm.codec.shared.conversion import original
from my_llm.codec.shared.usage import counts


def decode(value: dict | None) -> dict | None:
    if value is None:
        return None
    native = original(value, "openai_responses")
    if native is not None:
        return (
            native
            if {
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "input_tokens_details",
                "output_tokens_details",
            }
            <= native.keys()
            else None
        )
    input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
    if input_tokens is None or output_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "input_tokens_details": {
            "cached_tokens": cached or 0,
            "cache_write_tokens": cache_write or 0,
        },
        "output_tokens_details": {"reasoning_tokens": reasoning or 0},
    }
