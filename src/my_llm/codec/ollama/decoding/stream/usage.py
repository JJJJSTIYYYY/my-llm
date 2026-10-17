"""Render token accounting in this native protocol."""

from my_llm.codec.shared.conversion import original
from my_llm.codec.shared.usage import counts


def decode(value: dict | None) -> dict | None:
    if value is None:
        return None
    native = original(value, "ollama")
    if native is not None:
        return native if set() <= native.keys() else None
    input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
    return {
        key: child
        for key, child in {"prompt_eval_count": input_tokens, "eval_count": output_tokens}.items()
        if child is not None
    } or None
