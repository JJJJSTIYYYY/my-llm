"""Normalize native counters without losing their original values."""

from my_llm.codec.shared.usage import record


def encode(value: dict | None) -> dict | None:
    if value is None:
        return None
    return record(
        value,
        "ollama",
        input_base=value.get("prompt_eval_count"),
        output_base=value.get("eval_count"),
    )
