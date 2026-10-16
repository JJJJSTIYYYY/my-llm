"""Translate reported token counters without leaking another API's usage keys."""

from copy import deepcopy


def convert_usage(value: dict | None, source: str | None, target: str) -> dict | None:
    if value is None:
        return None
    if source is None or source == target:
        required = {
            "openai_chat": {"prompt_tokens", "completion_tokens", "total_tokens"},
            "openai_responses": {
                "input_tokens",
                "input_tokens_details",
                "output_tokens",
                "output_tokens_details",
                "total_tokens",
            },
            "anthropic": {"output_tokens"},
        }.get(target, set())
        return deepcopy(value) if required <= value.keys() else None

    cached = cache_write = reasoning = None
    total = value.get("total_tokens", value.get("total_token_count"))
    if source == "openai_chat":
        input_tokens, output_tokens = value.get("prompt_tokens"), value.get("completion_tokens")
        cached = (value.get("prompt_tokens_details") or {}).get("cached_tokens")
        reasoning = (value.get("completion_tokens_details") or {}).get("reasoning_tokens")
    elif source == "openai_responses":
        input_tokens, output_tokens = value.get("input_tokens"), value.get("output_tokens")
        cached = (value.get("input_tokens_details") or {}).get("cached_tokens")
        cache_write = (value.get("input_tokens_details") or {}).get("cache_write_tokens")
        reasoning = (value.get("output_tokens_details") or {}).get("reasoning_tokens")
    elif source == "anthropic":
        input_tokens, output_tokens = value.get("input_tokens"), value.get("output_tokens")
        cached = value.get("cache_read_input_tokens")
        cache_write = value.get("cache_creation_input_tokens")
        if input_tokens is not None:
            # Anthropic reports uncached, cache-read and cache-write input separately.
            input_tokens += (cached or 0) + (cache_write or 0)
    elif source == "gemini":
        input_tokens = value.get("prompt_token_count")
        if input_tokens is not None:
            input_tokens += value.get("tool_use_prompt_token_count") or 0
        output_tokens = value.get("candidates_token_count")
        reasoning = value.get("thoughts_token_count")
        if output_tokens is not None:
            output_tokens += reasoning or 0
        cached = value.get("cached_content_token_count")
    elif source == "ollama":
        input_tokens, output_tokens = value.get("prompt_eval_count"), value.get("eval_count")
    else:
        return None
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens

    if target in ("openai_chat", "openai_responses"):
        # These APIs require both splits when usage is present. A total alone
        # does not justify inventing an input/output split; omit usage instead.
        if input_tokens is None or output_tokens is None:
            return None
        if target == "openai_chat":
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
    if target == "anthropic":
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
    if target == "gemini":
        result = {
            "prompt_token_count": input_tokens,
            "total_token_count": total,
            "candidates_token_count": output_tokens - (reasoning or 0)
            if output_tokens is not None
            else None,
            "thoughts_token_count": reasoning,
            "cached_content_token_count": cached,
        }
    else:
        result = {"prompt_eval_count": input_tokens, "eval_count": output_tokens}
    return {key: child for key, child in result.items() if child is not None} or None
