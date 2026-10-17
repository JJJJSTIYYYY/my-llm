"""Explicit usage expectations from the native counter definitions."""

import pytest

from my_llm import MyCodec


def convert_usage(value, source, target):
    codec = MyCodec()
    return codec.codecs[target].decoder.usage(codec.codecs[source].encoder.usage(value))


@pytest.mark.parametrize(
    "source,value,target,expected",
    [
        (
            "anthropic",
            {
                "input_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
                "output_tokens": 7,
            },
            "openai_responses",
            {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 5},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 17,
            },
        ),
        (
            "openai_responses",
            {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 5},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 17,
            },
            "anthropic",
            {
                "input_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
                "output_tokens": 7,
            },
        ),
        (
            "gemini",
            {
                "prompt_token_count": 5,
                "tool_use_prompt_token_count": 2,
                "candidates_token_count": 3,
                "thoughts_token_count": 4,
                "cached_content_token_count": 2,
                "total_token_count": 14,
            },
            "openai_chat",
            {
                "prompt_tokens": 7,
                "completion_tokens": 7,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        ),
        (
            "openai_chat",
            {
                "prompt_tokens": 7,
                "completion_tokens": 7,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
            "gemini",
            {
                "prompt_token_count": 7,
                "candidates_token_count": 3,
                "thoughts_token_count": 4,
                "cached_content_token_count": 2,
                "total_token_count": 14,
            },
        ),
        (
            "ollama",
            {"prompt_eval_count": 1, "eval_count": 2, "eval_duration": 123},
            "openai_chat",
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        ),
    ],
)
def test_counters_follow_native_accounting(source, value, target, expected):
    assert convert_usage(value, source, target) == expected


@pytest.mark.parametrize("target", ["openai_chat", "openai_responses"])
def test_a_total_does_not_invent_missing_input_output_counts(target):
    assert convert_usage({"total_token_count": 3}, "gemini", target) is None
