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
    ],
)
def test_counters_follow_native_accounting(source, value, target, expected):
    assert convert_usage(value, source, target) == expected


@pytest.mark.parametrize("target", ["openai_chat", "openai_responses"])
def test_a_total_does_not_invent_missing_input_output_counts(target):
    assert convert_usage({"total_tokens": 3}, "openai_chat", target) is None
