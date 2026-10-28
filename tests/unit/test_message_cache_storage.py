"""Translated-history storage owns snapshots and retains only message fields."""

import pytest

from my_llm import ConfigurationError
from my_llm.cache import MessageCache


def test_cache_owns_batches_and_independent_request_snapshots():
    cache = MessageCache()
    batch = {"messages": [{"role": "user", "content": ["first"]}], "max_tokens": 1}
    cache.update("session", batch, paths=(("messages",),))
    first = cache.get("session", batch)
    batch["messages"][0]["content"].append("caller mutation")
    first["messages"][0]["content"].append("SDK mutation")

    second_batch = {"messages": [{"role": "assistant", "content": "second"}], "tools": []}
    cache.update("session", second_batch, paths=(("messages",),))
    second = cache.get("session", second_batch)

    assert second == {
        "messages": [
            {"role": "user", "content": ["first"]},
            {"role": "assistant", "content": "second"},
        ],
        "tools": [],
    }
    assert len(first["messages"]) == 1
    assert second_batch["messages"] == [{"role": "assistant", "content": "second"}]
    assert cache.has_messages("session", ("messages",))


def test_nested_instruction_fields_accumulate_without_caching_generation_options():
    cache = MessageCache()
    paths = (("contents",), ("config", "system_instruction"))
    cache.update(
        "session",
        {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"system_instruction": {"parts": [{"text": "first"}]}},
        },
        paths=paths,
    )
    batch = {
        "contents": [],
        "config": {
            "system_instruction": {"parts": [{"text": "second"}]},
            "max_output_tokens": 7,
        },
    }
    cache.update("session", batch, paths=paths)
    latest = cache.get("session", batch)
    assert latest["config"] == {
        "system_instruction": {"parts": [{"text": "first"}, {"text": "second"}]},
        "max_output_tokens": 7,
    }
    assert batch["config"]["system_instruction"]["parts"] == [{"text": "second"}]
    assert "max_output_tokens" not in cache.get("session", {})["config"]

    # Semantic hooks can replace the instruction shape explicitly.
    replacement = {"contents": [], "config": {"system_instruction": "replacement"}}
    cache.update("session", replacement, paths=paths)
    assert cache.get("session", {})["config"]["system_instruction"] == "replacement"


def test_failed_capture_leaves_previous_batch_unchanged():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("capture failed")

    cache = MessageCache()
    paths = (("messages",), ("system",))
    cache.update("session", {"messages": [{"content": "first"}]}, paths=paths)
    with pytest.raises(RuntimeError, match="capture failed"):
        cache.update(
            "session", {"messages": [{"content": "second"}], "system": Uncopyable()}, paths=paths
        )
    assert cache.get("session", {}) == {"messages": [{"content": "first"}]}
    with pytest.raises(ConfigurationError, match="native message list"):
        cache.update("session", {"messages": None}, paths=paths)
    assert cache.get("session", {}) == {"messages": [{"content": "first"}]}


def test_instance_keys_and_release_are_independent():
    cache = MessageCache()
    for key in ("first", "second"):
        cache.update(key, {"messages": [{"content": key}]}, paths=(("messages",),))
    cache.expire("first")
    cache.expire("first")
    assert "first" not in cache
    assert not cache.has_messages("first", ("messages",))
    assert cache.get("second", {}) == {"messages": [{"content": "second"}]}
    cache.clear()
    assert "second" not in cache
