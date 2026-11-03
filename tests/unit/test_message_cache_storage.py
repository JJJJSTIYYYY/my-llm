"""Translated message lists are independent of request structure and hook edits."""

from copy import deepcopy
from weakref import WeakSet

import pytest

from my_llm import ConfigurationError
from my_llm.cache import MessageCache


def test_cache_owns_batches_and_independent_message_snapshots():
    cache = MessageCache()
    batch = [{"role": "user", "content": ["first"]}]
    cache.update("session", batch)
    first = cache.get("session")
    batch[0]["content"].append("caller mutation")
    first[0]["content"].append("SDK mutation")

    second_batch = [{"role": "assistant", "content": "second"}]
    cache.update("session", second_batch)
    second = cache.get("session")
    assert second == [
        {"role": "user", "content": ["first"]},
        {"role": "assistant", "content": "second"},
    ]
    assert len(first) == 1
    assert second_batch == [{"role": "assistant", "content": "second"}]


def test_system_messages_append_as_regular_translated_list_items():
    cache = MessageCache()
    initial = [
        {"role": "system", "parts": [{"text": "first"}]},
        {"role": "user", "parts": [{"text": "hi"}]},
    ]
    cache.update("session", initial)
    cache.update("session", [{"role": "system", "parts": [{"text": "second"}]}])
    assert cache.get("session") == [
        *initial,
        {"role": "system", "parts": [{"text": "second"}]},
    ]


def test_failed_capture_leaves_previous_batch_and_fingerprint_unchanged():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("capture failed")

    cache = MessageCache()
    cache.update("session", [{"content": "first"}])
    messages = [{"content": "second"}]
    fingerprint = cache.fingerprint(messages)
    with pytest.raises(RuntimeError, match="capture failed"):
        cache.update("session", [*messages, {"content": Uncopyable()}], fingerprint=fingerprint)
    assert cache.get("session") == [{"content": "first"}]
    with pytest.raises(ConfigurationError, match="translated message list"):
        cache.update("session", None)
    cache.update("session", messages, fingerprint=fingerprint)
    assert cache.get("session") == [{"content": "first"}, {"content": "second"}]


def test_instance_keys_and_release_are_independent():
    cache = MessageCache()
    for key in ("first", "second"):
        cache.update(key, [{"content": key}])
    cache.expire("first")
    cache.expire("first")
    assert "first" not in cache
    assert cache.get("first") == []
    assert cache.get("second") == [{"content": "second"}]
    cache.clear()
    assert "second" not in cache


def test_retry_fingerprint_is_last_batch_only_and_released_with_history():
    cache = MessageCache()
    for content in ("A", "A", "B", "A"):
        messages = [{"content": content}]
        cache.update("session", messages, fingerprint=cache.fingerprint(messages))
    assert cache.get("session") == [{"content": "A"}, {"content": "B"}, {"content": "A"}]
    for release in (lambda: cache.expire("session"), cache.clear):
        release()
        messages = [{"content": "A"}]
        cache.update("session", messages, fingerprint=cache.fingerprint(messages))
        assert cache.get("session") == messages


def test_staged_translation_is_published_independently_of_request_and_hook_edits():
    cache = MessageCache()
    cache.update("session", [{"content": "first"}])
    batch = cache.batch("session")
    request_messages = batch.append([{"content": "second"}])
    assert cache.get("session") == [{"content": "first"}]
    request_messages[1]["content"] = "request edit"
    batch.commit()
    request_messages.append({"content": "hook edit"})
    assert cache.get("session") == [{"content": "first"}, {"content": "second"}]


def test_unpublished_batch_does_not_change_history_or_retry_identity():
    cache = MessageCache()
    cache.update("session", [{"content": "first"}])
    messages = [{"content": "second"}]
    fingerprint = cache.fingerprint(messages)
    abandoned = cache.batch("session", fingerprint=fingerprint)
    abandoned.append(messages)
    assert cache.get("session") == [{"content": "first"}]
    accepted = cache.batch("session", fingerprint=fingerprint)
    assert accepted.append(messages) == [{"content": "first"}, {"content": "second"}]
    accepted.commit()
    assert cache.get("session") == [{"content": "first"}, {"content": "second"}]


def test_opaque_extras_preserve_conversion_without_unsafe_deduplication():
    cache = MessageCache()
    assert cache.fingerprint([{"content": "first", "unknown": object()}]) is None
    assert cache.fingerprint([{"content": b"first"}]) != cache.fingerprint(
        [{"content": ["bytes", "6669727374"]}]
    )


def test_batch_extensions_own_messages_and_preserve_input_identity():
    cache = MessageCache()
    original = [{"role": "system", "content": "instructions"}]
    cache.update("session", original)
    incoming = [{"role": "tool", "content": "result"}]
    fingerprint = cache.fingerprint(incoming)
    batch = cache.batch("session", fingerprint=fingerprint)
    batch.append(incoming)
    batch.commit()
    identity = batch.identity
    additions = [{"role": "user", "content": ["extra"]}]
    batch.extend(additions)
    expected = [*original, *incoming, *deepcopy(additions)]
    additions[0]["content"].clear()
    assert cache.get("session") == expected
    retry = cache.batch("session", fingerprint=fingerprint)
    assert retry.append(incoming) == expected
    retry.commit()
    assert retry.identity is identity
    assert cache.get("session") == expected
    next_batch = cache.batch("session", fingerprint=cache.fingerprint([{"content": "next"}]))
    next_batch.append([{"content": "next"}])
    next_batch.commit()
    assert next_batch.identity is not identity
    next_batch.extend([expected[-1]])
    next_batch.extend([expected[-1]])
    # Generic extensions never decide whether an addition should be deduplicated.
    assert cache.get("session") == [*expected, {"content": "next"}, expected[-1], expected[-1]]


def test_consumer_state_uses_weak_batch_identity_and_is_released_with_history():
    cache = MessageCache()
    messages = [{"content": "initial"}]
    fingerprint = cache.fingerprint(messages)
    tracked = WeakSet()
    batch = cache.batch("session", fingerprint=fingerprint)
    with pytest.raises(ConfigurationError, match="staged history"):
        _ = batch.identity
    batch.append(messages)
    batch.commit()
    tracked.add(batch.identity)
    del batch
    retry = cache.batch("session", fingerprint=fingerprint)
    retry.append(messages)
    assert retry.identity in tracked
    del retry
    cache.update("session", [{"content": "next"}])
    assert not tracked
    for release in (lambda: cache.expire("session"), cache.clear):
        batch = cache.batch("session", fingerprint=fingerprint)
        batch.append(messages)
        batch.commit()
        tracked.add(batch.identity)
        del batch
        release()
        assert not tracked


def test_failed_batch_extension_leaves_published_history_and_identity_intact():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("extension capture failed")

    cache = MessageCache()
    incoming = [{"content": "initial"}]
    fingerprint = cache.fingerprint(incoming)
    batch = cache.batch("session", fingerprint=fingerprint)
    batch.append(incoming)
    batch.commit()
    with pytest.raises(RuntimeError, match="extension capture failed"):
        batch.extend([{"content": Uncopyable()}])
    assert cache.get("session") == incoming
    retry = cache.batch("session", fingerprint=fingerprint)
    assert retry.append(incoming) == incoming
    assert retry.identity is batch.identity
