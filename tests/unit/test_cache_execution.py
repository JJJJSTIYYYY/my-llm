"""Warm requests reuse compiled rules and translate only incremental messages."""

from copy import deepcopy

import pytest

from my_llm import MyLLM, TextContent, UserMessage
from my_llm.cache import compiled_cache
from my_llm.codec.base import decoder as decoder_module
from tests.helpers.protocols import PROTOCOLS, history, tools
from tests.performance.request_chain import caches
from tests.performance.workloads import controls, profiles, user_message


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("stdio", (*PROTOCOLS, "middleware"))
async def test_warm_incremental_prepare_skips_old_translation_and_type_compilation(
    monkeypatch, upstream, stdio
):
    provider, model = profiles(upstream)
    async with MyLLM(provider=provider, model=model, stdio_protocol=stdio) as llm:
        declared = tools(stdio) if stdio != "middleware" else []
        first = (
            history(stdio)
            if stdio != "middleware"
            else [UserMessage("user", [TextContent("initial")])]
        )
        delta = [
            user_message(stdio, "increment", media=False)
            if stdio != "middleware"
            else UserMessage("user", [TextContent("increment")])
        ]
        original = deepcopy((first, delta, declared))

        def prepare(messages):
            return llm.codec.prepare(
                {"messages": messages, "tools": declared, "stream": True, **controls(upstream)},
                llm.resolved_profile,
                input_format=stdio,
                client=llm.client_parameters,
                enable_message_cache=True,
                message_cache_key=llm._message_cache_key,
            )

        prepare(first)
        cache = llm.codec._message_cache
        initial_size = len(cache.get(llm._message_cache_key))
        before = caches()
        translations, associations, native_filters = [], [], []

        def track(owner, name, counts):
            method = getattr(owner, name)

            def call(self, messages, *args, **kwargs):
                counts.append(len(messages))
                return method(messages, *args, **kwargs)

            monkeypatch.setattr(type(owner), name, call)

        target = llm.codec.decoder(upstream)
        track(target, "_messages", translations)
        track(target, "filter_messages", native_filters)
        validate = decoder_module.validate_history

        def validate_history(messages, **kwargs):
            associations.append(len(messages))
            return validate(messages, **kwargs)

        monkeypatch.setattr(decoder_module, "validate_history", validate_history)
        encoded = []
        if stdio != "middleware":
            track(llm.codec.codecs[stdio].encoder, "_messages", encoded)

        def unexpected(*args, **kwargs):
            pytest.fail("a warm request must not rebuild SDK types or validators")

        monkeypatch.setattr(compiled_cache, "get_type_hints", unexpected)
        monkeypatch.setattr(compiled_cache, "TypeAdapter", unexpected)
        second = prepare(delta)
        retried = prepare(delta)
        assert second == retried
        assert translations == associations == [1, 1]
        assert encoded == ([1, 1] if stdio != "middleware" else [])
        assert native_filters == [initial_size + 1, initial_size + 1]
        assert len(cache.get(llm._message_cache_key)) == initial_size + 1
        assert (first, delta, declared) == original
        assert not llm._clients
        for name, after in caches().items():
            assert after["misses"] == before[name]["misses"], name
            assert after["currsize"] == before[name]["currsize"], name
