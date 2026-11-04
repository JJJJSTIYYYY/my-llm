"""Configuration fingerprints and reusable comparison results."""

from copy import deepcopy
from dataclasses import dataclass, field, make_dataclass, replace
from enum import Enum

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    InputParameter,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    omit,
    resolve_profiles,
    resolve_profiles_with_hash,
)
from tests.helpers.protocols import PROTOCOLS, chat_event
from tests.helpers.transport import FakeClient
from tests.performance.workloads import profiles


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_hashing_preserves_resolution_and_repeats_for_independent_inputs(protocol):
    provider, model = profiles(protocol)
    resolved, fingerprint = resolve_profiles_with_hash(provider, model)

    assert resolved == resolve_profiles(provider, model)
    assert type(fingerprint) is int
    assert resolve_profiles_with_hash(deepcopy(provider), deepcopy(model)) == (
        resolved,
        fingerprint,
    )
    llm = MyLLM(provider=provider, model=model)
    assert llm.resolved_profile == resolved
    assert llm.profile_hash == fingerprint
    assert llm.compare(deepcopy(provider), deepcopy(model)) is True


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        ({"nested": {"a": 1, "b": 2}}, {"nested": {"b": 2, "a": 1}}),
        ({"values": [1, 2]}, {"values": [2, 1]}),
        ({"value": [1]}, {"value": (1,)}),
        ({"value": None}, {"value": [omit]}),
        ({"value": [omit]}, {"value": [("omit",)]}),
        ({"value": True}, {"value": 1}),
        ({"value": -1}, {"value": -2}),
        ({"value": b"abc"}, {"value": "616263"}),
        ({"value": {1: "a"}}, {"value": {"1": "a"}}),
    ],
)
def test_hash_retains_order_and_distinguishes_values_and_types(left, right):
    model = ModelProfile(model=Attribute(default_value="demo"))
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    first = resolve_profiles_with_hash(
        replace(provider, extra_body=Config(default_value=left)), model
    )
    second = resolve_profiles_with_hash(
        replace(provider, extra_body=Config(default_value=right)), model
    )
    assert first[1] != second[1]


def test_hash_uses_contents_without_observing_shared_object_identity():
    shared = {"values": [1, 2]}
    model = ModelProfile(model=Attribute(default_value="demo"))
    provider = ProviderProfile(
        protocol=Config(default_value="openai_chat"),
        extra_body=Config(default_value={"a": shared, "b": shared}),
    )
    copied = replace(
        provider, extra_body=Config(default_value={"a": deepcopy(shared), "b": deepcopy(shared)})
    )
    assert resolve_profiles_with_hash(provider, model) == resolve_profiles_with_hash(copied, model)


def test_hash_retains_custom_parameter_declaration_order():
    def provider_type(names):
        return make_dataclass(
            "OrderedProvider",
            [
                (
                    name,
                    InputParameter[int] | None,
                    field(
                        default=None,
                    ),
                )
                for name in names
            ],
            bases=(ProviderProfile,),
            kw_only=True,
        )

    declarations = {
        "protocol": Config(default_value="openai_chat"),
        "first": InputParameter(path="$.extra_body.first", default_value=1),
        "second": InputParameter(path="$.extra_body.second", default_value=2),
    }
    model = ModelProfile(model=Attribute(default_value="demo"))
    first = resolve_profiles_with_hash(provider_type(("first", "second"))(**declarations), model)
    second = resolve_profiles_with_hash(provider_type(("second", "first"))(**declarations), model)
    assert first[0].values == second[0].values
    assert first[0].parameters == second[0].parameters
    assert first[0].request_parameters != second[0].request_parameters
    assert first[1] != second[1]


@pytest.mark.parametrize(
    "declaration",
    [
        InputParameter(path="$.extra_body.other", default_value=128),
        InputParameter(path="$.max_completion_tokens", default_value=None),
        InputParameter(path="$.max_completion_tokens", supports=False),
        InputParameter(path="$.max_completion_tokens", default_value=128, comment="new comment"),
        omit,
    ],
)
def test_compare_detects_complete_parameter_changes(declaration):
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(
        model=Attribute(default_value="demo"),
        max_tokens=InputParameter(path="$.max_completion_tokens", default_value=128),
    )
    llm = MyLLM(provider=provider, model=model)

    result = llm.compare(provider, replace(model, max_tokens=declaration))

    assert isinstance(result, tuple)
    assert result[1] != llm.profile_hash
    assert llm.resolved_profile.get("max_tokens") == 128


def test_compare_ignores_overridden_values_and_unresolved_attributes():
    provider = ProviderProfile(
        protocol=Config(default_value="openai_chat"),
        extra_body=Config(default_value={"temperature": 0.1}),
    )
    model = ModelProfile(
        model=Attribute(default_value="demo"),
        extra_body=Config(default_value={"temperature": 0.8}),
    )
    llm = MyLLM(provider=provider, model=model)

    assert (
        llm.compare(
            replace(provider, extra_body=Config(default_value={"temperature": 0.2})),
            replace(model, max_context_window=Attribute(default_value=1024)),
        )
        is True
    )
    assert isinstance(
        llm.compare(provider, replace(model, model=Attribute(default_value="new"))), tuple
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"timeout": Config(default_value=30)},
        {"api_key": InputParameter(path="$.api_key", default_value="offline-test-value")},
        {"extra_headers": Config(default_value={"x-test": "changed"})},
        {"protocol": Config(default_value="openai_responses")},
    ],
)
def test_compare_detects_connection_and_protocol_changes(changes):
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(model=Attribute(default_value="demo"))
    llm = MyLLM(provider=provider, model=model)
    assert isinstance(llm.compare(replace(provider, **changes), model), tuple)


def test_changed_result_can_be_reused_without_resolving_or_hashing_again(monkeypatch):
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(model=Attribute(default_value="demo"))
    llm = MyLLM(provider=provider, model=model)
    changed = replace(provider, extra_body=Config(default_value={"nested": [1]}))
    result = llm.compare(changed, model)
    assert isinstance(result, tuple)

    def unexpected(*args, **kwargs):
        raise AssertionError("precomputed profiles must skip resolution and hashing")

    with monkeypatch.context() as patch:
        patch.setattr("my_llm.my_llm.resolve_profiles_with_hash", unexpected)
        patch.setattr("my_llm.profile.resolution.resolve_profiles", unexpected)
        patch.setattr("my_llm.profile.resolution._profile_hash_data", unexpected)
        first = MyLLM(provider=changed, model=model, resolve_profile_with_hash=result)
        second = MyLLM(provider=changed, model=model, resolve_profile_with_hash=result)

    assert first.profile_hash == second.profile_hash == result[1]
    assert first.resolved_profile == second.resolved_profile == result[0]
    assert first.resolved_profile is not result[0]
    assert first.compare(changed, model) is True
    assert second.compare(changed, model) is True
    result[0].values["extra_body"]["nested"].append(2)
    first.resolved_profile.values["extra_body"]["nested"].append(3)
    assert second.resolved_profile.get("extra_body") == {"nested": [1]}
    assert changed.extra_body.default_value == {"nested": [1]}
    assert llm.resolved_profile.get("extra_body") == {}


def test_compare_resolves_once_using_the_instance_defaults(monkeypatch):
    import my_llm.my_llm as llm_module

    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(model=Attribute(default_value="demo"))
    calls = []
    original = llm_module.resolve_profiles_with_hash

    def traced(provider, model, *, defaults_for):
        calls.append(defaults_for.__self__)
        return original(provider, model, defaults_for=defaults_for)

    monkeypatch.setattr(llm_module, "resolve_profiles_with_hash", traced)
    llm = MyLLM(provider=provider, model=model)
    assert calls == [llm.codec]
    assert llm.compare(provider, model) is True
    assert calls == [llm.codec, llm.codec]
    with pytest.raises(ConfigurationError):
        llm.compare(replace(provider, timeout=Config(default_value=-1)), model)
    assert llm.compare(provider, model) is True


async def test_generation_uses_the_reused_changed_configuration(monkeypatch):
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(model=Attribute(default_value="demo"))
    async with MyLLM(provider=provider, model=model) as original:
        changed = replace(provider, extra_body=Config(default_value={"temperature": 0.2}))
        changed_model = replace(model, model=Attribute(default_value="changed-model"))
        result = original.compare(changed, changed_model)
        assert isinstance(result, tuple)
        async with MyLLM(
            provider=changed,
            model=changed_model,
            resolve_profile_with_hash=result,
            stdio_protocol="openai_chat",
        ) as replacement:
            message = await replacement.generate(messages=[{"role": "user", "content": "hi"}])
            assert message["content"] == "ok"
            assert client.requests[0]["model"] == "changed-model"
            assert client.requests[0]["extra_body"] == {"temperature": 0.2}
    assert client.closed
    assert client.close_count == 1


def test_precomputed_snapshot_still_undergoes_protocol_validation():
    provider = ProviderProfile(
        protocol=Config(default_value="ollama"),
        api_key=InputParameter(path="$.api_key", default_value="offline-test-value"),
    )
    model = ModelProfile(model=Attribute(default_value="demo"))
    result = resolve_profiles_with_hash(provider, model)

    with pytest.raises(ConfigurationError, match="Ollama api_key"):
        MyLLM(provider=provider, model=model, resolve_profile_with_hash=result)


@pytest.mark.parametrize("value", [(), (None, 1), (None, True), [None, 1], (None, 1, 2)])
def test_precomputed_input_requires_a_snapshot_and_integer_hash(value):
    with pytest.raises(ConfigurationError, match="resolve_profile_with_hash"):
        MyLLM(
            provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
            model=ModelProfile(model=Attribute(default_value="demo")),
            resolve_profile_with_hash=value,
        )


def test_hash_supports_dataclasses_enums_sets_and_bytes_without_copying_again():
    class Mode(Enum):
        FAST = "fast"

    @dataclass
    class Options:
        mode: Mode
        payload: bytes
        names: set[str]

    provider = ProviderProfile(
        protocol=Config(default_value="openai_chat"),
        extra_body=Config(default_value={"options": Options(Mode.FAST, b"abc", {"a", "b"})}),
    )
    model = ModelProfile(model=Attribute(default_value="demo"))
    first = resolve_profiles_with_hash(provider, model)
    assert first == resolve_profiles_with_hash(deepcopy(provider), deepcopy(model))


def test_unhashable_and_circular_payloads_raise_configuration_errors():
    model = ModelProfile(model=Attribute(default_value="demo"))
    circular = []
    circular.append(circular)
    for value in (object(), circular):
        provider = ProviderProfile(
            protocol=Config(default_value="openai_chat"),
            extra_body=Config(default_value={"value": value}),
        )
        with pytest.raises(ConfigurationError, match="unhashable payload"):
            resolve_profiles_with_hash(provider, model)


def test_hashing_is_exported_from_the_profile_package():
    from my_llm.profile import resolve_profiles_with_hash as exported

    assert exported is resolve_profiles_with_hash
