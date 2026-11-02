"""Same-type profile merging preserves declarations and isolates mutable data."""

from dataclasses import dataclass

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    ModelProfile,
    Parameter,
    ProviderProfile,
    merge_profiles,
    omit,
    resolve_profiles,
)


@pytest.mark.parametrize(
    "new_value", [None, omit, Config(default_value=None), Config(default_value=0)]
)
@pytest.mark.parametrize("old_value", [None, omit, Config(default_value=30)])
def test_merge_inherits_unset_fields_and_keeps_explicit_values(old_value, new_value):
    older = ProviderProfile(timeout=old_value)
    original_old = older.timeout
    newer = ProviderProfile(timeout=new_value)
    original = newer.timeout

    merged = merge_profiles(older, newer)

    assert merged.timeout == (older.timeout if new_value is None else original)
    if isinstance(merged.timeout, Config):
        assert merged.timeout is not older.timeout
        assert merged.timeout is not newer.timeout
    assert older.timeout is original_old
    assert newer.timeout is original


def test_merge_replaces_parameters_and_attributes_as_whole_declarations():
    older = ModelProfile(
        model=Attribute(default_value="older"),
        reasoning_enable=Parameter(path="$.thinking", default_value={"budget": 1024}),
    )
    newer = ModelProfile(
        model=Attribute(default_value="newer", comment="new identity"),
        reasoning_enable=Parameter(
            path="$.reasoning", default_value={"enabled": True}, comment="new route"
        ),
    )

    merged = merge_profiles(older, newer)

    assert type(merged) is ModelProfile
    assert merged.model == newer.model
    assert merged.reasoning_enable == newer.reasoning_enable
    assert merged.reasoning_enable.name == "reasoning_enable"
    assert merged.reasoning_enable.source == "input"
    merged.reasoning_enable.default_value["enabled"] = False
    merged.model.default_value = "changed"
    assert newer.reasoning_enable.default_value == {"enabled": True}
    assert newer.model.default_value == "newer"
    assert older.reasoning_enable.default_value == {"budget": 1024}


def test_merge_replaces_config_dictionaries_and_preserves_deletions_until_resolution():
    older = ModelProfile(
        model=Attribute(default_value="older"),
        extra_body=Config(default_value={"old": [1], "replace": {"a": 1}, "drop": 1}),
    )
    newer = ModelProfile(
        model=Attribute(default_value="newer"),
        extra_body=Config(
            default_value={"replace": {"b": 2}, "drop": omit, "base": omit},
            comment="merged options",
        ),
    )

    merged = merge_profiles(older, newer)

    assert merged.extra_body.default_value == {
        "replace": {"b": 2},
        "drop": omit,
        "base": omit,
    }
    assert merged.extra_body.name == "extra_body"
    assert merged.extra_body.comment == "merged options"
    resolved = resolve_profiles(
        ProviderProfile(
            protocol=Config(default_value="openai_chat"),
            extra_body=Config(default_value={"base": 1, "keep": 2}),
        ),
        merged,
    )
    assert resolved.get("extra_body") == {"replace": {"b": 2}, "keep": 2}
    merged.extra_body.default_value["replace"]["b"] = 3
    assert older.extra_body.default_value["old"] == [1]
    assert newer.extra_body.default_value["replace"] == {"b": 2}


@pytest.mark.parametrize(
    "old_value",
    [
        None,
        omit,
        Config(default_value=None),
        Config(default_value=omit),
        Config(default_value={"old": 1}),
    ],
)
def test_merge_dictionary_overrides_any_older_value(old_value):
    merged = merge_profiles(
        ProviderProfile(extra_body=old_value),
        ProviderProfile(extra_body=Config(default_value={"new": True})),
    )
    assert merged.extra_body.default_value == {"new": True}


def test_merge_copies_inherited_catalogue_and_replaces_new_catalogue():
    model = ModelProfile(model=Attribute(default_value="older"))
    older = ProviderProfile(models=Attribute(default_value=[model]))

    inherited = merge_profiles(older, ProviderProfile())
    inherited.list_models()[0].model.default_value = "changed"
    inherited.add_models(ModelProfile(model=Attribute(default_value="added")))
    assert older.list_models() == [model]
    assert model.model.default_value == "older"

    newer = ProviderProfile(models=Attribute(default_value=[]))
    replaced = merge_profiles(older, newer)
    assert replaced.list_models() == []
    assert replaced.models is not newer.models


@dataclass(kw_only=True)
class CustomProvider(ProviderProfile):
    label: Attribute[str] | None = None


def test_merge_preserves_subclass_and_inherited_fields():
    older = CustomProvider(
        label=Attribute(default_value="custom"), timeout=Config(default_value=30)
    )
    merged = merge_profiles(older, CustomProvider(timeout=Config(default_value=60)))
    assert type(merged) is CustomProvider
    assert merged.label == older.label
    assert merged.label is not older.label
    assert merged.timeout.default_value == 60


@pytest.mark.parametrize(
    "older,newer",
    [
        (ProviderProfile(), ModelProfile(model=Attribute(default_value="demo"))),
        (ProviderProfile(), CustomProvider()),
        (None, None),
        ({}, {}),
    ],
)
def test_merge_rejects_invalid_inputs_and_different_concrete_types(older, newer):
    with pytest.raises(ConfigurationError):
        merge_profiles(older, newer)


def test_merge_is_exported_from_profile_package():
    from my_llm.profile import merge_profiles as profile_merge

    assert profile_merge is merge_profiles
