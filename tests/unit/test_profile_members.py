"""Profile member enumeration preserves declaration categories and live values."""

from dataclasses import FrozenInstanceError, dataclass, field, fields
from typing import ClassVar

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    ModelProfile,
    Omit,
    OverridableProfile,
    Parameter,
    Profile,
    ProviderProfile,
    omit,
)
from my_llm.registry import codec_for
from tests.helpers.protocols import PROTOCOLS

type Label = Attribute[str] | None


@dataclass(frozen=True, kw_only=True)
class DescriptiveProvider(ProviderProfile):
    label: Label = None
    feature: Config[bool] | Omit = omit
    vendor_limit: Parameter[int] = field(
        default=Parameter(path="$.extra_body.vendor_limit", supports=False),
        metadata={"source": "input"},
    )
    catalogue_kind: ClassVar[Config[str]] = Config(default_value="descriptive")


def test_all_profiles_expose_the_shared_public_enumeration_contract():
    from my_llm.profile import Profile as ProfileExport

    assert Profile is ProfileExport
    for profile in (Profile(), OverridableProfile(), ProviderProfile()):
        assert isinstance(profile, Profile)
        members = (
            profile.get_attributes(),
            profile.get_configs(),
            profile.get_parameters(),
        )
        assert set().union(*members) == {item.name for item in fields(profile)}
        assert sum(map(len, members)) == len(fields(profile))


def test_enumeration_uses_declarations_for_inherited_unset_and_omitted_members():
    provider = DescriptiveProvider(protocol=Config(default_value="openai_chat"), max_tokens=omit)
    attributes = provider.get_attributes()
    configs = provider.get_configs()
    parameters = provider.get_parameters()

    assert attributes == {"models": None, "label": None}
    assert configs["protocol"] is provider.protocol
    assert configs["feature"] is omit
    assert configs["timeout"] is None
    assert parameters["api_key"] is None
    assert parameters["max_tokens"] is omit
    assert parameters["vendor_limit"] is provider.vendor_limit
    assert parameters["vendor_limit"].supports is False
    assert "content" in parameters and "reasoning_content" in parameters
    assert "catalogue_kind" not in attributes | configs | parameters


def test_model_attributes_are_wrappers_and_enumeration_preserves_frozen_identity():
    model = ModelProfile(model=Attribute(default_value="demo"))
    attributes = model.get_attributes()
    assert attributes == {"model": model.model, "max_context_window": model.max_context_window}
    assert attributes["model"] is model.model
    assert "supported_roles" not in attributes
    attributes.clear()
    assert model.get_attributes()["model"].default_value == "demo"
    with pytest.raises(FrozenInstanceError):
        model.model = Attribute(default_value="changed")


def test_each_call_reads_current_members_without_caching_values_or_copying_payloads():
    first = DescriptiveProvider(label=Attribute(default_value="first"))
    second = DescriptiveProvider(label=Attribute(default_value="second"))
    before = first.get_attributes()
    model = ModelProfile(model=Attribute(default_value="demo"))
    first.add_models(model)
    after = first.get_attributes()

    assert before["models"] is None
    assert after["models"] is first.models
    assert after["models"].default_value[0] is model
    assert after["label"].default_value == "first"
    assert second.get_attributes()["label"].default_value == "second"
    assert second.get_attributes()["models"] is None


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_protocol_defaults_include_all_declared_parameters_and_configs(protocol):
    defaults = codec_for(protocol).defaults
    assert isinstance(defaults, Profile)
    assert defaults.get_attributes() == {}
    assert defaults.get_parameters()["content"] is defaults.content
    assert defaults.get_configs()["usage_mode"] is defaults.usage_mode
    assert "api_key" in defaults.get_parameters()
    assert set(defaults.get_parameters()).isdisjoint(defaults.get_configs())


def test_enumeration_reuses_resolution_rules_for_ambiguous_declarations():
    @dataclass(frozen=True, kw_only=True)
    class Ambiguous(ProviderProfile):
        feature: Attribute[str] | Config[str] = Attribute(default_value="ambiguous")

    with pytest.raises(ConfigurationError, match="feature.*exactly one"):
        Ambiguous().get_attributes()
