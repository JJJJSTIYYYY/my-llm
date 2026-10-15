"""Public naming and documentation contracts."""

import inspect
from dataclasses import fields

import pytest

import my_llm
from my_llm import MessageChunk, MessageStream, ModelProfile, MyLLM, ProviderProfile, generate


def test_public_generation_names():
    assert {"generate", "MyLLM"} <= set(my_llm.__all__)
    assert inspect.iscoroutinefunction(generate)
    assert inspect.iscoroutinefunction(MyLLM.generate)
    assert not hasattr(my_llm, "create")
    assert not hasattr(my_llm, "Invoke")
    assert not hasattr(MyLLM, "create")


def test_exported_library_objects_have_docstrings():
    for name in my_llm.__all__:
        value = getattr(my_llm, name)
        if callable(value) and getattr(value, "__module__", "").startswith("my_llm"):
            assert inspect.getdoc(value), name
    for value in (
        MessageStream.aclose,
        ProviderProfile.add_models,
        ProviderProfile.list_models,
        ProviderProfile.delete_model,
        MessageChunk.__add__,
        MessageChunk.to_message,
        MessageChunk.has_delta,
        MessageChunk.is_finished,
    ):
        assert inspect.getdoc(value)


@pytest.mark.parametrize("interface", [generate, MyLLM.generate, MyLLM])
def test_generation_docstrings_explain_all_parameters(interface):
    doc = inspect.getdoc(interface)
    for name in inspect.signature(interface).parameters:
        if name != "self":
            assert f"{name}:" in doc, name


@pytest.mark.parametrize("profile", [ProviderProfile, ModelProfile])
def test_profile_docstrings_explain_all_configurable_fields(profile):
    doc = inspect.getdoc(profile)
    for field in fields(profile):
        assert f"{field.name}:" in doc, field.name
