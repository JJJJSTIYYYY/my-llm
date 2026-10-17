"""Pytest fixtures; reusable samples and utilities live in tests.helpers."""

from copy import deepcopy

import pytest

from my_llm import ModelProfile, ProviderProfile, resolve_profiles
from tests.helpers.protocols import CHAT_HISTORY, CHAT_TOOL


@pytest.fixture
def profile_factory():
    def make(protocol="openai_chat", *, provider=None, **model):
        return resolve_profiles(
            provider or ProviderProfile(protocol=protocol), ModelProfile(model="demo", **model)
        )

    return make


@pytest.fixture
def function_tool():
    return deepcopy(CHAT_TOOL)


@pytest.fixture
def tool_call():
    return deepcopy(CHAT_HISTORY[1]["tool_calls"][0])
