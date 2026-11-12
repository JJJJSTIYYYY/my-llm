"""Authentication control validation, omission and lifecycle contracts."""

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    InputParameter,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    generate,
    omit,
)
from tests.helpers.protocols import chat_event
from tests.helpers.transport import FakeClient


@pytest.fixture
def clients(monkeypatch):
    created = []

    def factory(**kwargs):
        client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
        created.append((kwargs, client))
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    return created


def make_llm(api_key=None):
    return MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat"), api_key=api_key),
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
    )


@pytest.mark.parametrize("value", ["", "  ", 0, False, [], {}, object()])
async def test_invalid_api_key_fails_before_client_creation(clients, value):
    async with make_llm() as llm:
        with pytest.raises(ConfigurationError, match="api_key must be a non-empty string"):
            await llm.generate(messages=[{"role": "user", "content": "hi"}], api_key=value)
    assert not clients


@pytest.mark.parametrize("declaration", [omit, InputParameter(path="$.api_key", supports=False)])
async def test_explicit_api_key_requires_an_enabled_declaration(clients, declaration):
    async with make_llm(declaration) as llm:
        with pytest.raises(ConfigurationError, match="supported Profile declaration"):
            await llm.generate(messages=[{"role": "user", "content": "hi"}], api_key="call-key")
        await llm.generate(messages=[{"role": "user", "content": "hi"}], api_key=omit)
    assert clients[0][0] == {}
    assert clients[0][1].close_count == 1


@pytest.mark.parametrize("path", ["$.api_key", "$.extra_headers.Authorization"])
async def test_omit_removes_only_this_calls_configured_authentication(clients, path):
    async with make_llm(InputParameter(path=path, default_value="profile-key")) as llm:
        original = dict(llm.client_parameters)
        for options in ({"api_key": omit}, {}, {"api_key": omit}, {"api_key": None}):
            await llm.generate(messages=[{"role": "user", "content": "hi"}], **options)
        assert llm.client_parameters == original
        assert llm.resolved_profile.get("api_key") == "profile-key"
        assert len(clients) == 2
        omitted, inherited = clients[0][0], clients[1][0]
        assert "api_key" not in omitted
        assert "Authorization" not in omitted.get("default_headers", {})
        assert inherited == original
        assert all(not client.closed for _, client in clients)
    assert all(client.close_count == 1 for _, client in clients)


async def test_unconsumed_override_stream_creates_no_client(clients):
    async with make_llm() as llm:
        stream = await llm.generate(
            messages=[{"role": "user", "content": "hi"}], api_key="call-key", stream=True
        )
        await stream.aclose()
    assert not clients


async def test_one_shot_helper_forwards_api_key_and_closes_client(clients):
    result = await generate(
        provider=ProviderProfile(
            protocol=Config(default_value="openai_chat"),
            api_key=InputParameter(path="$.api_key", default_value="profile-key"),
        ),
        model="demo",
        stdio_protocol="openai_chat",
        messages=[{"role": "user", "content": "hi"}],
        api_key="call-key",
    )
    assert result["content"] == "ok"
    assert clients[0][0] == {"api_key": "call-key"}
    assert clients[0][1].close_count == 1
