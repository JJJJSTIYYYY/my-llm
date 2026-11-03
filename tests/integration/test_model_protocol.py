"""Models sharing a provider can select independent upstream protocols."""

from dataclasses import replace

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile
from tests.helpers.protocols import stream_events
from tests.helpers.transport import FakeClient


async def test_models_sharing_a_provider_use_their_selected_protocols(monkeypatch):
    chat_client = FakeClient(stream_events("openai_chat"))
    responses_client = FakeClient(stream_events("openai_responses"))
    del chat_client.responses
    del responses_client.chat
    monkeypatch.setattr(
        "my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: chat_client
    )
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI", lambda **kwargs: responses_client
    )
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    inherited_model = ModelProfile(model=Attribute(default_value="chat-model"))
    overridden_model = ModelProfile(
        model=Attribute(default_value="responses-model"),
        protocol=Config(default_value="openai_responses"),
    )

    async with (
        MyLLM(provider=provider, model=inherited_model, stdio_protocol="openai_chat") as chat,
        MyLLM(provider=provider, model=overridden_model, stdio_protocol="openai_chat") as responses,
    ):
        for llm in (chat, responses):
            result = await llm.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=32)
            assert result["role"] == "assistant"
            assert result["content"] == "Hello"
        assert chat.resolved_profile.protocol == "openai_chat"
        assert responses.resolved_profile.protocol == "openai_responses"

    chat_request = chat_client.requests[0]
    responses_request = responses_client.requests[0]
    assert chat_request["model"] == "chat-model"
    assert chat_request["messages"] == [{"role": "user", "content": "hi"}]
    assert chat_request["max_completion_tokens"] == 32
    assert responses_request["model"] == "responses-model"
    assert responses_request["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
    assert responses_request["max_output_tokens"] == 32
    assert provider.protocol.default_value == "openai_chat"
    assert inherited_model.protocol is None
    for client in (chat_client, responses_client):
        assert client.close_count == 1
        assert all(stream.closed for stream in client.streams)


async def test_model_protocol_changes_can_be_compared_and_reused():
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(model=Attribute(default_value="demo"))
    changed_model = replace(model, protocol=Config(default_value="openai_responses"))

    async with MyLLM(provider=provider, model=model) as original:
        result = original.compare(provider, changed_model)
        assert isinstance(result, tuple)
        assert result[0].protocol == "openai_responses"
        assert result[1] != original.profile_hash
        async with MyLLM(
            provider=provider, model=changed_model, resolve_profile_with_hash=result
        ) as replacement:
            assert (
                replacement.resolved_profile.parameter("max_tokens").path == "$.max_output_tokens"
            )
            assert replacement.compare(provider, changed_model) is True
            assert (
                replacement.compare(
                    replace(provider, protocol=Config(default_value="anthropic")), changed_model
                )
                is True
            )
        assert original.resolved_profile.protocol == "openai_chat"
