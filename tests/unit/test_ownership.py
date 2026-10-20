from copy import deepcopy

from my_llm import ModelProfile, MyCodec, MyLLM, ProviderProfile
from my_llm.profile import resolve_profiles
from tests.helpers.profiles import config
from tests.helpers.protocols import chat_event
from tests.helpers.transport import FakeClient


def test_profile_and_request_snapshots_are_independent():
    body = {"nested": {"value": "original"}}
    provider = ProviderProfile(extra_body=config("extra_body", body))
    llm = MyLLM(stdio_protocol="openai_chat", provider=provider, model=ModelProfile(model="demo"))
    body["nested"]["value"] = "changed"
    assert llm.provider.extra_body.default_value["nested"]["value"] == "original"
    assert llm.resolved_profile.get("extra_body")["nested"]["value"] == "original"
    decoder = llm.codec.decoder(llm.resolved_profile.protocol)
    options = llm.codec.transport(llm.resolved_profile.protocol).validate_parameters(
        llm.resolved_profile
    )
    request = decoder.request_parameters(llm.resolved_profile, options)
    request["extra_body"]["nested"]["value"] = "changed request"
    assert (
        decoder.request_parameters(llm.resolved_profile, options)["extra_body"]["nested"]["value"]
        == "original"
    )


async def test_lazy_generation_captures_history_and_request_options(monkeypatch):
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        lambda **kwargs: client,
    )
    history = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
    body = {"nested": {"value": "original"}}
    llm = MyLLM(
        stdio_protocol="openai_chat", provider=ProviderProfile(), model=ModelProfile(model="demo")
    )
    stream = await llm.generate(messages=history, extra_body=body, stream=True)
    history[0]["content"][0]["text"] = "changed"
    body["nested"]["value"] = "changed"
    async with stream:
        async for _ in stream:
            pass
    assert client.requests[0]["messages"][0]["content"] == "original"
    assert client.requests[0]["extra_body"]["nested"]["value"] == "original"
    result = stream.result
    result["content"] = "changed"
    assert stream.result["content"] == "ok"


def test_converter_never_mutates_input_or_profile():
    codec = MyCodec()
    messages = [{"role": "user", "content": "hi", "nested": {"value": [1]}}]
    before = deepcopy(messages)
    profile = resolve_profiles(ProviderProfile(), ModelProfile(model="demo"))
    for protocol in codec.codecs:
        converted = codec.convert(
            messages, input_format="openai_chat", target_format=protocol, profile=profile
        )
        assert "nested" not in converted["messages"][0]
        converted["messages"][0]["content" if protocol != "gemini" else "parts"] = "changed"
    assert messages == before
