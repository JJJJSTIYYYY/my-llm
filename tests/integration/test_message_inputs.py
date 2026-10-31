import pytest

from my_llm import Attribute, Config, MessageValidationError, ModelProfile, MyLLM, ProviderProfile
from tests.helpers.profiles import config
from tests.helpers.protocols import chat_event
from tests.helpers.transport import FakeClient


async def test_mixed_protocol_history_is_rejected_before_request(monkeypatch):
    client = FakeClient([])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        lambda **kwargs: client,
    )
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(model=Attribute(default_value="demo")),
    )
    with pytest.raises(MessageValidationError):
        await llm.generate(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "model", "parts": [{"text": "different"}]},
            ]
        )
    assert not client.requests


async def test_previous_response_id_allows_external_tool_results(monkeypatch):
    from tests.helpers.protocols import response_completed

    client = FakeClient([response_completed()])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        lambda **kwargs: client,
    )
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_responses")),
        model=ModelProfile(
            model=Attribute(default_value="demo"), supports_function_calling=config(True)
        ),
        stdio_protocol="openai_responses",
    )
    result = [{"type": "function_call_output", "call_id": "server_call", "output": "ok"}]
    with pytest.raises(MessageValidationError, match="no usable history"):
        await llm.generate(messages=result)
    await llm.generate(messages=result, previous_response_id="resp_previous")
    assert client.requests[0]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "server_call",
            "output": [{"type": "input_text", "text": "ok"}],
        }
    ]
    assert client.requests[0]["previous_response_id"] == "resp_previous"
    await llm.aclose()


async def test_stdio_protocol_matches_tools_before_ambiguous_messages(monkeypatch):
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        lambda **kwargs: client,
    )
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(
            model=Attribute(default_value="demo"), supports_function_calling=config(True)
        ),
    )
    result = await llm.generate(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "weather", "input_schema": {"type": "object"}}],
    )
    assert llm.stdio_protocol == "anthropic"
    assert result["role"] == "assistant"
    assert result["content"][0]["text"] == "ok"
