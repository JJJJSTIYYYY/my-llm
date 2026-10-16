import pytest
from conftest import FakeClient, chat_event

from my_llm import MessageValidationError, ModelProfile, MyLLM, ProviderProfile


async def test_mixed_protocol_history_is_rejected_before_request(monkeypatch):
    client = FakeClient([])
    monkeypatch.setattr("my_llm.invoke.AsyncOpenAI", lambda **kwargs: client)
    llm = MyLLM(provider=ProviderProfile(), model=ModelProfile(model="demo"))
    with pytest.raises(MessageValidationError):
        await llm.generate(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "model", "parts": [{"text": "different"}]},
            ]
        )
    assert not client.requests


async def test_previous_response_id_allows_external_tool_results(monkeypatch):
    from conftest import response_completed

    client = FakeClient([response_completed()])
    monkeypatch.setattr("my_llm.invoke.AsyncOpenAI", lambda **kwargs: client)
    llm = MyLLM(
        provider=ProviderProfile(protocol="openai_responses"),
        model=ModelProfile(model="demo"),
        stdio_protocol="openai_responses",
    )
    result = [{"type": "function_call_output", "call_id": "server_call", "output": "ok"}]
    with pytest.raises(MessageValidationError):
        await llm.generate(messages=result)
    await llm.generate(messages=result, previous_response_id="resp_previous")
    assert client.requests[0]["input"] == result
    assert client.requests[0]["previous_response_id"] == "resp_previous"


async def test_stdio_protocol_matches_tools_before_ambiguous_messages(monkeypatch):
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    monkeypatch.setattr("my_llm.invoke.AsyncOpenAI", lambda **kwargs: client)
    llm = MyLLM(
        provider=ProviderProfile(), model=ModelProfile(model="demo", supports_function_calling=True)
    )
    result = await llm.generate(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "weather", "input_schema": {"type": "object"}}],
    )
    assert llm.stdio_protocol == "anthropic"
    assert result["role"] == "assistant"
    assert result["content"][0]["text"] == "ok"
