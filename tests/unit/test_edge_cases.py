import pytest
from conftest import FakeClient, chat_event

from my_llm import (
    ConfigurationError,
    MessageValidationError,
    ModelProfile,
    ProviderProfile,
    ResponseIncompleteError,
    create,
    omit,
)
from my_llm.codec import ChatCompletionsCodec, ResponsesCodec
from my_llm.profiles import resolve_profiles


def test_previous_response_id_links_external_tool_result(profile_factory):
    codec = ResponsesCodec(profile_factory("responses"))
    messages = [{"role": "tool", "tool_call_id": "call_from_previous", "content": "result"}]
    request = codec.encode(messages, tools=omit, previous_response_id="resp_previous")
    assert request["previous_response_id"] == "resp_previous"
    assert request["input"] == [
        {"type": "function_call_output", "call_id": "call_from_previous", "output": "result"}
    ]
    with pytest.raises(MessageValidationError):
        codec.encode(messages, tools=omit)


def test_native_and_unified_history_calls_must_match(profile_factory, tool_call):
    codec = ResponsesCodec(profile_factory("responses"))
    native = {"type": "function_call", "name": "weather", "call_id": "other_id", "arguments": "{}"}
    with pytest.raises(MessageValidationError, match="must match"):
        codec.encode(
            [{"role": "assistant", "tool_calls": [tool_call], "response_output": [native]}],
            tools=omit,
        )


def test_assistant_tool_messages_convert_without_native_metadata(profile_factory, tool_call):
    request = ResponsesCodec(profile_factory("responses")).encode(
        [
            {"role": "assistant", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        tools=omit,
    )
    assert request["input"][0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "weather",
        "arguments": tool_call["function"]["arguments"],
    }


def test_custom_chat_effort_named_parameter(profile_factory):
    profile = profile_factory(
        supports_reasoning_effort=True, reasoning_effort_mapping={"high": "medium"}
    )
    request = ChatCompletionsCodec(profile).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning_effort"] == "medium"
    assert request["extra_body"]["thinking"]["type"] == "enabled"


def test_custom_response_summary_override(profile_factory):
    request = ResponsesCodec(
        profile_factory(
            "responses",
            reasoning_effort_parameter={"reasoning": {"summary": "auto", "effort": "{effort}"}},
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="low")
    assert request["reasoning"] == {"summary": "auto", "effort": "low"}


def test_reasoning_boolean_enable_and_mapping_disabled(profile_factory):
    request = ResponsesCodec(profile_factory("responses")).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning=True
    )
    assert request["reasoning"]["effort"] == "medium"
    request = ResponsesCodec(
        profile_factory(
            "responses", reasoning_enable_parameter=omit, reasoning_effort_parameter=omit
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high")
    assert "reasoning" not in request


def test_malicious_reasoning_binding_cannot_disable_stream(profile_factory):
    profile = profile_factory(reasoning_enable_parameter={"extra_body": {"stream": False}})
    with pytest.raises(ConfigurationError):
        ChatCompletionsCodec(profile).encode(
            [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
        )


def test_chat_name_supported_by_default(profile_factory):
    request = ChatCompletionsCodec(profile_factory()).encode(
        [{"role": "user", "name": "user1", "content": "x"}], tools=None
    )
    assert request["messages"][0]["name"] == "user1"
    assert "tools" not in request


def test_custom_responses_path(profile_factory):
    decoder = ResponsesCodec(
        profile_factory("responses", content_path="custom.text", reasoning_content_path=omit)
    ).decoder()
    assert decoder.feed({"type": "custom.text", "delta": "hello"})[0].content == "hello"
    assert decoder.feed({"type": "response.reasoning_summary_text.delta", "delta": "ignore"}) == []


async def test_responses_error_preserves_native_status_and_partial_content():
    events = [
        {"type": "response.output_text.delta", "delta": "partial"},
        {
            "type": "response.incomplete",
            "response": {
                "id": "resp_partial",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
    ]
    client = FakeClient(events)
    with pytest.raises(ResponseIncompleteError) as error:
        await create(
            provider=ProviderProfile(api_type="responses"),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
            client=client,
        )
    assert error.value.details == {"reason": "max_output_tokens"}
    assert error.value.partial_message == {
        "role": "assistant",
        "content": "partial",
        "response_id": "resp_partial",
        "response_status": "incomplete",
        "finish_reason": "incomplete",
    }
    assert client.upstream.closed


async def test_owned_client_closes_on_failure_and_unconsumed_stream(monkeypatch):
    clients = []

    def factory(**kwargs):
        client = FakeClient([], failure=RuntimeError("failure"))
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.invoke.AsyncOpenAI", factory)
    stream = await create(
        provider=ProviderProfile(),
        model="demo",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
    )
    await stream.aclose()
    assert clients == []
    with pytest.raises(RuntimeError):
        await create(
            provider=ProviderProfile(), model="demo", messages=[{"role": "user", "content": "x"}]
        )
    assert clients[0].closed and clients[0].upstream.closed


def test_invalid_native_call_id_not_replayed(profile_factory):
    with pytest.raises(MessageValidationError):
        ResponsesCodec(profile_factory("responses")).encode(
            [
                {
                    "role": "assistant",
                    "response_output": [
                        {"type": "function_call", "name": "tool", "arguments": "{}"}
                    ],
                }
            ],
            tools=omit,
        )


def test_model_string_whitespace_rejected():
    with pytest.raises(ConfigurationError):
        resolve_profiles(ProviderProfile(), ModelProfile(model=" "))


def test_repeated_chat_tool_name_does_not_duplicate(profile_factory):
    decoder = ChatCompletionsCodec(profile_factory()).decoder()
    decoder.feed(
        chat_event(
            calls=[{"index": 0, "id": "call", "function": {"name": "tool", "arguments": "{"}}]
        )
    )
    decoder.feed(
        chat_event(
            calls=[{"index": 0, "function": {"name": "tool", "arguments": "}"}}],
            finish="tool_calls",
        )
    )
    assert decoder.finish().tool_calls[0]["function"]["name"] == "tool"


def test_default_response_summary_retained_with_partial_binding(profile_factory):
    request = ResponsesCodec(
        profile_factory(
            "responses",
            reasoning_enable_parameter={"reasoning": {"effort": "medium"}},
            reasoning_effort_parameter={"reasoning": {}},
        )
    ).encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high")
    assert request["reasoning"] == {"summary": "detailed", "effort": "high"}


def test_response_summary_can_be_explicitly_omitted(profile_factory):
    profile = profile_factory(
        "responses",
        reasoning_enable_parameter={"reasoning": {"summary": omit}},
        reasoning_effort_parameter={"reasoning": {"summary": omit, "effort": "{effort}"}},
    )
    request = ResponsesCodec(profile).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning"] == {"effort": "high"}
