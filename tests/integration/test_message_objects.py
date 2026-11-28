"""Native SDK representation follows each call's messages, including stream.result."""

import asyncio
import json
from copy import deepcopy

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from openai.types.chat import ChatCompletionMessage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from pydantic import field_serializer

from my_llm import (
    AssistantMessage,
    Attribute,
    Config,
    IncompleteStreamError,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    SemanticLossError,
    TextContent,
    ToolCall,
    UserMessage,
    omit,
)
from tests.helpers.json_output import install_sdk, make_llm, response_events
from tests.helpers.protocols import PROTOCOLS, chat_event
from tests.helpers.protocols import history as protocol_history
from tests.helpers.protocols import tools as protocol_tools
from tests.helpers.transport import FakeClient


def sdk_message(protocol):
    if protocol == "openai_chat":
        return ChatCompletionMessage(role="assistant", content="previous answer")
    if protocol == "openai_responses":
        return ResponseOutputMessage(
            id="msg_previous",
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text", text="previous answer", annotations=[], logprobs=[]
                )
            ],
        )
    return AnthropicMessage.model_construct(
        role="assistant", content=[TextBlock(type="text", text="previous answer")]
    )


def assert_answer(result, protocol, *, objects, text="answer"):
    if protocol == "openai_responses":
        assert isinstance(result, list) and len(result) == 1
        result = result[0]
        if objects:
            assert isinstance(result, ResponseOutputMessage)
            assert isinstance(result.content[0], ResponseOutputText)
            assert result.content[0].text == text
        else:
            assert isinstance(result, dict) and result["content"][0]["text"] == text
    elif protocol == "openai_chat":
        if objects:
            assert isinstance(result, ChatCompletionMessage) and result.content == text
        else:
            assert isinstance(result, dict) and result["content"] == text
    elif objects:
        assert isinstance(result, AnthropicMessage)
        assert isinstance(result.content[0], TextBlock) and result.content[0].text == text
    else:
        assert isinstance(result, dict) and result["content"][0]["text"] == text


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stdio", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("style", ["dict", "sdk", "mixed"])
async def test_representation_matrix(monkeypatch, protocol, stdio, streaming, cache, style):
    clients = install_sdk(monkeypatch, protocol, tools=False, text="answer")
    llm = make_llm(protocol, stdio, cache=cache)
    user = {"role": "user", "content": "question"}
    messages = [user] if style == "dict" else [sdk_message(stdio)]
    if style == "mixed":
        messages.insert(0, user)
    original = deepcopy(messages)
    next_user = {"role": "user", "content": "next question"}
    batches = [messages, [next_user] if cache else [*messages, next_user]]
    for index, batch in enumerate(batches):
        result = await llm.generate(messages=batch, stream=streaming, max_tokens=512)
        if streaming:
            assert result.result is None
            async with result:
                events = [event async for event in result]
            assert events and all(isinstance(event, dict) for event in events)
            assert result.result is result.result
            result = result.result
        assert_answer(result, stdio, objects=style != "dict" and (index == 0 or not cache))
    assert messages == original
    assert len(clients) == 2 and all(c.closed and c.upstream.closed for c in clients)
    if cache:
        key = "input" if protocol == "openai_responses" else "messages"
        first, second = [client.requests[0][key] for client in clients]
        assert second[: len(first)] == first


@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_sdk_input_is_preserved_through_protocol_inference(monkeypatch, stdio):
    install_sdk(monkeypatch, "openai_chat", tools=False, text="answer")
    llm = make_llm("openai_chat", None)
    result = await llm.generate(messages=[sdk_message(stdio)], tools=protocol_tools(stdio))
    assert llm.stdio_protocol == stdio
    assert_answer(result, stdio, objects=True)


@pytest.mark.parametrize("cache", [False, True])
async def test_sdk_input_losses_keep_original_content_objects(cache):
    thinking = ThinkingBlock(
        type="thinking", thinking="fixture reasoning", signature="fixture-signature"
    )
    source = AnthropicMessage.model_construct(
        role="assistant", content=[thinking, TextBlock(type="text", text="history")]
    )
    llm = make_llm("openai_chat", "anthropic", cache=cache)
    with pytest.raises(SemanticLossError) as caught:
        await llm.generate(messages=[source])
    assert caught.value.losses.items[0].origin_value is thinking
    assert caught.value.losses.items[0].origin_path == "$.messages[0].content[0]"
    assert source.content[0] is thinking


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("infer", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
async def test_sdk_ignored_fields_are_not_serialized_during_generation(
    monkeypatch, cache, infer, json_mode
):
    class Source(ChatCompletionMessage):
        ignored_payload: list[dict]

        @field_serializer("ignored_payload")
        def serialize_ignored(self, value):
            raise AssertionError("unsupported SDK fields must not enter input serialization")

    source = Source(role="assistant", content="history", ignored_payload=[{"unused": [1, 2, 3]}])
    events = (
        [
            response_events("openai_chat", tools=False, text="not JSON"),
            response_events("openai_chat", tools=False, text="still not JSON"),
            response_events("openai_chat", tools=True),
        ]
        if json_mode
        else [response_events("openai_chat", tools=False, text="answer")]
    )
    clients = [FakeClient(batch) for batch in events]
    pending = iter(clients)
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **_: next(pending))
    llm = make_llm("openai_chat", None if infer else "openai_chat", cache=cache)
    options = dict(messages=[source], tools=protocol_tools("openai_chat"))
    if json_mode:
        result = await llm.generate_json(**options, schema={"type": "object"}, max_retries=2)
        assert isinstance(result, ChatCompletionMessage)
        assert json.loads(result.content) == {"city": "上海"}
        assert len(llm.get_last_json_failures()) == 2
    else:
        result = await llm.generate(**options)
        assert_answer(result, "openai_chat", objects=True)
    assert source.ignored_payload == [{"unused": [1, 2, 3]}]
    assert all(client.closed and client.upstream.closed for client in clients)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stdio", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_calls_become_nested_sdk_objects(monkeypatch, protocol, stdio, streaming):
    clients = install_sdk(monkeypatch, protocol, tools=True)
    llm = make_llm(protocol, stdio)
    result = await llm.generate(
        messages=[sdk_message(stdio)], tools=protocol_tools(stdio), stream=streaming, max_tokens=512
    )
    if streaming:
        async with result:
            async for event in result:
                assert isinstance(event, dict)
        result = result.result
    if stdio == "openai_chat":
        call = result.tool_calls[0]
        assert call.function.name == "structured_output"
        assert json.loads(call.function.arguments) == {"city": "上海"}
    elif stdio == "openai_responses":
        call = next(item for item in result if isinstance(item, ResponseFunctionToolCall))
        assert call.name == "structured_output" and json.loads(call.arguments) == {"city": "上海"}
    else:
        call = next(part for part in result.content if isinstance(part, ToolUseBlock))
        assert call.name == "structured_output" and call.input == {"city": "上海"}
    assert clients[0].closed and clients[0].upstream.closed
    next_clients = install_sdk(monkeypatch, protocol, tools=False, text="answer")
    tool_result = protocol_history(stdio)[-1]
    continuation = [*result, tool_result] if stdio == "openai_responses" else [result, tool_result]
    completed = await llm.generate(messages=continuation, max_tokens=512)
    assert_answer(completed, stdio, objects=True)
    assert next_clients[0].closed and next_clients[0].upstream.closed


@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_concurrent_calls_keep_their_own_representation(monkeypatch, stdio):
    install_sdk(monkeypatch, "openai_chat", tools=False, text="answer")
    llm = make_llm("openai_chat", stdio)
    native, sdk = await asyncio.gather(
        llm.generate(messages=[{"role": "user", "content": "question"}]),
        llm.generate(messages=[sdk_message(stdio)]),
    )
    assert_answer(native, stdio, objects=False)
    assert_answer(sdk, stdio, objects=True)


@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_generate_json_returns_sdk_message_with_json_text(monkeypatch, stdio):
    install_sdk(monkeypatch, "openai_chat", tools=True)
    llm = make_llm("openai_chat", stdio)
    result = await llm.generate_json(
        messages=[sdk_message(stdio)],
        schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    if stdio == "openai_chat":
        text = result.content
    elif stdio == "openai_responses":
        text = result[0].content[0].text
    else:
        text = result.content[0].text
    assert json.loads(text) == {"city": "上海"}


@pytest.mark.parametrize("stdio", PROTOCOLS)
@pytest.mark.parametrize("cache", [False, True])
async def test_json_retries_keep_sdk_output_selection(monkeypatch, stdio, cache):
    clients = [
        FakeClient(response_events("openai_chat", tools=False, text="not JSON")),
        FakeClient(response_events("openai_chat", tools=True)),
    ]
    pending = iter(clients)
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **_: next(pending))
    llm = make_llm("openai_chat", stdio, cache=cache)
    result = await llm.generate_json(
        messages=[sdk_message(stdio)], schema={"type": "object"}, max_retries=1
    )
    assert_answer(
        result, stdio, objects=True, text=json.dumps({"city": "上海"}, ensure_ascii=False)
    )
    assert len(llm.get_last_json_failures()) == 1
    assert llm.get_last_usage().input == 2
    assert all(client.closed and client.upstream.closed for client in clients)


async def test_sdk_partial_message_keeps_error_type_and_closes_client(monkeypatch):
    client = FakeClient([chat_event(content="partial")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **_: client)
    llm = make_llm("openai_chat", "openai_chat")
    with pytest.raises(IncompleteStreamError) as caught:
        await llm.generate(messages=[sdk_message("openai_chat")])
    assert isinstance(caught.value.partial_message, ChatCompletionMessage)
    assert caught.value.partial_message.content == "partial"
    assert client.closed and client.upstream.closed


@pytest.mark.parametrize("stdio", [*PROTOCOLS, "middleware"])
async def test_json_final_result_owns_tool_arguments_after_output_hook(monkeypatch, stdio):
    events = response_events("openai_chat", tools=True, tool_name="weather")
    events.insert(0, chat_event(reasoning="fixture reasoning"))
    client = FakeClient(events)
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **_: client)
    retained = []

    def hook(draft, losses, context):
        assert isinstance(draft, AssistantMessage) and context.target_protocol == "middleware"
        call = next(part for part in draft.content if isinstance(part, ToolCall))
        call.arguments.update(city="北京", nested={"values": [1]})
        retained.append(call.arguments)

    llm = MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value="openai_chat"),
            reasoning_content=omit,
            feedback_reasoning_item=omit,
        ),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=True),
        ),
        stdio_protocol=stdio,
        semantic_loss_protocol="hook",
        output_semantic_loss_hook=hook,
    )
    messages = (
        [UserMessage("user", [TextContent("question")])]
        if stdio == "middleware"
        else [{"role": "user", "content": "question"}]
    )
    result = await llm.generate_json(
        messages=messages, schema={"type": "object"}, fake_tool_name="final_answer"
    )
    if stdio == "middleware":
        arguments = next(part.arguments for part in result.content if isinstance(part, ToolCall))
    elif stdio == "openai_chat":
        arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    elif stdio == "openai_responses":
        arguments = json.loads(
            next(item["arguments"] for item in result if item["type"] == "function_call")
        )
    else:
        arguments = next(part["input"] for part in result["content"] if part["type"] == "tool_use")
    assert arguments == {"city": "北京", "nested": {"values": [1]}}
    assert len(retained) == 1
    retained[0]["nested"]["values"].append(2)
    assert arguments["nested"]["values"] == [1]
    arguments["nested"]["values"].append(3)
    assert retained[0]["nested"]["values"] == [1, 2]
    assert client.closed and client.upstream.closed


@pytest.mark.parametrize(
    "protocol,native,model",
    [
        (
            "openai_chat",
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reason",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            ChatCompletionMessage,
        ),
        (
            "anthropic",
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "answer"},
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "f",
                        "input": {"nested": [1, True, None]},
                    },
                ],
            },
            AnthropicMessage,
        ),
        (
            "openai_responses",
            {
                "type": "message",
                "id": "m1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": [], "logprobs": []}
                ],
            },
            ResponseOutputMessage,
        ),
        (
            "openai_responses",
            {
                "type": "reasoning",
                "id": "r1",
                "summary": [{"type": "summary_text", "text": "reason"}],
            },
            ResponseReasoningItem,
        ),
        (
            "openai_responses",
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            ResponseFunctionToolCall,
        ),
        (
            "openai_responses",
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
            ResponseFunctionToolCallOutputItem,
        ),
    ],
)
def test_native_object_roundtrip_preserves_only_supplied_fields(protocol, native, model):
    codec = MyCodec()._codec(protocol)
    original = deepcopy(native)
    result = codec.to_object(native)
    assert isinstance(result, model)
    restored = codec.to_dict(result)
    assert restored == original and native == original
    restored.clear()
    assert codec.to_dict(result) == original
