"""Reverse acceptance of complete drafts, direction, and successful usage records."""

import json
from contextlib import asynccontextmanager

import pytest

from my_llm import (
    Attribute,
    Config,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    SemanticLossError,
    Usage,
    omit,
)
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from tests.helpers.protocols import PROTOCOLS, chat_event, history
from tests.helpers.transport import FakeClient, FakeStream


def llm_for_chat(**parameters):
    return MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value="openai_chat"), reasoning_content=omit
        ),
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
        **parameters,
    )


@pytest.mark.parametrize("direction", ["input", "output"])
def test_combined_decoder_collects_every_loss_before_one_directional_hook(direction):
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "reason", "signature": "signed"},
                {"type": "text", "text": "complete tail"},
            ],
        }
    ]
    tools = [
        {"name": "first", "input_schema": {"type": "object", "minProperties": 1}},
        {"name": "second", "input_schema": {"type": "object", "minProperties": 2}},
    ]
    chunks = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    calls = []

    def hook(draft, losses, context):
        calls.append(draft)
        assert losses.origin_source == context.origin_source == direction
        assert context.source_protocol == "anthropic"
        assert context.target_protocol == "ollama"
        assert set(draft) == {"messages", "tools", "chunks"}
        assert draft["messages"][0]["content"] == "complete tail"
        assert len(draft["tools"]) == 2 and draft["chunks"][-1]["done"]
        assert [loss.origin_path for loss in losses] == [
            "$.messages[0].content[0]",
            "$.tools[0].input_schema",
            "$.tools[1].input_schema",
            "$.chunks[0].content_block",
        ]
        assert losses.items[0].origin_value is messages[0]["content"][0]
        assert losses.items[1].origin_value is tools[0]["input_schema"]
        assert losses.items[2].origin_value is tools[1]["input_schema"]
        assert losses.items[3].origin_value is chunks[0]["content_block"]
        draft["messages"][0]["content"] += " repaired"
        return {"replacement": "ignored"}

    def wrong_direction(*args):
        pytest.fail("the opposite direction hook ran")

    codec = MyCodec(
        "hook",
        input_semantic_loss_hook=hook if direction == "input" else wrong_direction,
        output_semantic_loss_hook=hook if direction == "output" else wrong_direction,
    )
    result = codec.convert(
        messages,
        tools,
        chunks=chunks,
        input_format="anthropic",
        target_format="ollama",
        origin_source=direction,
    )
    assert calls == [result]
    assert calls[0] is result
    assert result["messages"][0]["content"] == "complete tail repaired"


async def test_nonstream_output_hook_receives_only_the_complete_returned_draft(monkeypatch):
    client = FakeClient(
        [chat_event(reasoning="reason"), chat_event(content="complete tail", finish="stop")]
    )
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    drafts = []

    def hook(draft, losses, context):
        drafts.append(draft)
        assert context.stage == "result"
        assert context.origin_source == losses.origin_source == "output"
        assert draft["content"] == "complete tail"
        draft["content"] += " repaired"

    async with llm_for_chat(semantic_loss_protocol="hook", output_semantic_loss_hook=hook) as llm:
        result = await llm.generate(messages=[{"role": "user", "content": "question"}])
        assert drafts == [result]
        assert drafts[0] is result
        assert result["content"] == "complete tail repaired"
        assert llm.get_last_usage() is None
    assert client.closed and client.upstream.closed


async def test_nonstream_raise_contains_the_tail_after_the_losing_chunk(monkeypatch):
    client = FakeClient(
        [chat_event(reasoning="reason"), chat_event(content="complete tail", finish="stop")]
    )
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    async with llm_for_chat() as llm:
        with pytest.raises(SemanticLossError) as caught:
            await llm.generate(messages=[{"role": "user", "content": "question"}])
        error = caught.value
        assert error.draft == {"role": "assistant", "content": "complete tail"}
        assert error.context.stage == "result"
        assert error.losses.origin_source == "output"
        assert llm.get_last_usage() is None
        assert client.upstream.closed
    assert client.closed


async def test_request_hook_edits_the_same_draft_before_transport(monkeypatch):
    messages = history("anthropic")
    messages[-1]["content"][0]["content"] = [
        {"type": "text", "text": "tool text"},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/tool.png"}},
    ]
    messages.append({"role": "user", "content": "complete tail"})
    client = FakeClient([chat_event(content="answer", finish="stop")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    drafts = []
    upstream = ChatCompletionsTransport.upstream

    @asynccontextmanager
    async def checked_upstream(self, actual_client, request):
        assert drafts == [request]
        assert drafts[0] is request
        assert request["messages"][-1] == {"role": "user", "content": "hook addition"}
        async with upstream(self, actual_client, request) as events:
            yield events

    def hook(draft, losses, context):
        assert not client.requests
        assert losses.origin_source == context.origin_source == "input"
        assert context.source_protocol == "anthropic"
        assert context.target_protocol == "openai_chat"
        assert draft["messages"][-1]["content"] == "complete tail"
        drafts.append(draft)
        draft["messages"].append({"role": "user", "content": "hook addition"})

    monkeypatch.setattr(ChatCompletionsTransport, "upstream", checked_upstream)
    async with MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=True),
        ),
        stdio_protocol="anthropic",
        semantic_loss_protocol="hook",
        input_semantic_loss_hook=hook,
    ) as llm:
        result = await llm.generate(messages=messages)
        assert result["content"] == [{"type": "text", "text": "answer"}]
    assert len(drafts) == 1 and client.upstream.closed


async def test_stream_failure_after_terminal_keeps_the_last_successful_usage(monkeypatch):
    client = FakeClient([chat_event(content="first", finish="stop", usage={"total_tokens": 7})])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    async with llm_for_chat() as llm:
        first = await llm.generate(messages=[{"role": "user", "content": "first"}], stream=True)
        async with first:
            async for _ in first:
                pass
        assert llm.get_last_usage() == Usage(total=7)

        client.events = [chat_event(content="failed", finish="stop", usage={"total_tokens": 99})]
        client.failure = RuntimeError("failure after protocol terminal")
        failed = await llm.generate(messages=[{"role": "user", "content": "second"}], stream=True)
        with pytest.raises(RuntimeError, match="after protocol terminal"):
            async with failed:
                async for _ in failed:
                    pass
        assert failed.result is None
        assert llm.get_last_usage() == Usage(total=7)
        assert all(response.closed for response in client.streams)


@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_native_tool_events_have_complete_calls_only_after_source_eof(monkeypatch, stdio):
    class TrackedStream(FakeStream):
        exhausted = False

        async def __anext__(self):
            try:
                return await super().__anext__()
            except StopAsyncIteration:
                self.exhausted = True
                raise

    monkeypatch.setattr("tests.helpers.transport.FakeStream", TrackedStream)
    client = FakeClient(
        [
            chat_event(
                calls=[
                    {
                        "index": 0,
                        "id": "first_call",
                        "type": "function",
                        "function": {"name": "first", "arguments": '{"x":'},
                    },
                    {
                        "index": 1,
                        "id": "second_call",
                        "type": "function",
                        "function": {"name": "second", "arguments": '{"y":'},
                    },
                ]
            ),
            chat_event(
                calls=[
                    {"index": 0, "function": {"arguments": '{"nested":[1,2]}}'}},
                    {"index": 1, "function": {"arguments": "2}"}},
                ]
            ),
            chat_event(finish="tool_calls"),
        ]
    )
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    messages = MyCodec().convert(
        [{"role": "user", "content": "question"}],
        input_format="openai_chat",
        target_format=stdio,
    )["messages"]
    expected = {"first": {"x": {"nested": [1, 2]}}, "second": {"y": 2}}
    seen = set()
    async with MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=True),
        ),
        stdio_protocol=stdio,
    ) as llm:
        stream = await llm.generate(messages=messages, stream=True)
        async with stream:
            async for event in stream:
                if stdio == "openai_chat":
                    calls = [
                        call["function"]
                        for choice in event.get("choices", [])
                        for call in choice["delta"].get("tool_calls", [])
                    ]
                elif stdio == "ollama":
                    calls = [call["function"] for call in event["message"].get("tool_calls", [])]
                elif stdio == "gemini":
                    calls = [
                        {
                            "name": part["function_call"]["name"],
                            "arguments": part["function_call"]["args"],
                        }
                        for candidate in event.get("candidates", [])
                        for part in candidate["content"]["parts"]
                        if "function_call" in part
                    ]
                elif stdio == "anthropic":
                    assert event.get("delta", {}).get("type") != "input_json_delta"
                    block = event.get("content_block", {})
                    calls = (
                        [{"name": block["name"], "arguments": block["input"]}]
                        if block.get("type") == "tool_use"
                        else []
                    )
                else:
                    assert event["type"] != "response.function_call_arguments.delta"
                    items = (
                        [event["item"]]
                        if "item" in event
                        else event["response"]["output"]
                        if "response" in event
                        else [event]
                    )
                    calls = [
                        {"name": item["name"], "arguments": item["arguments"]}
                        for item in items
                        if item.get("type")
                        in ("function_call", "response.function_call_arguments.done")
                    ]
                for call in calls:
                    assert client.upstream.exhausted
                    arguments = call["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    assert arguments == expected[call["name"]]
                    seen.add(call["name"])
        assert seen == expected.keys()
        native_messages = stream.result if isinstance(stream.result, list) else [stream.result]
        final = MyCodec().convert(native_messages, input_format=stdio, target_format="openai_chat")[
            "messages"
        ][0]
        assert {
            call["function"]["name"]: json.loads(call["function"]["arguments"])
            for call in final["tool_calls"]
        } == expected
    assert client.closed and client.upstream.closed
