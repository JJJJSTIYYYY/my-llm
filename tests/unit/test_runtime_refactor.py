"""Per-instance protocol state, loss policy ownership, and private usage records."""

import json

import pytest

from my_llm import (
    Attribute,
    Config,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    SemanticLossError,
)
from my_llm.middleware import Usage
from my_llm.registry import codec_for
from my_llm.semantic_loss import SemanticLossContext, collect_losses, originated, report_loss
from my_llm.stream._state import StreamState
from tests.helpers.protocols import PROTOCOLS, chat_event
from tests.helpers.transport import FakeClient


def make_llm(*, stdio_protocol="openai_chat", **parameters):
    return MyLLM(
        stdio_protocol=stdio_protocol,
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=True),
        ),
        **parameters,
    )


def assert_no_usage(value):
    if isinstance(value, dict):
        assert not (
            {"usage", "usage_metadata", "usageMetadata", "prompt_eval_count", "eval_count"}
            & value.keys()
        )
        for child in value.values():
            assert_no_usage(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_usage(child)


def test_registry_and_llms_create_independent_components():
    first, second = make_llm(), make_llm()
    for protocol, codec in first.codec.codecs.items():
        other = second.codec.codecs[protocol]
        assert codec is not other
        for component in ("encoder", "decoder", "transport", "semantic_loss_adaptor"):
            assert getattr(codec, component) is not getattr(other, component)
    assert codec_for("openai_chat") is not codec_for("openai_chat")
    assert MyCodec().codecs["openai_chat"] is not MyCodec().codecs["openai_chat"]
    first.codec.codecs["openai_responses"].defaults.reasoning_enable.default_value["effort"] = "low"
    assert (
        second.codec.codecs["openai_responses"].defaults.reasoning_enable.default_value["effort"]
        == "medium"
    )


def test_stdio_binding_cannot_be_reassigned():
    llm = make_llm()
    with pytest.raises(AttributeError):
        llm.stdio_protocol = "ollama"
    with pytest.raises(AttributeError):
        llm.resolved_profile = make_llm().resolved_profile
    assert llm.stdio_protocol == "openai_chat"
    assert llm.resolved_profile.protocol == "openai_chat"


def test_loss_policy_reconfiguration_is_local_to_one_llm():
    first, second = make_llm(), make_llm()
    previous = second.codec.semantic_loss_adaptor

    def hook(draft, losses, context):
        pass

    first.set_semantic_loss_policy(semantic_loss_protocol="hook", input_semantic_loss_hook=hook)
    assert second.codec.semantic_loss_adaptor is previous
    assert first.codec.semantic_loss_adaptor is not previous
    assert first.codec.semantic_loss_adaptor.input_semantic_loss_hook is hook
    assert all(
        codec.semantic_loss_adaptor is first.codec.semantic_loss_adaptor
        for codec in first.codec.codecs.values()
    )


def test_stream_materialization_keeps_source_location_for_response_losses():
    state = StreamState("demo")
    source = {"native_reasoning": {"text": "step", "signature": "opaque"}}
    operation = originated(
        {
            "type": "block_delta",
            "ref": ("reasoning", 0, 0),
            "kind": "thinking",
            "field": "thinking",
            "value": "step",
        },
        "$.chunks[7].native_reasoning",
        source,
    )
    emitted = state.feed([operation])
    operation["value"] = "caller edit"
    delta = next(event for event in emitted if event["type"] == "delta")
    assert delta.origin_path == "$.chunks[7].native_reasoning"
    assert delta.origin_value == source
    state.feed([{"type": "finish", "reason": "stop"}])
    final = state.finish()
    block = final[-1]["messages"][0]["content"][0]
    assert block.origin_path == delta.origin_path
    assert block.origin_value == source
    assert block["thinking"] == "step"
    with pytest.raises(SemanticLossError) as caught:
        with collect_losses(SemanticLossContext("gemini", "anthropic", "output")) as (losses, _):
            report_loss(block, "signature cannot cross protocols")
            losses.complete([])
    assert caught.value.losses.items[0].origin_value is source
    source["native_reasoning"]["text"] = "after loss"
    assert block["thinking"] == "step"


@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_stream_records_complete_usage_without_exposing_counters(monkeypatch, stdio):
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    events = [chat_event(content="ok"), chat_event(finish="stop")]
    events.append({**chat_event(usage=usage), "choices": []})
    client = FakeClient(events)
    monkeypatch.setattr(
        "my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **parameters: client
    )
    messages = MyCodec().convert(
        messages=[{"role": "user", "content": "hi"}],
        input_format="openai_chat",
        target_format=stdio,
    )["messages"]
    async with make_llm(stdio_protocol=stdio) as llm:
        assert llm.get_last_usage() is None
        stream = await llm.generate(messages=messages, stream=True)
        async with stream:
            async for event in stream:
                assert_no_usage(event)
                assert llm.get_last_usage() is None
        assert llm.get_last_usage() == Usage(input=11, output=7, total=18, cached=3, reasoning=2)
        record = llm.get_last_usage()
        assert record is not llm.get_last_usage()
        completed = stream.result if isinstance(stream.result, list) else [stream.result]
        assert (
            MyCodec().convert(messages=completed, input_format=stdio, target_format="openai_chat")[
                "messages"
            ][0]["content"]
            == "ok"
        )
        assert_no_usage(stream.result)


async def test_missing_usage_fields_stay_empty_and_nonstream_calls_expose_no_usage(monkeypatch):
    client = FakeClient(
        [chat_event(content="ok"), chat_event(finish="stop", usage={"total_tokens": 5})]
    )
    monkeypatch.setattr(
        "my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **parameters: client
    )
    async with make_llm() as llm:
        stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
        async with stream:
            async for _ in stream:
                pass
        assert llm.get_last_usage() == Usage(total=5)
        result = await llm.generate(messages=[{"role": "user", "content": "hi"}])
        assert_no_usage(result)
        assert llm.get_last_usage() is None


async def test_early_close_preserves_last_completed_usage(monkeypatch):
    client = FakeClient(
        [chat_event(content="ok"), chat_event(finish="stop", usage={"total_tokens": 5})]
    )
    monkeypatch.setattr(
        "my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **parameters: client
    )
    async with make_llm() as llm:
        first = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
        async with first:
            async for _ in first:
                pass
        second = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
        await anext(second)
        await second.aclose()
        assert second.result is None
        assert llm.get_last_usage() == Usage(total=5)


@pytest.mark.parametrize("stdio", ["openai_chat", "ollama"])
async def test_all_tool_calls_are_emitted_together_with_complete_arguments(monkeypatch, stdio):
    client = FakeClient(
        [
            chat_event(
                calls=[
                    {
                        "index": 0,
                        "id": "a",
                        "type": "function",
                        "function": {"name": "first", "arguments": '{"x":'},
                    },
                    {
                        "index": 1,
                        "id": "b",
                        "type": "function",
                        "function": {"name": "second", "arguments": '{"y":'},
                    },
                ]
            ),
            chat_event(
                calls=[
                    {"index": 0, "function": {"arguments": "1}"}},
                    {"index": 1, "function": {"arguments": "2}"}},
                ]
            ),
            chat_event(finish="tool_calls"),
        ]
    )
    monkeypatch.setattr(
        "my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **parameters: client
    )
    async with make_llm(stdio_protocol=stdio) as llm:
        stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
        emitted = [event async for event in stream]
        if stdio == "openai_chat":
            chunks = [
                choice["delta"]["tool_calls"]
                for event in emitted
                for choice in event.get("choices", [])
                if choice["delta"].get("tool_calls")
            ]
            assert len(chunks) == 1
            assert [json.loads(call["function"]["arguments"]) for call in chunks[0]] == [
                {"x": 1},
                {"y": 2},
            ]
        else:
            chunks = [
                event["message"]["tool_calls"]
                for event in emitted
                if event["message"].get("tool_calls")
            ]
            assert len(chunks) == 1
            assert [call["function"]["arguments"] for call in chunks[0]] == [{"x": 1}, {"y": 2}]
