"""Repeated input batches remain idempotent across abandoned and failed requests."""

import asyncio
import json
from copy import deepcopy

import pytest
from google.genai import types
from ollama import Message as OllamaMessage

from my_llm import (
    AssistantMessage,
    MyLLM,
    ReasoningContent,
    SemanticLossError,
    TextContent,
)
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools
from tests.helpers.transport import FakeClient
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import profiles


def native_history(protocol, request):
    return request[{"openai_responses": "input", "gemini": "contents"}.get(protocol, "messages")]


async def consume(stream):
    async with stream:
        async for _ in stream:
            pass
    return stream.result


@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_unconsumed_stream_retry_does_not_duplicate_cached_input(upstream):
    provider, model = profiles(upstream)
    messages = [{"role": "user", "content": "retry-input"}]
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            abandoned = await llm.generate(messages=messages, stream=True)
            await abandoned.aclose()
            assert not sdk.requests
            await llm.generate(messages=deepcopy(messages))
            await llm.generate(messages=deepcopy(messages))
        requests = [json.loads(request.content) for request in sdk.requests]
        histories = [native_history(upstream, request) for request in requests]
        assert len(histories) == 2
        assert len(histories[0]) == len(histories[1]) == 1
        assert histories[0] == histories[1]


@pytest.mark.parametrize("exit_mode", ["failure", "cancel", "early"])
async def test_stream_retry_after_failure_cancel_or_early_close_keeps_one_batch(
    monkeypatch, exit_mode
):
    failure = {
        "failure": RuntimeError("injected stream failure"),
        "cancel": asyncio.CancelledError(),
        "early": None,
    }[exit_mode]
    client = FakeClient(stream_events("openai_chat"), failure=failure)
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    provider, model = profiles("openai_chat")
    batch = [{"role": "user", "content": "retry-input"}]
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="openai_chat",
        enable_message_cache=True,
    ) as llm:
        stream = await llm.generate(messages=batch, stream=True)
        if exit_mode == "early":
            await anext(stream)
            await stream.aclose()
        else:
            with pytest.raises(type(failure)):
                await consume(stream)
        assert stream.result is None
        assert client.streams[0].closed
        client.failure = None
        await llm.generate(messages=deepcopy(batch))
        assert len(client.requests) == 2
        assert client.requests[0]["messages"] == client.requests[1]["messages"]
        assert len(client.requests[1]["messages"]) == 1


@pytest.mark.parametrize("source", ["ollama", "gemini", "legacy_chat"])
async def test_retry_idless_tool_round_keeps_cached_call_ids_and_matching_results(
    monkeypatch, source
):
    batch = (
        [
            {"role": "assistant", "function_call": {"name": "weather", "arguments": "{}"}},
            {"role": "function", "name": "weather", "content": "Sunny"},
        ]
        if source == "legacy_chat"
        else history(source)[1:]
    )
    if source == "gemini":
        for part in batch[0]["parts"]:
            if "function_call" in part:
                part["function_call"].pop("id", None)
        batch[1]["parts"][0]["function_response"].pop("id", None)
    client = FakeClient(stream_events("openai_chat"), failure=RuntimeError("retry request"))
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    provider, model = profiles("openai_chat")
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="openai_chat" if source == "legacy_chat" else source,
        enable_message_cache=True,
    ) as llm:
        with pytest.raises(RuntimeError, match="retry request"):
            await llm.generate(messages=batch)
        first = deepcopy(client.requests[0]["messages"])
        client.failure = None
        await llm.generate(messages=deepcopy(batch))
        latest = client.requests[1]["messages"]
        assert latest == first
        calls = [call["id"] for message in latest for call in message.get("tool_calls", [])]
        results = [message["tool_call_id"] for message in latest if message["role"] == "tool"]
        assert len(calls) == 1 and calls == results


async def test_retry_anonymous_middleware_assistant_keeps_responses_item_ids(monkeypatch):
    batch = [AssistantMessage("assistant", [TextContent("visible"), ReasoningContent("reason")])]
    client = FakeClient(stream_events("openai_responses"), failure=RuntimeError("retry request"))
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI", lambda **kwargs: client
    )
    provider, model = profiles("openai_responses")
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="middleware",
        enable_message_cache=True,
    ) as llm:
        with pytest.raises(RuntimeError, match="retry request"):
            await llm.generate(messages=batch)
        first = deepcopy(client.requests[0]["input"])
        client.failure = None
        await llm.generate(messages=deepcopy(batch))
        latest = client.requests[1]["input"]
        assert latest == first
        assert len(latest) == 2
        assert len({item["id"] for item in latest}) == 2


async def test_consecutive_input_equality_ignores_key_order_but_detects_mutation_and_later_repeats():
    provider, model = profiles("openai_chat")
    batch = [{"role": "user", "content": "A"}]
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 61)
    ) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=batch)
            await llm.generate(messages=[{"content": "A", "role": "user"}])
            batch[0]["content"] = "B"
            await llm.generate(messages=batch)
            batch[0]["content"] = "A"
            await llm.generate(messages=batch)
        histories = [json.loads(request.content)["messages"] for request in sdk.requests]
        assert [len(value) for value in histories] == [1, 1, 2, 3]
        assert [message["content"] for message in histories[-1]] == ["A", "B", "A"]


async def test_retried_messages_keep_current_tools_headers_and_generation_controls():
    provider, model = profiles("openai_chat")
    batch = [{"role": "user", "content": "retry-input"}]
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 61)
    ) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            await llm.generate(
                messages=batch,
                tools=tools("openai_chat"),
                max_tokens=33,
                extra_headers={"x-retry": "first"},
            )
            await llm.generate(
                messages=deepcopy(batch),
                max_tokens=66,
                extra_headers={"x-retry": "second"},
            )
        first, latest = [json.loads(request.content) for request in sdk.requests]
        assert latest["messages"] == first["messages"]
        assert len(latest["messages"]) == 1
        assert first["tools"] and "tools" not in latest
        assert first["max_completion_tokens"] == 33
        assert latest["max_completion_tokens"] == 66
        assert sdk.requests[0].headers["x-retry"] == "first"
        assert sdk.requests[1].headers["x-retry"] == "second"


@pytest.mark.parametrize("source", ["ollama", "gemini"])
async def test_equal_sdk_model_batches_have_stable_fingerprints(source):
    message = (
        OllamaMessage(role="user", content="retry-input")
        if source == "ollama"
        else types.Content(role="user", parts=[types.Part(text="retry-input")])
    )
    provider, model = profiles("openai_chat")
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 61)
    ) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol=source,
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=[message])
            await llm.generate(messages=[deepcopy(message)])
        first, latest = [json.loads(request.content)["messages"] for request in sdk.requests]
        assert first == latest
        assert len(latest) == 1


async def test_adaptor_rejection_keeps_translation_cached_and_retries_deduplicate():
    provider, model = profiles("anthropic")
    batch = [{"role": "user", "content": "second-input", "name": "named-user"}]
    with LocalSDK("anthropic", wire_frames("anthropic", stream_events("anthropic"), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=[{"role": "user", "content": "first-input"}])
            with pytest.raises(SemanticLossError):
                await llm.generate(messages=batch)
            llm.set_semantic_loss_policy(semantic_loss_protocol="ignore")
            await llm.generate(messages=deepcopy(batch))
            await llm.generate(messages=deepcopy(batch))
        histories = [json.loads(request.content)["messages"] for request in sdk.requests]
        assert [len(value) for value in histories] == [1, 2, 2]
        assert [message["content"][0]["text"] for message in histories[-1]] == [
            "first-input",
            "second-input",
        ]


@pytest.mark.parametrize("upstream", ["anthropic", "gemini"])
async def test_retried_system_only_batch_retains_one_instruction_and_cached_user_history(upstream):
    provider, model = profiles(upstream)
    batch = [{"role": "system", "content": "retry-instruction"}]
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=[{"role": "user", "content": "first-input"}])
            abandoned = await llm.generate(messages=batch, stream=True)
            await abandoned.aclose()
            await llm.generate(messages=deepcopy(batch))
            await llm.generate(messages=deepcopy(batch))
        initial, *retries = [json.loads(request.content) for request in sdk.requests]
        assert len(retries) == 2
        for request in retries:
            assert native_history(upstream, request) == native_history(upstream, initial)
            instruction = (
                request["system"]
                if upstream == "anthropic"
                else request["systemInstruction"]["parts"]
            )
            assert [part["text"] for part in instruction] == ["retry-instruction"]


async def test_retry_runs_loss_hook_and_uses_current_controls_without_readding_hook_messages():
    hooks = []

    def hook(draft, losses, context):
        hooks.append(context)
        draft["messages"].append(
            {"role": "user", "content": [{"type": "text", "text": "hook-added"}]}
        )
        draft["max_tokens"] = 10 + len(hooks)

    provider, model = profiles("anthropic")
    batch = [{"role": "user", "content": "named-input", "name": "named-user"}]
    with LocalSDK("anthropic", wire_frames("anthropic", stream_events("anthropic"), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            semantic_loss_protocol="hook",
            input_semantic_loss_hook=hook,
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=batch)
            await llm.generate(messages=deepcopy(batch))
        first, retry = [json.loads(request.content) for request in sdk.requests]
        assert len(hooks) == 2
        assert retry["messages"] == first["messages"]
        assert [message["content"][0]["text"] for message in retry["messages"]] == [
            "named-input",
            "hook-added",
        ]
        assert first["max_tokens"] == 11 and retry["max_tokens"] == 12
