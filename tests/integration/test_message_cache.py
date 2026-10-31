"""Incremental histories retain translated values across real SDK requests."""

import json
from copy import deepcopy
from uuid import UUID

import pytest

from my_llm import (
    AssistantMessage,
    ConfigurationError,
    ItemIdentity,
    MyLLM,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    UserMessage,
)
from my_llm.codec.base import decoder as decoder_module
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import profiles, user_message


def request_history(protocol, request):
    return request[{"openai_responses": "input", "gemini": "contents"}.get(protocol, "messages")]


def request_limit(protocol, request):
    if protocol == "gemini":
        return request["generationConfig"]["maxOutputTokens"]
    if protocol == "ollama":
        return request["options"]["num_predict"]
    return request[
        {
            "openai_chat": "max_completion_tokens",
            "openai_responses": "max_output_tokens",
            "anthropic": "max_tokens",
        }[protocol]
    ]


def system_texts(protocol, request):
    if protocol == "gemini":
        value = request["systemInstruction"]
        return [part["text"] for part in value["parts"]]
    value = request["system"]
    return [value] if isinstance(value, str) else [part["text"] for part in value]


async def consume(stream):
    async with stream:
        async for _ in stream:
            pass
    return stream.result


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_incremental_native_histories_cross_all_protocol_paths(upstream, stdio):
    provider, model = profiles(upstream)
    initial = history(stdio)
    before = deepcopy(initial)
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 53)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol=stdio,
            enable_message_cache=True,
        ) as llm:
            key = llm._message_cache_key
            assert UUID(key).version == 4
            result = await llm.generate(
                messages=initial,
                tools=tools(stdio),
                max_tokens=33,
                extra_headers={"x-cache": "first"},
            )
            initial_request = json.loads(sdk.requests[-1].content)
            previous = result if isinstance(result, list) else [result]
            delta = [*previous, user_message(stdio, "next-input", media=False)]
            delta_before = deepcopy(delta)
            stream = await llm.generate(
                messages=delta,
                stream=True,
                max_tokens=66,
                extra_headers={"x-cache": "second"},
            )
            await consume(stream)
            latest = json.loads(sdk.requests[-1].content)
            first_history = request_history(upstream, initial_request)
            latest_history = request_history(upstream, latest)
            assert latest_history[: len(first_history)] == first_history
            assert "next-input" in json.dumps(latest_history[len(first_history) :])
            # The caller supplied the assistant response exactly once.
            assert json.dumps(latest_history).count("Hello") == 1
            assert initial_request.get("tools") and not latest.get("tools")
            assert request_limit(upstream, initial_request) == 33
            assert request_limit(upstream, latest) == 66
            assert sdk.requests[0].headers["x-cache"] == "first"
            assert sdk.requests[1].headers["x-cache"] == "second"
            assert all(
                field not in latest
                for field in ("enable_message_cache", "message_cache_key", "message_transform")
            )
            assert initial == before and delta == delta_before
            # Ollama binds each distinct request-header set to its SDK client.
            assert sdk.created_clients == (2 if upstream == "ollama" else 1)
            assert key in llm.codec._message_cache
        assert key not in llm.codec._message_cache
    assert all(body.closed for body in sdk.bodies)


@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_incremental_middleware_histories_keep_typed_public_io(upstream):
    provider, model = profiles(upstream)
    first = UserMessage("user", [TextContent("first-input")])
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="middleware",
            enable_message_cache=True,
        ) as llm:
            response = await llm.generate(messages=[first])
            assert isinstance(response, AssistantMessage)
            await llm.generate(
                messages=[response, UserMessage("user", [TextContent("next-input")])]
            )
        histories = [request_history(upstream, json.loads(item.content)) for item in sdk.requests]
        assert len(histories[0]) == 1 and len(histories[1]) == 3
        assert histories[1][:1] == histories[0]
        assert json.dumps(histories[1]).count("Hello") == 1
        assert first.content[0].text == "first-input"


@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_cache_is_disabled_by_default(upstream):
    provider, model = profiles(upstream)
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(provider=provider, model=model, stdio_protocol="openai_chat") as llm:
            await llm.generate(messages=[{"role": "user", "content": "first-input"}])
            await llm.generate(messages=[{"role": "user", "content": "second-input"}])
        histories = [request_history(upstream, json.loads(item.content)) for item in sdk.requests]
        assert len(histories[0]) == len(histories[1]) == 1
        assert "first-input" not in json.dumps(histories[1])


@pytest.mark.parametrize("upstream", ["anthropic", "gemini"])
async def test_system_only_deltas_accumulate_and_survive_later_user_batches(upstream):
    provider, model = profiles(upstream)
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            with pytest.raises(ConfigurationError, match="non-system"):
                await llm.generate(messages=[{"role": "system", "content": "discarded"}])
            assert not sdk.requests
            await llm.generate(
                messages=[
                    {"role": "system", "content": "initial-instruction"},
                    {"role": "user", "content": "first-input"},
                ]
            )
            await llm.generate(messages=[{"role": "system", "content": "next-instruction"}])
            await llm.generate(messages=[{"role": "user", "content": "next-input"}])
        requests = [json.loads(item.content) for item in sdk.requests]
        assert system_texts(upstream, requests[0]) == ["initial-instruction"]
        assert system_texts(upstream, requests[1]) == ["initial-instruction", "next-instruction"]
        assert system_texts(upstream, requests[2]) == ["initial-instruction", "next-instruction"]
        assert request_history(upstream, requests[1]) == request_history(upstream, requests[0])
        assert len(request_history(upstream, requests[2])) == 2


@pytest.mark.parametrize("upstream", ["anthropic", "gemini"])
async def test_system_only_deltas_remain_invalid_without_caching(upstream):
    provider, model = profiles(upstream)
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(provider=provider, model=model, stdio_protocol="openai_chat") as llm:
            await llm.generate(messages=[{"role": "user", "content": "first-input"}])
            with pytest.raises(ConfigurationError, match="non-system"):
                await llm.generate(messages=[{"role": "system", "content": "next-instruction"}])
        assert len(sdk.requests) == 1


async def test_tool_associations_only_scan_current_batches_and_repair_invalid_deltas(monkeypatch):
    calls = []
    original = decoder_module.validate_messages

    def validate(messages, **kwargs):
        if kwargs.get("history") is not None:
            calls.append(deepcopy(messages))
        return original(messages, **kwargs)

    monkeypatch.setattr(decoder_module, "validate_messages", validate)
    provider, model = profiles("openai_chat")
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 61)
    ) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=history("openai_chat"))
            delta = history("openai_chat")[1:]
            delta[0]["tool_calls"][0]["id"] = "call_2"
            delta[1]["tool_call_id"] = "call_2"
            await llm.generate(messages=delta)
            await llm.generate(
                messages=[{"role": "tool", "tool_call_id": "missing", "content": "bad"}]
            )
            await llm.generate(messages=[{"role": "user", "content": "next-input"}])
        assert [len(batch) for batch in calls] == [3, 2, 1, 1]
        assert len(sdk.requests) == 4
        latest = json.loads(sdk.requests[-1].content)["messages"]
        assert len(latest) == 6
        assert "missing" not in json.dumps(latest)


async def test_loss_hook_edits_are_request_local_and_do_not_update_cached_translation():
    drafts = []
    cached_at_hook = []

    def hook(draft, losses, context):
        assert context.origin_source == "input" and context.stage == "request"
        cached_at_hook.append(llm.codec._message_cache.get(llm._message_cache_key))
        draft["messages"].append(
            {"role": "user", "content": [{"type": "text", "text": "hook-added"}]}
        )
        draft["max_tokens"] = 11
        drafts.append(draft)

    provider, model = profiles("anthropic")
    with LocalSDK("anthropic", wire_frames("anthropic", stream_events("anthropic"), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            semantic_loss_protocol="hook",
            input_semantic_loss_hook=hook,
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=[{"role": "user", "content": "named", "name": "n"}])
            drafts[0]["messages"][0]["content"][0]["text"] = "changed-after-caching"
            await llm.generate(messages=[{"role": "user", "content": "next-input"}], max_tokens=22)
        first, second = [json.loads(item.content) for item in sdk.requests]
        assert [part["content"][0]["text"] for part in first["messages"]] == [
            "named",
            "hook-added",
        ]
        assert [part["content"][0]["text"] for part in second["messages"]] == [
            "named",
            "next-input",
        ]
        assert first["max_tokens"] == 11 and second["max_tokens"] == 22
        assert len(drafts) == 1
        assert [part["content"][0]["text"] for part in cached_at_hook[0]] == ["named"]


@pytest.mark.parametrize("upstream", ["anthropic", "gemini"])
async def test_loss_hook_system_replacements_do_not_modify_cached_translation(upstream):
    def hook(draft, losses, context):
        if upstream == "anthropic":
            draft["system"] = "hook-instruction"
        else:
            draft["config"]["system_instruction"] = "hook-instruction"

    provider, model = profiles(upstream)
    with LocalSDK(upstream, wire_frames(upstream, stream_events(upstream), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            semantic_loss_protocol="hook",
            input_semantic_loss_hook=hook,
            enable_message_cache=True,
        ) as llm:
            await llm.generate(
                messages=[
                    {"role": "system", "content": "initial-instruction", "name": "n"},
                    {"role": "user", "content": "first-input"},
                ]
            )
            await llm.generate(messages=[{"role": "user", "content": "next-input"}])
        first, second = [json.loads(item.content) for item in sdk.requests]
        assert system_texts(upstream, first) == ["hook-instruction"]
        assert system_texts(upstream, second) == ["initial-instruction"]


@pytest.mark.parametrize("policy", ["raise", "hook"])
async def test_translation_is_cached_before_raising_adaptor_or_hook(policy):
    def hook(draft, losses, context):
        draft["messages"].append({"role": "user", "content": "discarded-hook-edit"})
        raise RuntimeError("hook failed")

    provider, model = profiles("anthropic")
    with LocalSDK("anthropic", wire_frames("anthropic", stream_events("anthropic"), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            semantic_loss_protocol=policy,
            input_semantic_loss_hook=hook,
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=[{"role": "user", "content": "first-input"}])
            error = SemanticLossError if policy == "raise" else RuntimeError
            with pytest.raises(error):
                await llm.generate(
                    messages=[{"role": "user", "content": "translated", "name": "n"}]
                )
            await llm.generate(messages=[{"role": "user", "content": "next-input"}])
        assert len(sdk.requests) == 2
        latest = json.loads(sdk.requests[-1].content)["messages"]
        assert [part["content"][0]["text"] for part in latest] == [
            "first-input",
            "translated",
            "next-input",
        ]
        assert "discarded-hook-edit" not in json.dumps(latest)


async def test_prepared_stream_owns_snapshot_after_caller_mutation():
    provider, model = profiles("openai_chat")
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 61)
    ) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_chat",
            enable_message_cache=True,
        ) as llm:
            first_input = [{"role": "user", "content": "first-input"}]
            first = await llm.generate(messages=first_input, stream=True)
            first_input[0]["content"] = "edited-by-caller"
            assert not sdk.requests
            await consume(first)
            second = await llm.generate(
                messages=[{"role": "user", "content": "next-input"}], stream=True
            )
            await consume(second)
        assert [
            [message["content"] for message in json.loads(item.content)["messages"]]
            for item in sdk.requests
        ] == [["first-input"], ["first-input", "next-input"]]


async def test_instances_keep_distinct_cache_keys_and_histories():
    provider, model = profiles("openai_chat")
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 61)
    ) as sdk:
        async with (
            MyLLM(
                provider=provider,
                model=model,
                stdio_protocol="openai_chat",
                enable_message_cache=True,
            ) as first,
            MyLLM(
                provider=provider,
                model=model,
                stdio_protocol="openai_chat",
                enable_message_cache=True,
            ) as second,
        ):
            assert first._message_cache_key != second._message_cache_key
            await first.generate(messages=[{"role": "user", "content": "first-instance"}])
            await second.generate(messages=[{"role": "user", "content": "second-instance"}])
            await first.generate(messages=[{"role": "user", "content": "first-again"}])
        histories = [json.loads(item.content)["messages"] for item in sdk.requests]
        assert [[message["content"] for message in values] for values in histories] == [
            ["first-instance"],
            ["second-instance"],
            ["first-instance", "first-again"],
        ]


@pytest.mark.parametrize("content_kind", ["text", "reasoning", "both"])
async def test_responses_generated_item_ids_remain_unique_across_assistant_deltas(content_kind):
    def assistant(value):
        content = []
        if content_kind in ("text", "both"):
            content.append(TextContent(value))
        if content_kind in ("reasoning", "both"):
            content.append(ReasoningContent(value))
        return AssistantMessage("assistant", content)

    protocol = "openai_responses"
    provider, model = profiles(protocol)
    with LocalSDK(protocol, wire_frames(protocol, stream_events(protocol), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="middleware",
            enable_message_cache=True,
        ) as llm:
            await llm.generate(messages=[assistant("first-assistant")])
            initial = json.loads(sdk.requests[-1].content)["input"]
            await llm.generate(messages=[assistant("second-assistant")])
            latest = json.loads(sdk.requests[-1].content)["input"]
            assert latest[: len(initial)] == initial
            item_ids = [item["id"] for item in latest]
            assert len(set(item_ids)) == len(item_ids)
            await llm.generate(
                messages=[
                    AssistantMessage(
                        "assistant",
                        [TextContent("explicit", item=ItemIdentity(id="msg_external"))],
                    )
                ]
            )
            assert json.loads(sdk.requests[-1].content)["input"][-1]["id"] == "msg_external"


@pytest.mark.parametrize("source", ["ollama", "gemini", "legacy_chat"])
async def test_idless_tool_deltas_keep_distinct_paired_ids_without_old_validation(
    monkeypatch, source
):
    def tool_round(value, *, explicit=False):
        if source == "gemini":
            call = {"name": "weather", "args": {}}
            result = {"name": "weather", "response": {"result": value}}
            if explicit:
                call["id"] = result["id"] = "call_external"
            return [
                {"role": "model", "parts": [{"function_call": call}]},
                {"role": "user", "parts": [{"function_response": result}]},
            ]
        if source == "ollama":
            call = {"function": {"name": "weather", "arguments": {}}}
            result = {"role": "tool", "tool_name": "weather", "content": value}
            if explicit:
                call["id"] = result["tool_call_id"] = "call_external"
            return [{"role": "assistant", "tool_calls": [call]}, result]
        if explicit:
            return [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_external",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_external", "content": value},
            ]
        return [
            {"role": "assistant", "function_call": {"name": "weather", "arguments": "{}"}},
            {"role": "function", "name": "weather", "content": value},
        ]

    validated_lengths = []
    original = decoder_module.validate_messages

    def validate(messages, **kwargs):
        if kwargs.get("history") is not None:
            validated_lengths.append(len(messages))
        return original(messages, **kwargs)

    monkeypatch.setattr(decoder_module, "validate_messages", validate)
    protocol = "openai_chat"
    provider, model = profiles(protocol)
    with LocalSDK(protocol, wire_frames(protocol, stream_events(protocol), 61)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol=protocol if source == "legacy_chat" else source,
            enable_message_cache=True,
        ) as llm:
            first_input = tool_round("first-result")
            before = deepcopy(first_input)
            await llm.generate(messages=first_input)
            initial = json.loads(sdk.requests[-1].content)["messages"]
            await llm.generate(messages=tool_round("second-result"))
            latest = json.loads(sdk.requests[-1].content)["messages"]
            calls = [call["id"] for message in latest for call in message.get("tool_calls", [])]
            results = [message["tool_call_id"] for message in latest if message["role"] == "tool"]
            assert len(calls) == len(set(calls)) == 2
            assert results == calls
            assert latest[: len(initial)] == initial
            assert first_input == before
            await llm.generate(messages=tool_round("explicit-result", explicit=True))
            explicit = json.loads(sdk.requests[-1].content)["messages"]
            assert explicit[:-2] == latest
            assert explicit[-2]["tool_calls"][0]["id"] == "call_external"
            assert explicit[-1]["tool_call_id"] == "call_external"
        assert validated_lengths == [2, 2, 2]
