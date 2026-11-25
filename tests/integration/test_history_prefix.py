"""Measure exact upstream history prefixes with full and cached incremental input."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from my_llm import MessageEvent, MyCodec, MyLLM, ReasoningContent, TextContent, ToolCall
from my_llm.codec.openai.chat.prebuilt import (
    chat_semantic_loss_hook,
    move_tool_images_to_user_message,
    reasoning_content_feedback,
)
from tests.helpers.chat_history import assert_chat_tool_sequence
from tests.helpers.generation_history import (
    INPUT_PROTOCOLS,
    MODE_IDS,
    MODES,
    SCENARIOS,
    calls_for,
    compensation_history,
    declarations,
    initial_history,
    reasoned_events,
    tool_results,
    upstream_events,
    user,
)
from tests.helpers.profiles import config
from tests.helpers.protocols import PROTOCOLS
from tests.helpers.schema import validate_request
from tests.helpers.tool_images import image_references
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import profiles


async def generate_delta(llm, messages, tools, streaming):
    """Replay result after exhaustion and compare it with the finish projection."""
    response = await llm.generate(messages=messages, tools=tools, stream=streaming)
    if not streaming:
        native = response if isinstance(response, list) else [response]
        completed = (
            native
            if llm.stdio_protocol == "middleware"
            else MyCodec().encode(native, input_format=llm.stdio_protocol).messages
        )
        return native, completed
    async with response:
        events = [event async for event in response]
    public = (
        events
        if llm.stdio_protocol == "middleware"
        else MyCodec().encode(chunks=events, input_format=llm.stdio_protocol).stream
    )
    finishes = [event for event in public if event.type == "finish"]
    assert len(finishes) == 1 and public[-1] is finishes[0]
    finish = finishes[0]
    assert isinstance(finish, MessageEvent) and finish.messages
    # A fresh public converter supplies the input format for native I/O; the
    # caller directly appends finish.messages for middleware I/O.
    delta = (
        finish.messages
        if llm.stdio_protocol == "middleware"
        else MyCodec().decode(finish.messages, target_format=llm.stdio_protocol)["messages"]
    )
    result = response.result
    assert result is not None
    completed = result if isinstance(result, list) else [result]
    if llm.stdio_protocol == "middleware":
        assert [message.to_wire() for message in completed] == [
            message.to_wire() for message in finish.messages
        ]
    else:
        assert completed == delta
    return completed, finish.messages


def request_history(protocol, request):
    return request["input" if protocol == "openai_responses" else "messages"]


def assert_completion(messages, scenario, round_index):
    """Independent output checks prevent an empty/lost response passing a prefix test."""
    blocks = [block for message in messages for block in message.content]
    actual = [block for block in blocks if isinstance(block, ToolCall)]
    expected = calls_for(scenario, round_index)
    assert [(call.call_id, call.name, call.arguments) for call in actual] == [
        (call["id"], call["name"], call["input"]) for call in expected
    ]
    text = "".join(block.text for block in blocks if isinstance(block, TextContent))
    assert text == (
        "" if scenario == "tool-only" and expected else f"answer-{round_index} / 上海 / 😀"
    )


def assert_tool_history(protocol, messages, expected_ids):
    """Check tool associations independently of the production history repair."""
    if protocol == "openai_chat":
        calls = [call["id"] for message in messages for call in message.get("tool_calls", [])]
        results = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    elif protocol == "openai_responses":
        calls = [
            message["call_id"] for message in messages if message.get("type") == "function_call"
        ]
        results = [
            message["call_id"]
            for message in messages
            if message.get("type") == "function_call_output"
        ]
    else:
        blocks = [block for message in messages for block in message["content"]]
        calls = [block["id"] for block in blocks if block["type"] == "tool_use"]
        results = [block["tool_use_id"] for block in blocks if block["type"] == "tool_result"]
    assert calls == expected_ids
    assert sorted(results) == sorted(expected_ids)


def assert_prefix_behavior(upstream, current, previous, unstable):
    old = request_history(upstream, previous)
    new = request_history(upstream, current)
    assert current.get("system") == previous.get("system")
    if unstable:
        # Image hashes stay fixed, but the hook relocates the image message to
        # the end of each complete request. Record that remaining raw mismatch.
        assert new[: len(old)] != old
        old_refs = {ref for message in old for ref in image_references(message)}
        new_refs = {ref for message in new for ref in image_references(message)}
        assert len(old_refs) == len(new_refs) == 2
        assert old_refs == new_refs
        assert new[-1] == old[-1]
        assert new[: len(old) - 1] == old[:-1]
    else:
        assert new[: len(old)] == old


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("stdio", INPUT_PROTOCOLS)
@pytest.mark.parametrize("modes", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
async def test_generate_history_prefix_behavior(upstream, stdio, modes, scenario, cached):
    provider, model = profiles(upstream)
    if scenario == "reasoning-not-replayed":
        model = replace(model, should_feedback_reasoning_item=config("none"))
    messages = initial_history(stdio, scenario)
    compensate = scenario == "tool-images" and upstream == "openai_chat" and stdio != "openai_chat"
    unstable = compensate and not cached
    tools = (
        declarations(stdio)
        if scenario
        in (
            "history",
            "repair-history",
            "single-tool",
            "parallel-tools",
            "tool-only",
            "empty-results",
            "tool-images",
        )
        else None
    )
    observations = []
    saved_inputs = []
    expected_ids = []
    if scenario in ("history", "repair-history"):
        expected_ids = ["call_1"]
    elif scenario == "tool-images":
        expected_ids = [{"openai_chat": "call_1", "anthropic": "call_a"}.get(stdio, "c")]

    def compensate_images(draft, losses, context):
        assert (context.message_cache is not None) is cached
        assert {loss.reason for loss in losses} == {"Chat tool results cannot represent images"}
        chat_semantic_loss_hook(draft, losses, context)

    with LocalSDK(upstream, ()) as sdk:
        llm = MyLLM(
            provider=provider,
            model=model,
            stdio_protocol=stdio,
            enable_message_cache=cached,
            semantic_loss_protocol="hook" if compensate else "raise",
            input_semantic_loss_hook=compensate_images if compensate else None,
        )
        assert llm.enable_message_cache is cached
        for round_index, streaming in enumerate(modes):
            sdk.frames = wire_frames(upstream, upstream_events(upstream, scenario, round_index), 17)
            before = deepcopy((messages, tools))
            delta, completed = await generate_delta(llm, messages, tools, streaming)
            assert_completion(completed, scenario, round_index)
            assert (messages, tools) == before
            saved_inputs.append((messages, deepcopy(messages)))
            request = json.loads(sdk.requests[-1].content)
            validate_request(upstream, request)
            current = request_history(upstream, request)
            assert current
            if upstream == "openai_chat":
                assert_chat_tool_sequence(current)
            assert_tool_history(upstream, current, expected_ids)
            rendered = json.dumps(request, ensure_ascii=False)
            assert all(
                rendered.count(f"answer-{index} / 上海 / 😀") == 1
                for index in range(round_index)
                if scenario != "tool-only" or not calls_for(scenario, index)
            )
            if scenario != "empty-results":
                assert all(
                    rendered.count(f"result-{call['id']}") == 1
                    for index in range(round_index)
                    for call in calls_for(scenario, index)
                )
            for previous in observations:
                previous_history = request_history(upstream, previous)
                assert len(current) > len(previous_history)
                assert_prefix_behavior(upstream, request, previous, unstable)
            observations.append(request)
            if scenario == "reasoning-not-replayed":
                assert "old-plan" not in rendered
                assert "old-answer" in rendered
            # The same full history must serialize identically on a retry,
            # including a switch between the two public return modes.
            if round_index == 1:
                retry_delta, retry_completed = await generate_delta(
                    llm, messages, tools, not streaming
                )
                assert_completion(retry_completed, scenario, round_index)
                retry = json.loads(sdk.requests[-1].content)
                validate_request(upstream, retry)
                assert retry_delta
                assert retry == request
                assert (messages, tools) == before
            assert (llm._message_cache_key in llm._codec._message_cache) is cached
            calls = calls_for(scenario, round_index)
            expected_ids.extend(call["id"] for call in calls)
            tail = (
                tool_results(stdio, calls, empty=scenario == "empty-results")
                if calls
                else [user(stdio, f"follow-up-{round_index}")]
            )
            messages = [*delta, *tail] if cached else [*messages, *delta, *tail]
        assert len(sdk.requests) == 5
        assert all(original == snapshot for original, snapshot in saved_inputs)
        assert all(body.closed for body in sdk.bodies)
        assert all(client.is_closed for client in sdk.http_clients)


@pytest.mark.parametrize("stdio", ["middleware", "openai_responses", "anthropic"])
@pytest.mark.parametrize("modes", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize(
    "hook,scenario",
    [
        pytest.param(reasoning_content_feedback, "reasoning", id="reasoning-hook"),
        pytest.param(move_tool_images_to_user_message, "images", id="image-hook"),
        pytest.param(chat_semantic_loss_hook, "reasoning", id="combined-hook-reasoning"),
        pytest.param(chat_semantic_loss_hook, "images", id="combined-hook-images"),
        pytest.param(chat_semantic_loss_hook, "both", id="combined-hook-both"),
    ],
)
async def test_compensation_from_first_request_to_second_round(
    stdio, modes, hook, scenario, cached
):
    provider, model = profiles("openai_chat")
    model = replace(model, should_feedback_reasoning_item=config("always"))
    messages = compensation_history(stdio, scenario)
    has_images = scenario in ("images", "both")
    has_reasoning = scenario in ("reasoning", "both")
    expected_losses = set()
    if has_images:
        expected_losses.add("Chat tool results cannot represent images")
    if has_reasoning:
        expected_losses.add(
            "target protocol cannot preserve another protocol's reasoning feedback_item"
        )
    compensated = []
    with LocalSDK("openai_chat", ()) as sdk:

        def compensate(draft, losses, context):
            # The first baseline must already contain the hook's compensation.
            # The second request adds only the first response and a new user turn.
            assert len(sdk.requests) == len(compensated)
            assert (context.message_cache is not None) is cached
            assert context.origin_source == "input" and context.stage == "request"
            assert {loss.reason for loss in losses} == expected_losses
            hook(draft, losses, context)
            if has_reasoning:
                assert [
                    message["reasoning_content"]
                    for message in draft["messages"]
                    if message.get("reasoning_content")
                ] == ["old-plan"]
            if has_images:
                assert len(image_references(draft["messages"][-1])) == 2
            compensated.append(deepcopy(draft))

        llm = MyLLM(
            provider=provider,
            model=model,
            stdio_protocol=stdio,
            enable_message_cache=cached,
            semantic_loss_protocol="hook",
            input_semantic_loss_hook=compensate,
        )
        for round_index, streaming in enumerate(modes[:2]):
            sdk.frames = wire_frames(
                "openai_chat", upstream_events("openai_chat", "text", round_index), 17
            )
            before = deepcopy(messages)
            delta, completed = await generate_delta(llm, messages, None, streaming)
            assert_completion(completed, "text", round_index)
            assert messages == before
            assert len(compensated) == (1 if cached else round_index + 1)
            assert (llm._message_cache_key in llm._codec._message_cache) is cached
            request = json.loads(sdk.requests[-1].content)
            validate_request("openai_chat", request)
            assert_chat_tool_sequence(request["messages"])
            if not cached or round_index == 0:
                assert request["messages"] == compensated[round_index]["messages"]
            if round_index == 0:
                first = request
                tail = [*delta, user(stdio, "second-input")]
                messages = tail if cached else [*messages, *tail]
            else:
                assert len(request["messages"]) > len(first["messages"])
                assert_prefix_behavior(
                    "openai_chat", request, first, unstable=has_images and not cached
                )
                assert json.dumps(request, ensure_ascii=False).count("answer-0 / 上海 / 😀") == 1
        assert len(sdk.requests) == 2
        assert len(compensated) == (1 if cached else 2)
        assert all(body.closed for body in sdk.bodies)
        assert all(client.is_closed for client in sdk.http_clients)


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("middleware", [False, True], ids=["native", "middleware"])
@pytest.mark.parametrize("feedback", ["none", "always", "when_tool_calls"])
@pytest.mark.parametrize("modes", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
async def test_reasoning_feedback_uses_completed_messages(
    upstream, middleware, feedback, modes, cached
):
    provider, model = profiles(upstream)
    stdio = "middleware" if middleware else upstream
    model = replace(model, should_feedback_reasoning_item=config(feedback))
    messages = [user(stdio, "first-input")]
    tools = declarations(stdio)
    observations = []
    with LocalSDK(upstream, ()) as sdk:
        llm = MyLLM(
            provider=provider, model=model, stdio_protocol=stdio, enable_message_cache=cached
        )
        for round_index, streaming in enumerate(modes):
            sdk.frames = wire_frames(upstream, reasoned_events(upstream, round_index), 17)
            before = deepcopy(messages)
            delta, completed = await generate_delta(llm, messages, tools, streaming)
            assert messages == before
            assert_completion(completed, "single-tool", round_index)
            reasons = [
                block
                for message in completed
                for block in message.content
                if isinstance(block, ReasoningContent)
            ]
            assert len(reasons) == 1 and reasons[0].reasoning == f"plan-{round_index}"
            assert reasons[0].feedback_item is not None
            request = json.loads(sdk.requests[-1].content)
            validate_request(upstream, request)
            for previous in observations:
                assert_prefix_behavior(upstream, request, previous, unstable=False)
            for index in range(round_index):
                retained = feedback == "always" or feedback == "when_tool_calls" and index < 2
                assert (f"plan-{index}" in json.dumps(request)) is retained
            observations.append(request)
            calls = calls_for("single-tool", round_index)
            tail = (
                tool_results(stdio, calls) if calls else [user(stdio, f"follow-up-{round_index}")]
            )
            messages = [*delta, *tail] if cached else [*messages, *delta, *tail]
            assert (llm._message_cache_key in llm._codec._message_cache) is cached
        assert len(sdk.requests) == 4
        assert all(body.closed for body in sdk.bodies)
        assert all(client.is_closed for client in sdk.http_clients)
