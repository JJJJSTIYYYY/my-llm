"""Public typed I/O uses existing native components and stream ownership."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest

import my_llm
from my_llm import (
    AssistantMessage,
    ConfigurationError,
    FunctionTool,
    ImageContent,
    ImageSource,
    IncompleteStreamError,
    MessageEvent,
    MessageValidationError,
    MyCodec,
    MyLLM,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    ToolCall,
    ToolResult,
    UserMessage,
    generate,
    omit,
)
from tests.helpers.protocols import PROTOCOLS, chat_event
from tests.performance.transport import Body, LocalSDK, wire_frames
from tests.performance.validation import validate_exchange
from tests.performance.workloads import Size, build, controls, profiles

SMALL = Size(
    rounds=2,
    tools=2,
    parallel_calls=2,
    text_chars=32,
    output_chars=96,
    fragment_chars=17,
    argument_items=1,
    schema_depth=1,
)


def fail_component(*args, **kwargs):
    pytest.fail("direct middleware I/O called a skipped encoder or decoder")


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_typed_generation_skips_input_encoder_and_output_decoder(
    monkeypatch, protocol, streaming
):
    workload = build(protocol, protocol, SMALL, native=True)
    middleware = MyCodec().encode(workload.messages, workload.tools, input_format=protocol)
    before = deepcopy(middleware.to_wire())
    provider, model = profiles(protocol)
    with LocalSDK(protocol, wire_frames(protocol, workload.events, 37)) as sdk:
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            native = llm.codec.codecs[protocol]
            monkeypatch.setattr(type(native.encoder), "request", fail_component)
            monkeypatch.setattr(type(native.decoder), "messages", fail_component)
            monkeypatch.setattr(type(native.decoder), "chunks", fail_component)
            monkeypatch.setattr(type(native.decoder), "result", fail_component)
            result = await llm.generate(
                messages=middleware.messages,
                tools=middleware.tools,
                stream=streaming,
                **controls(protocol),
            )
            if streaming:
                async with result:
                    events = []
                    async for event in result:
                        assert isinstance(event, MessageEvent)
                        assert event.field != "arguments"
                        assert "usage" not in event.to_wire()
                        if event.type == "tool":
                            assert sdk.bodies[-1].read == len(sdk.frames)
                            assert isinstance(event.block, ToolCall)
                            assert isinstance(event.block.arguments, dict)
                        events.append(event)
                assert events[-1].type == "finish"
                assert events[-1].messages[0].role == "assistant"
                assert (
                    len([event for event in events if event.type == "tool"]) == SMALL.parallel_calls
                )
                assert result.result is result.result
                result = result.result
                assert llm.get_last_usage() is not None
            else:
                assert llm.get_last_usage() is None
            assert isinstance(result, AssistantMessage)
            # Native oracle is kept outside the bypass assertions.
            monkeypatch.undo()
            rendered = MyCodec().result(
                [result],
                target_format=protocol,
                input_format=protocol,
            )
            validate_exchange(workload, protocol, rendered, sdk)
            assert set(llm.codec.codecs.instances) == {protocol}
            assert "middleware" not in llm.codec.codecs
    assert middleware.to_wire() == before
    assert all(body.closed for body in sdk.bodies)


@pytest.mark.parametrize("stdio", [None, "anthropic", "openai_chat"])
async def test_typed_input_never_infers_or_uses_a_native_stdio(stdio):
    provider, model = profiles("openai_chat")
    async with MyLLM(provider=provider, model=model, stdio_protocol=stdio) as llm:
        with pytest.raises(MessageValidationError, match="stdio_protocol"):
            await llm.generate(messages=[UserMessage("user", [TextContent("hi")])])
        assert llm.stdio_protocol == stdio
        assert not llm._clients


@pytest.mark.parametrize(
    "messages,tools",
    [
        ([], omit),
        ([{"role": "user", "content": "hi"}], omit),
        ([UserMessage("tool", [TextContent("hi")])], omit),
        ([UserMessage("user", [ToolCall("a", "f")])], omit),
        ([UserMessage("user", [TextContent(42)])], omit),
        ([UserMessage("user", [ImageContent(ImageSource("url"))])], omit),
        (
            [
                AssistantMessage(
                    "assistant", [ImageContent(ImageSource("url", url="https://example.com"))]
                )
            ],
            omit,
        ),
        ([UserMessage("user", [ToolResult("a", [TextContent(42)])])], omit),
        ([AssistantMessage("assistant", [ReasoningContent(mark="invalid")])], omit),
        ([UserMessage("user", [TextContent("hi")])], [{"name": "f"}]),
        ([UserMessage("user", [TextContent("hi")])], [FunctionTool("f", strict="yes")]),
    ],
)
async def test_invalid_public_models_fail_before_sdk_creation(messages, tools):
    provider, model = profiles("openai_chat")
    async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
        with pytest.raises(MessageValidationError):
            await llm.generate(messages=messages, tools=tools)
        assert not llm._clients


@pytest.mark.parametrize("tools,expected", [(omit, None), (None, None), ([], [])])
async def test_tools_omission_is_preserved(tools, expected):
    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    provider, model = profiles("openai_chat")
    with LocalSDK("openai_chat", wire_frames("openai_chat", workload.events, 97)) as sdk:
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            await llm.generate(messages=(UserMessage("user", [TextContent("hi")]),), tools=tools)
        request = json.loads(sdk.requests[-1].content)
        assert request.get("tools") == expected
        assert ("tools" in request) == (expected == [])


@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
async def test_direct_input_losses_keep_actual_objects_and_complete_request(policy):
    image = ImageContent(ImageSource("url", url="https://example.com/a.png"))
    history = [
        AssistantMessage("assistant", [ToolCall("a", "f")]),
        UserMessage("user", [ToolResult("a", [TextContent("ok"), image])]),
        UserMessage("user", [TextContent("tail")]),
    ]
    observations = []

    def hook(draft, losses, context):
        assert losses.items[0].origin_value is image
        assert losses.items[0].origin_path == "$.messages[1].content[0].content[1]"
        assert context.source_protocol == "middleware"
        assert draft["messages"][-1]["content"] == "tail"
        draft["temperature"] = 0.2
        observations.append(draft)

    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    provider, model = profiles("openai_chat")
    with LocalSDK("openai_chat", wire_frames("openai_chat", workload.events, 97)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="middleware",
            semantic_loss_protocol=policy,
            input_semantic_loss_hook=hook,
        ) as llm:
            if policy == "raise":
                with pytest.raises(SemanticLossError) as error:
                    await llm.generate(messages=history)
                assert error.value.losses.items[0].origin_value is image
                assert error.value.draft["messages"][-1]["content"] == "tail"
                assert not sdk.requests
            else:
                await llm.generate(messages=history)
                assert json.loads(sdk.requests[-1].content).get("temperature") == (
                    0.2 if policy == "hook" else None
                )
    assert image.origin is None
    assert history[1].origin is None


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_early_close_and_failed_eof_do_not_publish_tools(protocol):
    workload = build(protocol, protocol, SMALL, native=True)
    provider, model = profiles(protocol)
    frames = wire_frames(protocol, workload.events, 37)
    with LocalSDK(protocol, frames):
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            stream = await llm.generate(
                messages=[UserMessage("user", [TextContent("hi")])], stream=True
            )
            async with stream:
                assert (await anext(stream)).type == "start"
            assert stream.result is None
            assert llm.get_last_usage() is None
    # Remove every terminal source event, leaving a truncated text/tool response.
    partial = workload.events[:-1]
    if protocol == "anthropic":
        partial = workload.events[:-2]
    elif protocol == "openai_chat":
        partial = [
            event
            for event in partial
            if not any(choice.get("finish_reason") for choice in event.get("choices", []))
        ]
    with LocalSDK(protocol, wire_frames(protocol, partial, 37)):
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            stream = await llm.generate(
                messages=[UserMessage("user", [TextContent("hi")])], stream=True
            )
            events = []
            with pytest.raises(IncompleteStreamError) as error:
                async with stream:
                    async for event in stream:
                        events.append(event)
            assert all(event.type != "tool" for event in events)
            assert isinstance(error.value.partial_message, AssistantMessage)
            assert not any(
                isinstance(part, ToolCall) for part in error.value.partial_message.content
            )
            assert stream.result is None
            assert llm.get_last_usage() is None


async def test_concurrent_calls_isolate_inputs_results_and_events():
    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    provider, model = profiles("openai_chat")
    frames = wire_frames("openai_chat", workload.events, 37)
    with LocalSDK("openai_chat", frames) as sdk:
        barrier = asyncio.Barrier(2)
        respond = sdk.respond

        class ConcurrentBody(Body):
            async def __aiter__(self):
                await barrier.wait()
                async for frame in super().__aiter__():
                    yield frame

        def overlap(request):
            response = respond(request)
            sdk.bodies[-1] = response.stream = ConcurrentBody(frames)
            return response

        sdk.respond = overlap
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            history = [UserMessage("user", [TextContent("before")])]
            streams = await asyncio.gather(
                *[llm.generate(messages=history, stream=True) for _ in range(2)]
            )
            history[0].content[0].text = "after"

            async def consume(stream):
                async with stream:
                    return [event async for event in stream]

            events = await asyncio.wait_for(asyncio.gather(*(consume(s) for s in streams)), 5)
            assert sdk.created_clients == 1
            assert all(
                json.loads(r.content)["messages"][0]["content"] == "before" for r in sdk.requests
            )
            first = next(e.block for e in events[0] if e.type == "tool")
            second = next(e.block for e in events[1] if e.type == "tool")
            saved = deepcopy(streams[0].result.to_wire())
            first.arguments.clear()
            assert second.arguments
            assert streams[0].result.to_wire() == saved
            assert streams[0].result is not streams[1].result
        assert all(body.closed for body in sdk.bodies)


async def test_one_shot_helper_supports_typed_io_and_closes_clients():
    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    provider, model = profiles("openai_chat")
    with LocalSDK("openai_chat", wire_frames("openai_chat", workload.events, 97)) as sdk:
        result = await generate(
            provider=provider,
            model=model,
            stdio_protocol="middleware",
            messages=[UserMessage("user", [TextContent("hi")])],
        )
        assert isinstance(result, AssistantMessage)
        assert all(client.is_closed for client in sdk.http_clients)


def test_all_public_middleware_field_types_are_exported():
    for name in (
        "MessageContent",
        "ResultContent",
        "JsonObject",
        "JsonValue",
        "ItemIdentity",
        "MessageEvent",
    ):
        assert name in my_llm.__all__
        assert hasattr(my_llm, name)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
async def test_output_loss_policy_still_applies_to_typed_drafts(streaming, policy):
    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    provider, model = profiles("openai_chat")
    provider = replace(provider, reasoning_content=omit)
    workload.events.insert(0, chat_event(reasoning="reason"))
    drafts = []

    def hook(draft, losses, context):
        assert context.target_protocol == "middleware"
        assert context.origin_source == losses.origin_source == "output"
        drafts.append(draft)
        if isinstance(draft, AssistantMessage):
            draft.content.append(TextContent("hook tail"))

    with LocalSDK("openai_chat", wire_frames("openai_chat", workload.events, 37)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="middleware",
            semantic_loss_protocol=policy,
            output_semantic_loss_hook=hook,
        ) as llm:

            async def execute():
                result = await llm.generate(
                    messages=[UserMessage("user", [TextContent("hi")])],
                    stream=streaming,
                )
                if streaming:
                    async with result:
                        async for _ in result:
                            pass
                    result = result.result
                return result

            if policy == "raise":
                with pytest.raises(SemanticLossError):
                    await execute()
            else:
                result = await execute()
                assert not any(isinstance(part, ReasoningContent) for part in result.content)
                if policy == "hook":
                    assert any(draft is result for draft in drafts)
                    assert result.content[-1].text == "hook tail"
                else:
                    assert not drafts
        assert all(body.closed for body in sdk.bodies)


async def test_decoder_repairs_history_and_retains_generation_validation():
    provider, model = profiles("openai_chat")
    async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
        stream = await llm.generate(
            messages=[AssistantMessage("assistant", [ToolCall("a", "f")])], stream=True
        )
        await stream.aclose()
        with pytest.raises(MessageValidationError, match="no usable history"):
            await llm.generate(
                messages=[UserMessage("user", [ToolResult("a", [TextContent("x")])])], stream=True
            )
        with pytest.raises(ConfigurationError):
            await llm.generate(messages=[UserMessage("user", [TextContent("x")])], max_tokens=-1)
        assert not llm._clients


def test_middleware_is_reserved_for_stdio_and_never_registers_components():
    from my_llm.codec.openai.chat.codec import ChatCompletionsCodec

    class ReservedCodec(ChatCompletionsCodec):
        protocol_or_api_style = "middleware"

    with pytest.raises(ConfigurationError, match="reserved"):
        MyCodec.register(ReservedCodec)
    assert "middleware" not in MyCodec().codecs


def test_offline_conversion_can_return_typed_models_and_complete_public_events():
    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    converted = MyCodec().convert(
        workload.messages,
        workload.tools,
        chunks=workload.events,
        input_format="openai_chat",
        target_format="middleware",
    )
    assert isinstance(converted["messages"][0], my_llm.Message)
    assert isinstance(converted["tools"][0], FunctionTool)
    assert isinstance(converted["chunks"][-1], MessageEvent)
    assert converted["chunks"][-1].type == "finish"


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_prepared_request_owns_nested_application_data(protocol):
    workload = build(protocol, protocol, SMALL, native=True)
    middleware = MyCodec().encode(workload.messages, workload.tools, input_format=protocol)
    provider, model = profiles(protocol)
    with LocalSDK(protocol, wire_frames(protocol, workload.events, 97)) as sdk:
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            stream = await llm.generate(
                messages=middleware.messages,
                tools=middleware.tools,
                stream=True,
                **controls(protocol),
            )
            for tool in middleware.tools:
                tool.parameters.clear()
            for message in middleware.messages:
                for part in message.content:
                    if isinstance(part, ToolCall):
                        part.arguments.clear()
                    elif isinstance(part, ToolResult):
                        for child in part.content:
                            if isinstance(child, TextContent):
                                child.text = "changed after prepare"
            async with stream:
                async for _ in stream:
                    pass
            validate_exchange(workload, "middleware", stream.result, sdk)


async def test_cancellation_closes_middleware_response_without_publishing_completion():
    workload = build("openai_chat", "openai_chat", SMALL, native=True)
    provider, model = profiles("openai_chat")
    frames = wire_frames("openai_chat", workload.events, 97)
    reading = asyncio.Event()
    blocked = asyncio.Event()

    with LocalSDK("openai_chat", frames) as sdk:
        respond = sdk.respond

        class WaitingBody(Body):
            async def __aiter__(self):
                reading.set()
                await blocked.wait()
                async for frame in super().__aiter__():
                    yield frame

        def wait_response(request):
            response = respond(request)
            sdk.bodies[-1] = response.stream = WaitingBody(frames)
            return response

        sdk.respond = wait_response
        async with MyLLM(provider=provider, model=model, stdio_protocol="middleware") as llm:
            stream = await llm.generate(
                messages=[UserMessage("user", [TextContent("hi")])],
                stream=True,
            )
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(reading.wait(), 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert stream.result is None
            assert llm.get_last_usage() is None
            assert sdk.bodies[-1].closed
