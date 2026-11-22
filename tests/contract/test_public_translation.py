"""Public translation and generation preserve supported stream semantics."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from my_llm import (
    IncompleteStreamError,
    MessageEvent,
    MessageValidationError,
    MyCodec,
    MyLLM,
    SemanticLossError,
    TextContent,
    UserMessage,
)
from tests.helpers.profiles import parameter
from tests.helpers.protocols import (
    PROTOCOLS,
    chat_event,
    history,
    response_completed,
    stream_events,
)
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import profiles


def public_chunks(chunks, saved):
    """Exercise both typed delivery and an actual persisted JSON event list."""
    return json.loads(json.dumps([event.to_wire() for event in chunks])) if saved else chunks


def reasoning_events(protocol, *, opaque=False):
    """Native fixtures independent of middleware stream construction."""
    if protocol == "openai_chat":
        return [
            chat_event(reasoning="plan"),
            chat_event(content="answer"),
            chat_event(finish="stop"),
        ]
    if protocol == "openai_responses":
        return [
            response_completed(
                [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [] if opaque else [{"type": "summary_text", "text": "plan"}],
                        "encrypted_content": "opaque",
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "answer",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                ]
            )
        ]
    return [
        stream_events("anthropic")[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque"}
            if opaque
            else {"type": "thinking", "thinking": "", "signature": ""},
        },
        *(
            []
            if opaque
            else [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "plan"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "signed"},
                },
            ]
        ),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "answer"},
        },
        {"type": "content_block_stop", "index": 1},
        *stream_events("anthropic")[-2:],
    ]


def assert_reasoning_output(protocol, events, *, opaque=False):
    """Check native fields directly so a matching encoder bug cannot hide loss."""
    if protocol == "openai_chat":
        deltas = [event["choices"][0]["delta"] for event in events]
        assert "".join(delta.get("reasoning_content", "") for delta in deltas) == "plan"
        assert "".join(delta.get("content", "") for delta in deltas) == "answer"
    elif protocol == "openai_responses":
        output = events[-1]["response"]["output"]
        assert output[0]["type"] == "reasoning"
        assert output[0]["id"] == "rs_1"
        assert output[0]["encrypted_content"] == "opaque"
        assert output[0]["summary"] == (
            [] if opaque else [{"type": "summary_text", "text": "plan"}]
        )
        assert output[1]["content"][0]["text"] == "answer"
    else:
        starts = [
            event["content_block"] for event in events if event["type"] == "content_block_start"
        ]
        if opaque:
            assert starts[0] == {"type": "redacted_thinking", "data": "opaque"}
        else:
            deltas = [event["delta"] for event in events if event["type"] == "content_block_delta"]
            assert "".join(delta.get("thinking", "") for delta in deltas) == "plan"
            assert "".join(delta.get("signature", "") for delta in deltas) == "signed"
        assert starts[-1]["type"] == "text"


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("tool", [False, True], ids=["text", "tool"])
@pytest.mark.parametrize("saved", [False, True], ids=["typed", "json"])
def test_public_stream_translation_preserves_text_and_complete_tools(source, target, tool, saved):
    codec = MyCodec()
    incoming = stream_events(source, tool=tool)
    before = deepcopy(incoming)
    middleware = codec.convert(chunks=incoming, input_format=source, target_format="middleware")[
        "chunks"
    ]
    output = codec.decode(chunks=public_chunks(middleware, saved), target_format=target)["chunks"]
    completed = codec.convert(chunks=output, input_format=target, target_format="middleware")[
        "chunks"
    ][-1].messages
    rendered = codec.decode(completed, target_format="openai_chat")["messages"]
    expected = {"role": "assistant", "content": "Hello"}
    if tool:
        expected["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "上海"}'},
            }
        ]
    assert rendered == [expected]
    if target == "openai_responses":
        assert output[-1]["response"]["output"] == [
            event["item"] for event in output if event["type"] == "response.output_item.done"
        ]
    assert incoming == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_encode_then_decode_matches_direct_conversion(source, target):
    codec = MyCodec()
    incoming = stream_events(source, tool=True)
    encoded = codec.encode(chunks=incoming, input_format=source)
    assert all(isinstance(event, MessageEvent) for event in encoded.stream)
    assert encoded.stream[-1].type == "finish"
    separately = codec.decode(chunks=encoded.stream, target_format=target)
    together = codec.convert(chunks=incoming, input_format=source, target_format=target)
    assert separately == together


@pytest.mark.parametrize("saved", [False, True], ids=["typed", "json"])
@pytest.mark.parametrize(
    "protocol,opaque",
    [
        ("openai_chat", False),
        ("openai_responses", False),
        ("openai_responses", True),
        ("anthropic", False),
        ("anthropic", True),
    ],
)
def test_same_protocol_public_reasoning_keeps_native_feedback(protocol, saved, opaque):
    codec = MyCodec()
    incoming = reasoning_events(protocol, opaque=opaque)
    middleware = codec.convert(chunks=incoming, input_format=protocol, target_format="middleware")[
        "chunks"
    ]
    output = codec.decode(chunks=public_chunks(middleware, saved), target_format=protocol)["chunks"]
    assert_reasoning_output(protocol, output, opaque=opaque)


@pytest.mark.parametrize(
    "source,target",
    [
        ("openai_responses", "openai_chat"),
        ("openai_responses", "anthropic"),
        ("anthropic", "openai_chat"),
        ("anthropic", "openai_responses"),
    ],
)
def test_foreign_opaque_reasoning_still_reports_semantic_loss(source, target):
    codec = MyCodec()
    middleware = codec.convert(
        chunks=reasoning_events(source, opaque=True),
        input_format=source,
        target_format="middleware",
    )["chunks"]
    with pytest.raises(SemanticLossError) as error:
        codec.decode(chunks=middleware, target_format=target)
    assert any("feedback" in item.reason for item in error.value.losses)
    assert error.value.context.source_protocol == source
    assert error.value.context.target_protocol == target


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("unknown_only", [False, True], ids=["recognized", "unknown-only"])
def test_public_encode_rejects_incomplete_stream(protocol, unknown_only):
    incoming = stream_events(protocol)
    if unknown_only:
        incoming = [{"type": "vendor.future", "payload": {"unsupported": True}}]
    elif protocol == "openai_chat":
        incoming = [chat_event(content="unfinished")]
    else:
        incoming = incoming[:-2] if protocol == "anthropic" else incoming[:-1]
    with pytest.raises(IncompleteStreamError):
        MyCodec().encode(chunks=incoming, input_format=protocol)


@pytest.mark.parametrize("field", ["reason", "messages", "output_blocks"])
@pytest.mark.parametrize("target", PROTOCOLS)
def test_public_decode_requires_complete_terminal_content(field, target):
    codec = MyCodec()
    events = public_chunks(
        codec.encode(chunks=stream_events("openai_chat"), input_format="openai_chat").stream,
        True,
    )
    events[-1].pop(field)
    with pytest.raises(MessageValidationError):
        codec.decode(chunks=events, target_format=target)


@pytest.mark.parametrize("target", ["openai_responses", "anthropic"])
def test_public_decode_owns_its_output_index_state(target):
    codec = MyCodec()
    events = public_chunks(
        codec.encode(chunks=stream_events("openai_chat"), input_format="openai_chat").stream,
        True,
    )
    events[-1]["metadata"]["stream_indices"] = {}
    before = deepcopy(events)
    first = codec.decode(chunks=events, target_format=target)
    second = codec.decode(chunks=events, target_format=target)
    assert events == before
    assert first == second


def test_public_middleware_projection_owns_source_addresses():
    codec = MyCodec()
    events = public_chunks(
        codec.encode(
            chunks=stream_events("openai_responses"), input_format="openai_responses"
        ).stream,
        True,
    )
    projected = codec.decode(chunks=events, target_format="middleware")["chunks"]
    before = deepcopy([event.to_wire() for event in projected])
    next(event for event in events if event.get("source_ref"))["source_ref"][1] = 42
    assert [event.to_wire() for event in projected] == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_explicit_source_format_supplies_public_reasoning_context(protocol):
    codec = MyCodec()
    events = public_chunks(
        codec.encode(chunks=reasoning_events(protocol), input_format=protocol).stream,
        True,
    )
    for event in events:
        event.pop("protocol", None)
    output = codec.decode(chunks=events, input_format=protocol, target_format=protocol)["chunks"]
    assert_reasoning_output(protocol, output)


@pytest.mark.parametrize("public", [False, True], ids=["direct", "public-events"])
def test_responses_filtered_output_indices_match_final_snapshot(profile_factory, public):
    codec = MyCodec(semantic_loss_protocol="ignore")
    profile = profile_factory(reasoning_enable=parameter("reasoning_enable", supports=False))
    incoming = [
        chat_event(reasoning="hidden"),
        chat_event(content="Hello"),
        chat_event(finish="stop"),
    ]
    if public:
        middleware = codec.convert(
            chunks=incoming,
            input_format="openai_chat",
            target_format="middleware",
            semantic_profile=profile,
        )["chunks"]
        output = codec.decode(chunks=middleware, target_format="openai_responses")["chunks"]
    else:
        output = codec.convert(
            chunks=incoming,
            input_format="openai_chat",
            target_format="openai_responses",
            semantic_profile=profile,
        )["chunks"]
    assert {event["output_index"] for event in output if "output_index" in event} == {0}
    assert output[-1]["response"]["output"][0]["content"][0]["text"] == "Hello"
    replayed = codec.convert(
        chunks=output, input_format="openai_responses", target_format="middleware"
    )["chunks"]
    assert replayed[-1].messages[0].content[0].text == "Hello"


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_generate_middleware_events_can_be_translated_independently(protocol):
    provider, model = profiles(protocol)
    incoming = reasoning_events(protocol)
    with LocalSDK(protocol, wire_frames(protocol, incoming, 37)) as sdk:
        llm = MyLLM(provider=provider, model=model, stdio_protocol="middleware")
        stream = await llm.generate(
            messages=[UserMessage("user", [TextContent("hi")])], stream=True
        )
        async with stream:
            middleware = [event async for event in stream]
        assert stream.result is not None
        assert all(body.closed for body in sdk.bodies)
    translated = MyCodec().decode(chunks=public_chunks(middleware, True), target_format=protocol)[
        "chunks"
    ]
    assert_reasoning_output(protocol, translated)


async def test_generate_filtered_responses_stream_keeps_output_addresses_consistent():
    provider, model = profiles("openai_chat")
    model = replace(model, reasoning_enable=parameter("reasoning_enable", supports=False))
    incoming = [
        chat_event(reasoning="hidden"),
        chat_event(content="Hello"),
        chat_event(finish="stop"),
    ]
    with LocalSDK("openai_chat", wire_frames("openai_chat", incoming, 37)):
        llm = MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_responses",
            semantic_loss_protocol="ignore",
        )
        stream = await llm.generate(messages=history("openai_responses")[:1], stream=True)
        async with stream:
            output = [event async for event in stream]
        assert stream.result[0]["content"][0]["text"] == "Hello"
    assert {event["output_index"] for event in output if "output_index" in event} == {0}
    replayed = MyCodec().convert(
        chunks=output, input_format="openai_responses", target_format="middleware"
    )["chunks"]
    assert replayed[-1].messages[0].content[0].text == "Hello"
