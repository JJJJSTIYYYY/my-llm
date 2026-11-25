"""Native item identities and target addresses survive public stream projection."""

import json
from dataclasses import replace

import pytest

from my_llm import MyCodec, MyLLM
from tests.helpers.profiles import parameter
from tests.helpers.protocols import response_completed, stream_events
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import profiles


def message(identity, *texts):
    return {
        "type": "message",
        "id": identity,
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
            for text in texts
        ],
    }


def native_events(output):
    """Include in-progress empty headers before their eventual text parts."""
    events = [stream_events("openai_responses")[0]]
    for index, item in enumerate(output):
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "status": "in_progress", "content": []},
            }
        )
        for part, block in enumerate(item["content"]):
            if block["type"] != "output_text":
                continue
            address = {"output_index": index, "item_id": item["id"], "content_index": part}
            events.extend(
                [
                    {
                        "type": "response.content_part.added",
                        **address,
                        "part": {**block, "text": ""},
                    },
                    {
                        "type": "response.output_text.delta",
                        **address,
                        "delta": block["text"],
                        "logprobs": [],
                    },
                    {"type": "response.content_part.done", **address, "part": block},
                ]
            )
    events.append(response_completed(output))
    return events


def translated(incoming, *, saved=False, codec=None, **parameters):
    codec = codec or MyCodec()
    middleware = codec.encode(chunks=incoming, input_format="openai_responses").stream
    if saved:
        middleware = json.loads(json.dumps([event.to_wire() for event in middleware]))
    return codec.decode(chunks=middleware, target_format="openai_responses", **parameters)["chunks"]


def assert_addresses(output):
    completed = output[-1]["response"]["output"]
    done = [event["item"] for event in output if event["type"] == "response.output_item.done"]
    assert done == completed
    for event in output:
        if "output_index" not in event:
            continue
        item = completed[event["output_index"]]
        assert event.get("item_id", event.get("item", {}).get("id")) == item["id"]
        if "content_index" in event:
            assert event["content_index"] < len(item["content"])
    MyCodec().convert(chunks=output, input_format="openai_responses", target_format="middleware")
    return completed


@pytest.mark.parametrize("saved", [False, True], ids=["typed", "json"])
def test_native_parts_and_true_empty_items_keep_their_own_identity(saved):
    incoming = native_events(
        [
            message("empty_before"),
            message("parts", "one", "two"),
            message("empty_after"),
            message("last", "three"),
        ]
    )
    completed = assert_addresses(translated(incoming, saved=saved))
    assert {item["id"]: [part["text"] for part in item["content"]] for item in completed} == {
        "empty_before": [],
        "parts": ["one", "two"],
        "empty_after": [],
        "last": ["three"],
    }


@pytest.mark.parametrize("saved", [False, True], ids=["typed", "json"])
def test_dropping_unsupported_native_parts_keeps_supported_text_source_addresses(saved):
    item = message("parts", "one", "two")
    item["content"].insert(0, {"type": "refusal", "refusal": "unsupported"})
    item["content"].insert(2, {"type": "refusal", "refusal": "unsupported"})
    completed = assert_addresses(translated(native_events([item]), saved=saved))
    assert completed == [message("parts", "one", "two")]


@pytest.mark.parametrize("saved", [False, True], ids=["typed", "json"])
def test_filtered_native_text_does_not_reappear_as_an_empty_item(profile_factory, saved):
    profile = profile_factory(
        "openai_responses",
        content=parameter("content", supports=False, protocol="openai_responses"),
    )
    codec = MyCodec(semantic_loss_protocol="ignore")
    incoming = native_events([message("text", "hidden"), message("actually_empty")])
    completed = assert_addresses(
        translated(incoming, saved=saved, codec=codec, semantic_profile=profile)
    )
    assert completed == [message("actually_empty")]


@pytest.mark.parametrize(
    "summary",
    [[], [{"type": "summary_text", "text": ""}], [{"type": "summary_text", "text": "plan"}]],
)
@pytest.mark.parametrize("filtered", [False, True])
def test_native_reasoning_empty_summary_and_filtering_keep_addresses(
    profile_factory, summary, filtered
):
    reasoning = {
        "type": "reasoning",
        "id": "reason",
        "summary": summary,
        "encrypted_content": "opaque",
    }
    profile = profile_factory(
        "openai_responses",
        reasoning_enable=parameter(
            "reasoning_enable", supports=not filtered, protocol="openai_responses"
        ),
    )
    output = translated(
        [response_completed([reasoning, message("answer", "visible")])],
        codec=MyCodec(semantic_loss_protocol="ignore"),
        semantic_profile=profile,
    )
    completed = assert_addresses(output)
    if filtered:
        assert completed == [message("answer", "visible")]
    else:
        assert completed == [reasoning, message("answer", "visible")]


@pytest.mark.parametrize("visible", [False, True], ids=["opaque", "empty-visible"])
def test_foreign_empty_reasoning_never_becomes_a_message_or_leaves_index_holes(visible):
    incoming = [
        stream_events("anthropic")[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": "signed"}
            if visible
            else {"type": "redacted_thinking", "data": "opaque"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": "visible"},
        },
        {"type": "content_block_stop", "index": 1},
        *stream_events("anthropic")[-2:],
    ]
    output = MyCodec(semantic_loss_protocol="ignore").convert(
        chunks=incoming, input_format="anthropic", target_format="openai_responses"
    )["chunks"]
    completed = assert_addresses(output)
    assert completed[-1]["content"][0]["text"] == "visible"
    assert len(completed) == 2 if visible else len(completed) == 1
    if visible:
        assert completed[0]["type"] == "reasoning"
        assert completed[0]["summary"] == []


@pytest.mark.parametrize("filtered", [False, True])
async def test_generate_native_items_use_the_same_addresses_until_completion(filtered):
    provider, model = profiles("openai_responses")
    if filtered:
        model = replace(
            model, content=parameter("content", supports=False, protocol="openai_responses")
        )
    incoming = native_events([message("empty_before"), message("parts", "one", "two")])
    with LocalSDK("openai_responses", wire_frames("openai_responses", incoming, 37)):
        llm = MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_responses",
            semantic_loss_protocol="ignore",
        )
        stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
        async with stream:
            output = [event async for event in stream]
        collected = await llm.generate(messages=[{"role": "user", "content": "hi"}])
    completed = assert_addresses(output)
    assert stream.result == completed
    assert collected == completed
    assert {item["id"] for item in completed} == (
        {"empty_before"} if filtered else {"empty_before", "parts"}
    )


async def test_result_keeps_empty_native_items_not_carried_by_finish_messages():
    provider, model = profiles("openai_responses")
    incoming = native_events(
        [message("empty_before"), message("parts", "one", "two"), message("empty_after")]
    )
    with LocalSDK("openai_responses", wire_frames("openai_responses", incoming, 17)):
        llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_responses")
        stream = await llm.generate(messages=[{"role": "user", "content": "hi"}], stream=True)
        async with stream:
            events = [event async for event in stream]
        result = stream.result
    assert result == events[-1]["response"]["output"]
    assert {item["id"] for item in result} == {"empty_before", "parts", "empty_after"}
    finish = MyCodec().encode(chunks=events, input_format="openai_responses").stream[-1]
    replay = MyCodec().decode(finish.messages, target_format="openai_responses")["messages"]
    assert replay != result
    assert [item["id"] for item in replay] == ["parts"]
    assert [part["text"] for part in replay[0]["content"]] == ["one", "two"]
