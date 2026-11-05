"""Check pairing and adjacency on real SDK wire requests after image relocation."""

import json
from copy import deepcopy

import pytest

from my_llm import Config
from my_llm.codec.openai.chat.prebuilt import move_tool_images_to_user_message
from tests.helpers.chat_history import assert_chat_tool_sequence
from tests.helpers.protocols import stream_events
from tests.helpers.schema import validate_request
from tests.helpers.tool_images import (
    IMAGE_REFERENCE,
    configured_llm,
    consume,
    image_references,
    without_image_ids,
)
from tests.performance.transport import LocalSDK, wire_frames


@pytest.fixture
def chat_sdk():
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 31)
    ) as sdk:
        yield sdk


def call_messages(source, ids):
    if source == "openai_responses":
        return [
            {"type": "function_call", "call_id": key, "name": "look", "arguments": "{}"}
            for key in ids
        ]
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": key, "name": "look", "input": {}} for key in ids
            ],
        }
    ]


def result_messages(source, ids):
    if source == "openai_responses":
        return [
            {
                "type": "function_call_output",
                "call_id": key,
                "output": [
                    {"type": "input_text", "text": f"result:{key}"},
                    {"type": "input_image", "image_url": f"https://example.test/{key}.png"},
                ],
            }
            for key in ids
        ]
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": key,
                    "content": [
                        {"type": "text", "text": f"result:{key}"},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": f"https://example.test/{key}.png"},
                        },
                    ],
                }
            ],
        }
        for key in ids
    ]


def wire_request(sdk):
    body = json.loads(sdk.requests[-1].content)
    validate_request("openai_chat", body)
    return body


def image_urls(message):
    assert message["role"] == "user"
    content = message["content"]
    if isinstance(content, str):
        return []
    return [part["image_url"]["url"] for part in content if part["type"] == "image_url"]


@pytest.mark.parametrize("source", ["anthropic", "openai_responses"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_hook_preserves_complete_parallel_tool_groups_on_wire_and_cached_retries(
    chat_sdk, source, count, rounds, cached, stream
):
    batch, result_ids = [], []
    for round_index in range(rounds):
        ids = [f"round_{round_index}_{index}" for index in range(count)]
        result_ids.extend(reversed(ids))
        batch.extend(call_messages(source, ids))
        batch.extend(result_messages(source, reversed(ids)))
    before = deepcopy(batch)
    snapshots = []

    def hook(draft, losses, context):
        snapshots.append(deepcopy(draft["messages"]))
        move_tool_images_to_user_message(draft, losses, context)
        assert_chat_tool_sequence(draft["messages"])

    async with configured_llm("openai_chat", source, cached) as llm:
        llm.set_semantic_loss_policy(semantic_loss_protocol="hook", input_semantic_loss_hook=hook)
        for attempt in range(2):
            result = await llm.generate(messages=batch, stream=stream)
            if stream:
                await consume(result)
            body = wire_request(chat_sdk)
            assert_chat_tool_sequence(body["messages"])
            assert image_urls(body["messages"][-1]) == [
                f"https://example.test/{key}.png" for key in result_ids
            ]
            references = image_references(body["messages"][-1])
            assert len(references) == len(set(references)) == len(result_ids)
            tools = {m["tool_call_id"]: m for m in body["messages"] if m["role"] == "tool"}
            for reference, call_id in zip(references, result_ids, strict=True):
                assert reference in tools[call_id]["content"]
            assert len([m for m in body["messages"] if m["role"] == "tool"]) == count * rounds
            if attempt == 0:
                first = body["messages"]
            else:
                assert without_image_ids(body["messages"]) == without_image_ids(first)
                if cached:
                    assert body["messages"] == first
        await llm.generate(messages=[{"role": "user", "content": "next"}])
        following = wire_request(chat_sdk)["messages"]
        assert_chat_tool_sequence(following)
        if cached:
            assert following[:-1] == first
    assert snapshots
    for messages in snapshots:
        assert_chat_tool_sequence(messages)
    assert batch == before


@pytest.mark.parametrize("source", ["anthropic", "openai_responses"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("vision", [False, True])
async def test_unsupported_image_markers_preserve_parallel_tool_pairing(
    chat_sdk, source, cached, vision
):
    batch = [*call_messages(source, ["a", "b"]), *result_messages(source, ["b", "a"])]
    async with configured_llm(
        "openai_chat", source, cached, supports_vision=Config(default_value=vision)
    ) as llm:
        await llm.generate(messages=batch)
    messages = wire_request(chat_sdk)["messages"]
    assert_chat_tool_sequence(messages)
    if vision:
        assert len(image_urls(messages[-1])) == 2
    else:
        assert not image_urls(messages[-1])
        assert str(messages[-1]).count("[Unsupported Image]") == 2


@pytest.mark.parametrize("source", ["anthropic", "openai_responses"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "repair",
    [
        "orphan_before",
        "orphan_after",
        "idless_result",
        "duplicate_call",
        "duplicate_result",
        "missing_result",
        "missing_all_results",
        "image_only_results",
    ],
)
async def test_pairing_fallback_and_image_hook_compose_on_final_wire(
    chat_sdk, source, cached, repair
):
    calls = call_messages(source, ["a", "b"])
    results = result_messages(source, ["a", "b"])
    expected_ids = ["a", "b"]
    if repair == "orphan_before":
        calls = [*result_messages(source, ["orphan"]), *calls]
    elif repair == "orphan_after":
        results.extend(result_messages(source, ["orphan"]))
    elif repair == "idless_result":
        if source == "anthropic":
            results[-1]["content"][0].pop("tool_use_id")
        else:
            results[-1].pop("call_id")
        expected_ids = ["a"]
    elif repair == "duplicate_call":
        calls.extend(call_messages(source, ["a"]))
    elif repair == "duplicate_result":
        results.extend(result_messages(source, ["a"]))
        expected_ids = ["a", "a", "b"]
    elif repair == "missing_result":
        results.pop()
        expected_ids = ["a"]
    elif repair == "missing_all_results":
        results.clear()
        expected_ids = []
    elif repair == "image_only_results":
        for message in results:
            content = (
                message["content"][0]["content"] if source == "anthropic" else message["output"]
            )
            content.pop(0)
    batch = [*calls, *results]
    before = deepcopy(batch)
    async with configured_llm("openai_chat", source, cached) as llm:
        for _ in range(2):
            await llm.generate(messages=batch)
            messages = wire_request(chat_sdk)["messages"]
            assert_chat_tool_sequence(messages)
            tools = {m["tool_call_id"]: m for m in messages if m["role"] == "tool"}
            assert set(tools) == {"a", "b"}
            assert len([m for m in messages if m["role"] == "tool"]) == 2
            if repair == "duplicate_result":
                assert IMAGE_REFERENCE.sub("", tools["a"]["content"]) == "result:aresult:a"
            if repair in ("idless_result", "missing_result", "missing_all_results"):
                assert tools["b"]["content"] == ""
            if repair == "image_only_results":
                assert all(
                    IMAGE_REFERENCE.sub("", tool["content"]) == "" for tool in tools.values()
                )
                assert all(len(image_references(tool)) == 1 for tool in tools.values())
            if expected_ids:
                assert image_urls(messages[-1]) == [
                    f"https://example.test/{key}.png" for key in expected_ids
                ]
            else:
                assert not any(m["role"] == "user" for m in messages)
            assert "orphan" not in str(messages)
    assert batch == before


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    ("source", "interruption"),
    [
        ("anthropic", "user_before_results"),
        ("openai_responses", "user_before_results"),
        ("anthropic", "user_between_results"),
        ("openai_responses", "user_between_results"),
        ("anthropic", "assistant_between_results"),
        ("openai_responses", "assistant_between_results"),
        ("openai_responses", "system_between_results"),
        ("openai_responses", "developer_between_results"),
        ("anthropic", "mixed_text_and_results"),
        ("anthropic", "missing_result_with_interruption"),
        ("openai_responses", "missing_result_with_interruption"),
    ],
)
async def test_tool_results_stay_contiguous_when_history_contains_ordinary_text(
    chat_sdk, source, interruption, cached
):
    calls = call_messages(source, ["a", "b"])
    results = result_messages(source, ["a", "b"])
    text = {"role": "user", "content": "ordinary text"}
    if interruption == "user_before_results":
        batch = [*calls, text, *results]
    elif interruption == "mixed_text_and_results":
        batch = [
            *calls,
            {
                "role": "user",
                "content": [
                    *results[0]["content"],
                    *results[1]["content"],
                    {"type": "text", "text": "ordinary text"},
                ],
            },
        ]
    elif interruption == "missing_result_with_interruption":
        batch = [*calls, text, results[0]]
    else:
        text["role"] = interruption.removesuffix("_between_results")
        batch = [*calls, results[0], text, results[1]]
    before = deepcopy(batch)
    before_hook = []

    def hook(draft, losses, context):
        before_hook.append(deepcopy(draft["messages"]))
        move_tool_images_to_user_message(draft, losses, context)

    async with configured_llm("openai_chat", source, cached) as llm:
        llm.set_semantic_loss_policy(semantic_loss_protocol="hook", input_semantic_loss_hook=hook)
        await llm.generate(messages=batch)
        messages = wire_request(chat_sdk)["messages"]
        await llm.generate(messages=batch)
        retry = wire_request(chat_sdk)["messages"]
        assert without_image_ids(retry) == without_image_ids(messages)
        if cached:
            assert retry == messages
        await llm.generate(messages=[{"role": "user", "content": "next"}])
        following = wire_request(chat_sdk)["messages"]
        assert_chat_tool_sequence(following)
        if cached:
            assert following[:-1] == messages
    assert messages[-1]["role"] == "user"
    assert len(image_urls(messages[-1])) == (1 if "missing_result" in interruption else 2)
    assert "ordinary text" in str(messages)
    assert batch == before
    assert len(before_hook) == 2
    for snapshot in before_hook:
        assert_chat_tool_sequence(snapshot)
    # Tool text gains references in place; all other existing messages retain order.
    restored = deepcopy(messages[:-1])
    for message in restored:
        if message["role"] == "tool":
            message["content"] = IMAGE_REFERENCE.sub("", message["content"])
    assert restored == before_hook[0]
    assert before_hook[1] == (retry if cached else before_hook[0])
    assert sorted(call["id"] for m in messages for call in m.get("tool_calls", [])) == ["a", "b"]
    assert sorted(m["tool_call_id"] for m in messages if m["role"] == "tool") == ["a", "b"]
    # Deliberately enforce the requested protocol contract, without xfail or a
    # second normalization pass that could conceal an invalid wire sequence.
    assert_chat_tool_sequence(messages)


@pytest.mark.parametrize("source", ["anthropic", "openai_responses"])
@pytest.mark.parametrize("cached", [False, True])
async def test_interleaved_tool_rounds_keep_each_call_group_contiguous(chat_sdk, source, cached):
    batch = [
        *call_messages(source, ["a", "b"]),
        *result_messages(source, ["a"]),
        {"role": "user", "content": "between rounds"},
        *call_messages(source, ["c", "d"]),
        *result_messages(source, ["d", "b", "c"]),
    ]
    before = deepcopy(batch)
    async with configured_llm("openai_chat", source, cached) as llm:
        for _ in range(2):
            await llm.generate(messages=batch)
            messages = wire_request(chat_sdk)["messages"]
            assert_chat_tool_sequence(messages)
            assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
                "a",
                "b",
                "d",
                "c",
            ]
            assert image_urls(messages[-1]) == [
                f"https://example.test/{key}.png" for key in ("a", "d", "b", "c")
            ]
            assert [m["content"] for m in messages if m["role"] == "user"][:-1] == [
                "between rounds"
            ]
    assert batch == before
