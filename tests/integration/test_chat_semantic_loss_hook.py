"""Chat repairs reach the actual SDK wire and persist across cached input batches."""

import json
from copy import deepcopy

import pytest

from my_llm import (
    AssistantMessage,
    Config,
    ReasoningContent,
    TextContent,
    UserMessage,
)
from my_llm.cache.message_cache import MessageCacheBatch
from my_llm.codec.openai.chat.prebuilt import reasoning_content_feedback
from tests.helpers.chat_history import assert_chat_tool_sequence
from tests.helpers.profiles import parameter
from tests.helpers.protocols import stream_events
from tests.helpers.schema import validate_request
from tests.helpers.tool_images import (
    IMAGE_REFERENCE,
    configured_llm,
    consume,
    history,
    image_references,
    prepare,
    without_image_ids,
)
from tests.performance.transport import LocalSDK, wire_frames


def reasoning_history(source):
    messages = history(source)
    if source == "middleware":
        messages[0].protocol = "anthropic"
        messages[0].content[:0] = [
            ReasoningContent("tool one\n", {"type": "thinking", "signature": "opaque-secret"}),
            ReasoningContent("tool two", {"type": "redacted_thinking", "data": "opaque-secret"}),
        ]
        return [
            AssistantMessage(
                "assistant",
                [
                    ReasoningContent("plain thought", {"encrypted_content": "opaque-secret"}),
                    TextContent("plain answer"),
                ],
                protocol="openai_responses",
            ),
            UserMessage("user", [TextContent("separator")]),
            *messages,
        ]
    if source == "anthropic":

        def block(text):
            return {"type": "thinking", "thinking": text, "signature": "opaque-secret"}

        messages[0]["content"][:0] = [block("tool one\n"), block("tool two")]
        plain = [
            {
                "role": "assistant",
                "content": [block("plain thought"), {"type": "text", "text": "plain answer"}],
            }
        ]
    else:

        def block(text, identity):
            return {
                "type": "reasoning",
                "id": identity,
                "summary": [{"type": "summary_text", "text": text}],
                "encrypted_content": "opaque-secret",
            }

        messages[:0] = [block("tool one\n", "rs_one"), block("tool two", "rs_two")]
        plain = [
            block("plain thought", "rs_plain"),
            {"role": "assistant", "content": "plain answer"},
        ]
    return [*plain, {"role": "user", "content": "separator"}, *messages]


def user_message(source, text):
    return (
        UserMessage("user", [TextContent(text)])
        if source == "middleware"
        else {
            "role": "user",
            "content": text,
        }
    )


def assert_repairs(messages, mode):
    assert_chat_tool_sequence(messages)
    plain = next(m for m in messages if m.get("content") == "plain answer")
    tool = next(m for m in messages if m.get("tool_calls"))
    assert plain.get("reasoning_content") == ("plain thought" if mode == "always" else None)
    assert tool.get("reasoning_content") == (None if mode == "none" else "tool one\ntool two")
    references = image_references(messages[-1])
    assert len(references) == len(set(references)) == 2
    result = next(m for m in messages if m["role"] == "tool")
    assert result["content"] == "before" + "".join(references) + "after"
    assert "opaque-secret" not in str(messages)


@pytest.mark.parametrize("source", ["anthropic", "openai_responses", "middleware"])
@pytest.mark.parametrize("mode", ["always", "when_tool_calls", "none"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_combined_repairs_preserve_reasoning_images_and_cache_on_actual_wire(
    source, mode, cached, stream
):
    batch = reasoning_history(source)
    before = deepcopy(batch)
    with LocalSDK(
        "openai_chat", wire_frames("openai_chat", stream_events("openai_chat"), 37)
    ) as sdk:
        async with configured_llm(
            "openai_chat",
            source,
            cached,
            should_feedback_reasoning_item=Config(default_value=mode),
        ) as llm:
            if cached:
                await llm.generate(messages=[user_message(source, "cached prefix")])
            requests = []
            for _ in range(2):
                result = await llm.generate(messages=deepcopy(batch), stream=stream)
                if stream:
                    await consume(result)
                body = json.loads(sdk.requests[-1].content)
                validate_request("openai_chat", body)
                messages = body["messages"]
                assert_repairs(messages, mode)
                requests.append(messages)
                if cached:
                    assert llm.codec._message_cache.get(llm._message_cache_key) == messages
                    assert messages[0]["content"] == "cached prefix"
            assert without_image_ids(requests[0]) == without_image_ids(requests[1])
            if cached:
                assert requests[0] == requests[1]
                await llm.generate(messages=[user_message(source, "next")])
                following = json.loads(sdk.requests[-1].content)["messages"]
                assert following[:-1] == requests[0]
            assert batch == before


@pytest.mark.parametrize("source", ["anthropic", "openai_responses", "middleware"])
async def test_reasoning_hook_can_be_used_independently_and_updates_existing_cache(source):
    llm = configured_llm("openai_chat", source, True)
    llm.set_semantic_loss_policy(
        semantic_loss_protocol="hook", input_semantic_loss_hook=reasoning_content_feedback
    )
    await prepare(llm, [user_message(source, "prefix")])
    request = await prepare(llm, reasoning_history(source))
    tool = next(m for m in request["messages"] if m.get("tool_calls"))
    assert tool["reasoning_content"] == "tool one\ntool two"
    plain = next(m for m in request["messages"] if m.get("content") == "plain answer")
    assert plain["reasoning_content"] == "plain thought"
    assert not IMAGE_REFERENCE.search(str(request))
    assert llm.codec._message_cache.get(llm._message_cache_key) == request["messages"]
    expected = deepcopy(request["messages"])
    plain["reasoning_content"] = "caller edit"
    retry = await prepare(llm, reasoning_history(source))
    assert retry["messages"] == expected
    following = await prepare(llm, [user_message(source, "next")])
    assert following["messages"][:-1] == expected


@pytest.mark.parametrize("failure_at", [1, 2])
async def test_combined_hook_retries_after_either_cache_write_fails(monkeypatch, failure_at):
    write = MessageCacheBatch.replace
    attempts = 0

    def fail_second(batch, messages):
        nonlocal attempts
        attempts += 1
        if attempts == failure_at:
            raise RuntimeError("image cache failed")
        write(batch, messages)

    monkeypatch.setattr(MessageCacheBatch, "replace", fail_second)
    llm = configured_llm("openai_chat", "middleware", True)
    messages = reasoning_history("middleware")
    with pytest.raises(RuntimeError, match="image cache failed"):
        await prepare(llm, messages)
    request = await prepare(llm, deepcopy(messages))
    assert_repairs(request["messages"], "always")
    assert (await prepare(llm, messages))["messages"] == request["messages"]
    assert attempts == 3


async def test_new_cached_round_retains_prior_repairs_and_gets_new_image_ids():
    llm = configured_llm("openai_chat", "anthropic", True)
    first = await prepare(llm, reasoning_history("anthropic"))
    before = deepcopy(first["messages"])
    batch = reasoning_history("anthropic")
    batch[0]["content"][0]["thinking"] = "second plain thought"
    for message in batch:
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                block["id"] = "second-call"
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                block["tool_use_id"] = "second-call"
    following = await prepare(llm, batch)
    assert following["messages"][: len(before)] == before
    current = following["messages"][len(before) :]
    assert current[0]["reasoning_content"] == "second plain thought"
    assert (
        next(m for m in current if m.get("tool_calls"))["reasoning_content"] == "tool one\ntool two"
    )
    assert set(image_references(current[-1])).isdisjoint(image_references(before[-1]))
    assert (await prepare(llm, deepcopy(batch)))["messages"] == following["messages"]
    assert_chat_tool_sequence(following["messages"])


@pytest.mark.parametrize("feedback", ["original replay", {"encrypted_content": "opaque-secret"}])
async def test_native_chat_uses_visible_fallback_only_when_feedback_is_unrepresentable(feedback):
    llm = configured_llm("openai_chat", "middleware", True)
    batch = history("middleware")
    batch[0].protocol = "openai_chat"
    batch[0].content.insert(0, ReasoningContent("visible thought", feedback))
    request = await prepare(llm, batch)
    assistant = next(m for m in request["messages"] if m.get("tool_calls"))
    assert assistant["reasoning_content"] == (
        feedback if isinstance(feedback, str) else "visible thought"
    )
    assert "opaque-secret" not in str(request)
    assert (await prepare(llm, deepcopy(batch)))["messages"] == request["messages"]


@pytest.mark.parametrize("visible", [None, ""])
async def test_opaque_reasoning_never_becomes_visible_text(visible):
    llm = configured_llm("openai_chat", "middleware", True)
    batch = [
        AssistantMessage(
            "assistant",
            [ReasoningContent(visible, {"encrypted_content": "opaque-secret"})],
            protocol="openai_responses",
        )
    ]
    request = await prepare(llm, batch)
    assert "opaque-secret" not in str(request)
    assert request["messages"][0].get("reasoning_content") == visible


async def test_reasoning_hook_respects_disabled_model_reasoning():
    llm = configured_llm(
        "openai_chat",
        "middleware",
        True,
        reasoning_enable=parameter("reasoning_enable", supports=False),
    )
    request = await prepare(llm, reasoning_history("middleware"))
    assert all("reasoning_content" not in message for message in request["messages"])
