"""Preset image relocation preserves native data and explicit cache repairs."""

from copy import deepcopy
from inspect import isawaitable

import pytest

from my_llm import (
    Config,
    MyCodec,
    MyLLM,
    SemanticLossContext,
    SemanticLossError,
    SemanticLossGroup,
    SemanticLossItem,
    TextContent,
    UserMessage,
    omit,
)
from my_llm.cache.message_cache import MessageCacheBatch
from tests.helpers.chat_history import assert_chat_tool_sequence
from tests.helpers.profiles import parameter
from tests.helpers.sdk import MESSAGE_TYPES, assert_input_native
from tests.helpers.tool_images import (
    DATA,
    HOOKS,
    URL,
    configured_llm,
    expected_image_message,
    history,
    image_references,
    picture_history,
    prepare,
    without_image_ids,
)
from tests.performance.workloads import PNG, profiles


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("source", ["anthropic", "openai_responses", "middleware"])
@pytest.mark.parametrize("cached", [False, True])
async def test_presets_move_images_without_mutating_inputs_and_preserve_retries(
    target, source, cached
):
    messages = history(source)
    before = deepcopy(messages)
    llm = configured_llm(target, source, cached)
    first = await prepare(llm, messages)
    retry = await prepare(llm, messages)
    key = "messages"
    references = image_references(first[key][-1])
    assert len(references) == len(set(references)) == 2
    expected = expected_image_message(target, references)
    assert first[key][-1] == expected
    assert without_image_ids(first[key]) == without_image_ids(retry[key])
    if cached:
        assert first[key] == retry[key]
    if target == "openai_chat":
        assert_chat_tool_sequence(first[key])
        assert_chat_tool_sequence(retry[key])
    tool = next(message for message in first[key] if message["role"] == "tool")
    assert tool["content"] == f"before{references[0]}{references[1]}after"
    assert_input_native(first[key][-1], MESSAGE_TYPES[target])
    assert messages == before
    if cached:
        # Request edits and unrelated generation options cannot replace history.
        first[key][-1].clear()
        next_input = (
            [UserMessage("user", [TextContent("next")])]
            if source == "middleware"
            else [{"role": "user", "content": "next"}]
        )
        following = await prepare(llm, next_input)
        assert following[key][-2] == expected
        assert following[key][:-1] == retry[key]
        if target == "openai_chat":
            assert_chat_tool_sequence(following[key])


@pytest.mark.parametrize("target", HOOKS)
async def test_presets_retry_after_cache_extension_failure(monkeypatch, target):
    replace = MessageCacheBatch.replace
    attempts = 0

    def fail_once(batch, messages):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cache write failed")
        replace(batch, messages)

    monkeypatch.setattr(MessageCacheBatch, "replace", fail_once)
    llm = configured_llm(target, cached=True)
    with pytest.raises(RuntimeError, match="cache write failed"):
        await prepare(llm, picture_history())
    request = await prepare(llm, picture_history())
    retry = await prepare(llm, picture_history())
    key = "messages"
    assert request[key][-1] == expected_image_message(target, image_references(request[key][-1]))
    assert retry[key] == request[key]
    assert attempts == 2


@pytest.mark.parametrize("target", HOOKS)
async def test_expiring_cache_allows_same_input_to_be_handled_again(target):
    llm = configured_llm(target, cached=True)
    first = await prepare(llm, picture_history())
    llm.codec.expire_message_cache(llm._message_cache_key)
    following = await prepare(llm, picture_history())
    key = "messages"
    assert without_image_ids(first[key]) == without_image_ids(following[key])
    assert first[key] != following[key]
    assert following[key][-1] == expected_image_message(
        target, image_references(following[key][-1])
    )


@pytest.mark.parametrize("target", HOOKS)
async def test_presets_accept_unrelated_losses_while_still_moving_images(target):
    llm = configured_llm(
        target, max_tokens=parameter("max_tokens", protocol=target, supports=False)
    )
    request = await prepare(llm, picture_history(), max_tokens=128)
    key = "messages"
    assert request[key][-1] == expected_image_message(target, image_references(request[key][-1]))


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("vision", [Config(default_value=False), None, omit])
async def test_presets_insert_placeholders_for_models_without_vision(target, vision):
    llm = configured_llm(target, cached=True, supports_vision=vision)
    request = await prepare(llm, picture_history())
    assert URL not in str(request) and DATA not in str(request)
    tool = next(message for message in request["messages"] if message["role"] == "tool")
    references = image_references(tool)
    assert tool["content"] == f"before{references[0]}{references[1]}after"
    assert str(request).count("[Unsupported Image]") == 2
    key = "messages"
    retry = await prepare(llm, picture_history())
    assert retry[key] == request[key]
    following = await prepare(llm, [{"role": "user", "content": "next"}])
    assert following[key][:-1] == request[key]


@pytest.mark.parametrize("target", HOOKS)
def test_default_loss_policy_does_not_call_presets(target):
    with pytest.raises(SemanticLossError):
        MyCodec().convert(picture_history(), input_format="anthropic", target_format=target)


@pytest.mark.parametrize("target", HOOKS)
async def test_presets_use_the_resolved_profile_snapshot(target):
    provider, model = profiles(target)
    llm = MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="anthropic",
        semantic_loss_protocol="hook",
        input_semantic_loss_hook=HOOKS[target],
    )
    model.supports_vision = Config(default_value=False)
    request = await prepare(llm, picture_history())
    key = "messages"
    assert request[key][-1] == expected_image_message(target, image_references(request[key][-1]))


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("source", ["openai_chat"])
async def test_model_image_losses_also_receive_markers_for_native_user_images(target, source):
    message = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}],
    }
    request = await prepare(
        configured_llm(target, source, supports_vision=Config(default_value=False)), [message]
    )
    assert PNG not in str(request)
    assert str(request).count("[Unsupported Image]") == 1


@pytest.mark.parametrize("target", HOOKS)
async def test_presets_treat_output_or_other_losses_as_handled(target):
    handler = HOOKS[target]
    draft = {"messages": []}
    losses = SemanticLossGroup([SemanticLossItem(object(), "$", "unrelated")])
    for context in (
        SemanticLossContext("anthropic", target, "input", "request"),
        SemanticLossContext("anthropic", target, "output", "result"),
    ):
        result = handler(draft, losses, context)
        if isawaitable(result):
            await result
    assert draft == {"messages": []}


async def test_repeated_image_values_are_not_deduplicated_inside_one_tool_result():
    messages = picture_history()
    messages[1]["content"][0]["content"].insert(
        2, deepcopy(messages[1]["content"][0]["content"][1])
    )
    request = await prepare(configured_llm("openai_chat"), messages)
    assert [
        part["image_url"]["url"]
        for part in request["messages"][-1]["content"]
        if part["type"] == "image_url"
    ] == [
        URL,
        URL,
        f"data:image/png;base64,{DATA}",
    ]
    references = image_references(request["messages"][-1])
    assert len(references) == len(set(references)) == 3
    tool = next(message for message in request["messages"] if message["role"] == "tool")
    assert tool["content"] == "before" + "".join(references) + "after"
