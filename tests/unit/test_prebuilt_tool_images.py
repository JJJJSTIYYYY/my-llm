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
    gemini_image_history,
    history,
    picture_history,
    prepare,
)
from tests.helpers.tool_images import unavailable_tool_images as unavailable_tool_images
from tests.performance.workloads import PNG, profiles

pytestmark = pytest.mark.usefixtures("unavailable_tool_images")


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
    key = "contents" if target == "gemini" else "messages"
    expected = expected_image_message(target)
    assert first[key][-1] == retry[key][-1] == expected
    assert first[key] == retry[key]
    if target == "openai_chat":
        assert_chat_tool_sequence(first[key])
        assert_chat_tool_sequence(retry[key])
    assert "beforeafter" in str(first)
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
    extend = MessageCacheBatch.extend
    attempts = 0

    def fail_once(batch, messages):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cache write failed")
        extend(batch, messages)

    monkeypatch.setattr(MessageCacheBatch, "extend", fail_once)
    llm = configured_llm(target, cached=True)
    with pytest.raises(RuntimeError, match="cache write failed"):
        await prepare(llm, picture_history())
    request = await prepare(llm, picture_history())
    retry = await prepare(llm, picture_history())
    key = "contents" if target == "gemini" else "messages"
    assert request[key][-1] == expected_image_message(target)
    assert retry[key] == request[key]
    assert attempts == 2


@pytest.mark.parametrize("target", HOOKS)
async def test_expiring_cache_allows_same_input_to_be_handled_again(target):
    llm = configured_llm(target, cached=True)
    first = await prepare(llm, picture_history())
    llm.codec.expire_message_cache(llm._message_cache_key)
    following = await prepare(llm, picture_history())
    key = "contents" if target == "gemini" else "messages"
    assert first[key] == following[key]
    assert following[key][-1] == expected_image_message(target)


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize("media", ["inline", "file"])
async def test_presets_read_gemini_tool_images_without_inspecting_unknown_parts(target, sdk, media):
    messages, _ = gemini_image_history(media, sdk)
    request = await prepare(configured_llm(target, "gemini"), messages)
    key = "contents" if target == "gemini" else "messages"
    if target == "ollama" and media == "file":
        assert request[key][-1] == {"role": "user", "content": "[Unsupported Image]"}
    else:
        message = request[key][-1]
        assert message["role"] == "user"
        assert ("cG5n" if media == "inline" else "https://example.test/p.png") in str(message)
        assert_input_native(message, MESSAGE_TYPES[target])
    if target == "openai_chat":
        assert_chat_tool_sequence(request[key])


@pytest.mark.parametrize("target", HOOKS)
async def test_presets_accept_unrelated_losses_while_still_moving_images(target):
    llm = configured_llm(
        target, max_tokens=parameter("max_tokens", protocol=target, supports=False)
    )
    request = await prepare(llm, picture_history(), max_tokens=128)
    key = "contents" if target == "gemini" else "messages"
    assert request[key][-1] == expected_image_message(target)


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("vision", [Config(default_value=False), None, omit])
async def test_presets_insert_placeholders_for_models_without_vision(target, vision):
    llm = configured_llm(target, cached=True, supports_vision=vision)
    request = await prepare(llm, picture_history())
    assert URL not in str(request) and DATA not in str(request)
    assert "beforeafter" in str(request)
    assert str(request).count("[Unsupported Image]") == 2
    key = "contents" if target == "gemini" else "messages"
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
    key = "contents" if target == "gemini" else "messages"
    assert request[key][-1] == expected_image_message(target)


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("source", ["openai_chat", "ollama"])
async def test_model_image_losses_also_receive_markers_for_native_user_images(target, source):
    message = (
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
            ],
        }
        if source == "openai_chat"
        else {"role": "user", "content": "picture", "images": [PNG]}
    )
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
    assert [part["image_url"]["url"] for part in request["messages"][-1]["content"]] == [
        URL,
        URL,
        f"data:image/png;base64,{DATA}",
    ]
