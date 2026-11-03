"""Real SDK requests retain relocated images across retries and new input rounds."""

import json
from copy import deepcopy
from inspect import isawaitable

import pytest

from my_llm import MyLLM
from tests.helpers.chat_history import assert_chat_tool_sequence
from tests.helpers.protocols import stream_events
from tests.helpers.schema import validate_request
from tests.helpers.tool_images import HOOKS, consume, request_history, responses_history
from tests.helpers.tool_images import unavailable_tool_images as unavailable_tool_images
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.workloads import profiles

pytestmark = pytest.mark.usefixtures("unavailable_tool_images")


@pytest.mark.parametrize("target", HOOKS)
@pytest.mark.parametrize("stream", [False, True])
async def test_relocated_images_survive_retry_and_incremental_requests(target, stream):
    drafts = []
    handler = HOOKS[target]

    async def hook(draft, losses, context):
        assert context.profile is llm.resolved_profile
        assert context.message_cache is not None
        result = handler(draft, losses, context)
        if isawaitable(result):
            await result
        drafts.append(draft)

    provider, model = profiles(target)
    batch = responses_history()
    before = deepcopy(batch)
    with LocalSDK(target, wire_frames(target, stream_events(target), 43)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_responses",
            semantic_loss_protocol="hook",
            input_semantic_loss_hook=hook,
            enable_message_cache=True,
        ) as llm:

            async def generate(messages, *, streaming=False):
                result = await llm.generate(messages=messages, stream=streaming)
                return await consume(result) if streaming else result

            await generate(
                [
                    {"role": "system", "content": "keep instructions"},
                    {"role": "user", "content": "initial"},
                ]
            )
            await generate(batch, streaming=stream)
            first = json.loads(sdk.requests[-1].content)
            image_message = request_history(target, first)[-1]
            assert image_message["role"] == "user"
            assert "aW1hZ2U=" in str(image_message)
            # Hook-draft ownership is independent from the cached repair.
            key = "contents" if target == "gemini" else "messages"
            drafts[-1][key][-1].clear()
            await generate(batch, streaming=stream)
            retry = json.loads(sdk.requests[-1].content)
            assert request_history(target, retry) == request_history(target, first)
            await generate([{"role": "user", "content": "next"}], streaming=stream)
            following = json.loads(sdk.requests[-1].content)
            assert request_history(target, following)[:-1] == request_history(target, first)
            assert "keep instructions" in str(following)
            # A new tool round with equal image bytes must add a new image message.
            next_batch = responses_history()
            for item in next_batch:
                if item.get("type") in ("function_call", "function_call_output"):
                    item["call_id"] = "second-call"
            await generate(next_batch, streaming=stream)
            final = json.loads(sdk.requests[-1].content)
            assert request_history(target, final)[-1] == image_message
            assert str(request_history(target, final)).count("aW1hZ2U=") == 2
            assert batch == before
        for request in sdk.requests:
            body = json.loads(request.content)
            validate_request(target, body)
            if target == "openai_chat":
                assert_chat_tool_sequence(body["messages"])
        assert all(body.closed for body in sdk.bodies)


@pytest.mark.parametrize("target", HOOKS)
async def test_unconsumed_stream_retry_does_not_duplicate_cached_images(target):
    provider, model = profiles(target)
    with LocalSDK(target, wire_frames(target, stream_events(target), 47)) as sdk:
        async with MyLLM(
            provider=provider,
            model=model,
            stdio_protocol="openai_responses",
            semantic_loss_protocol="hook",
            input_semantic_loss_hook=HOOKS[target],
            enable_message_cache=True,
        ) as llm:
            pending = await llm.generate(messages=responses_history(), stream=True)
            assert not sdk.requests
            await pending.aclose()
            await consume(await llm.generate(messages=responses_history(), stream=True))
            request = json.loads(sdk.requests[-1].content)
            assert str(request_history(target, request)).count("aW1hZ2U=") == 1
            if target == "openai_chat":
                assert_chat_tool_sequence(request["messages"])
