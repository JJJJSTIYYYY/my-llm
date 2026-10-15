"""Regressions for request snapshots and copy-free internal message handling."""

from collections import Counter
from copy import deepcopy

import pytest
from conftest import FakeClient, chat_event, response_completed

from my_llm import (
    MessageChunk,
    MessageValidationError,
    ModelProfile,
    MyLLM,
    ProviderProfile,
    generate,
)
from my_llm.codec.chat import ChatCompletionsCodec
from my_llm.codec.common import request_parameters
from my_llm.codec.validation import validate_messages
from my_llm.profiles import resolve_profiles


class TrackedDict(dict):
    copies = Counter()

    def __init__(self, label, **values):
        super().__init__(values)
        self.label = label

    def __deepcopy__(self, memo):
        self.copies[self.label] += 1
        result = type(self)(self.label)
        memo[id(self)] = result
        result.update({key: deepcopy(value, memo) for key, value in self.items()})
        return result


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
@pytest.mark.parametrize("entrypoint", ["generate", "invoke"])
@pytest.mark.parametrize("input_format", ["chat_completions", "responses"])
async def test_message_snapshot_is_taken_once_at_generate_before_lazy_consumption(
    api_type, entrypoint, input_format
):
    TrackedDict.copies.clear()
    block = TrackedDict("block", type="text", text="original")
    message = TrackedDict("message", role="user", content=[block])
    if input_format == "responses":
        message["type"] = "message"
        block["type"] = "input_text"
    history = [message]
    client = FakeClient(
        [response_completed()] if api_type == "responses" else [chat_event(finish="stop")]
    )
    provider, model = ProviderProfile(api_type=api_type), ModelProfile(model="demo")
    if entrypoint == "generate":
        stream = await generate(
            provider=provider, model=model, messages=history, client=client, stream=True
        )
    else:
        stream = await MyLLM(provider=provider, model=model, client=client).generate(
            messages=history, stream=True
        )
    assert TrackedDict.copies == {"message": 1, "block": 1}
    block["text"] = "caller changed after generate"
    history.append({"role": "user", "content": "new message"})
    async with stream:
        async for _ in stream:
            pass
    wire_history = client.requests[0]["input" if api_type == "responses" else "messages"]
    assert len(wire_history) == 1
    assert wire_history[0]["content"][0]["text"] == "original"
    wire_history[0]["content"][0]["text"] = "transport changed"
    assert block["text"] == "caller changed after generate"
    assert TrackedDict.copies == {"message": 1, "block": 1}


def test_validation_and_codec_borrow_owned_history_without_copying(profile_factory):
    TrackedDict.copies.clear()
    message = TrackedDict(
        "message", role="user", content=[TrackedDict("block", type="text", text="hello")]
    )
    history = [message]
    validated = validate_messages(history)
    assert validated[0] is message
    request = ChatCompletionsCodec(profile_factory()).encode(history, tools=None)
    assert request["messages"][0]["content"] is message["content"]
    assert TrackedDict.copies == {}
    assert history == [message]


@pytest.mark.parametrize(
    "location", ["uniform_same", "uniform_turns", "native_same", "native_turns", "mixed_turns"]
)
def test_read_only_history_validation_keeps_duplicate_call_detection(tool_call, location):
    uniform = {"role": "assistant", "tool_calls": [tool_call]}
    item = {"type": "function_call", "call_id": tool_call["id"], **tool_call["function"]}
    native = {"role": "assistant", "response_output": [item]}
    histories = {
        "uniform_same": [{"role": "assistant", "tool_calls": [tool_call, tool_call]}],
        "uniform_turns": [uniform, uniform],
        "native_same": [{"role": "assistant", "response_output": [item, item]}],
        "native_turns": [native, native],
        "mixed_turns": [uniform, native],
    }
    with pytest.raises(MessageValidationError, match="duplicate"):
        validate_messages(histories[location])


async def test_response_native_replay_is_owned_by_request():
    original = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "original"}],
    }
    history = [{"role": "assistant", "response_output": [original]}]
    client = FakeClient([response_completed()])
    stream = await generate(
        provider=ProviderProfile(api_type="responses"),
        model="demo",
        messages=history,
        client=client,
        stream=True,
    )
    original["summary"][0]["text"] = "changed"
    async with stream:
        async for _ in stream:
            pass
    assert client.requests[0]["input"][0]["summary"][0]["text"] == "original"
    assert original["summary"][0]["text"] == "changed"


def test_chunk_concatenation_defers_payload_copy_until_public_snapshot():
    TrackedDict.copies.clear()
    call = TrackedDict(
        "tool", id="call_1", type="function", function={"name": "tool", "arguments": "{}"}
    )
    first = MessageChunk(
        tool_calls=[call],
        usage={"details": {"tokens": 3}},
        response_output=[{"type": "reasoning", "summary": [{"text": "reason"}]}],
    )
    combined = first
    for _ in range(100):
        combined += MessageChunk(content="x")
    assert TrackedDict.copies == {}
    assert combined.tool_calls is not first.tool_calls
    result = combined.to_message()
    assert TrackedDict.copies == {"tool": 1}
    result["tool_calls"][0]["function"]["name"] = "changed"
    result["usage"]["details"]["tokens"] = 99
    result["response_output"][0]["summary"][0]["text"] = "changed"
    assert call["function"]["name"] == "tool"
    assert first.usage["details"]["tokens"] == 3
    assert first.response_output[0]["summary"][0]["text"] == "reason"


def test_resolved_profiles_and_request_extras_have_independent_snapshots():
    body = {"nested": {"value": "original"}}
    profile = resolve_profiles(ProviderProfile(extra_body=body), ModelProfile(model="demo"))
    body["nested"]["value"] = "caller change"
    assert profile.get("extra_body")["nested"]["value"] == "original"
    first = request_parameters(profile)
    first["extra_body"]["nested"]["value"] = "request change"
    assert request_parameters(profile)["extra_body"]["nested"]["value"] == "original"
    extra = {"caller": {"value": "request input"}}
    second = request_parameters(profile, extra_body=extra)
    extra["caller"]["value"] = "later change"
    assert second["extra_body"]["caller"]["value"] == "request input"


async def test_non_stream_large_response_and_error_keep_all_text():
    from my_llm import IncompleteStreamError

    events = [chat_event(content=str(index), reasoning="r") for index in range(2000)]
    expected = "".join(str(index) for index in range(2000))
    client = FakeClient([*events, chat_event(finish="stop")])
    result = await generate(
        provider=ProviderProfile(),
        model="demo",
        messages=[{"role": "user", "content": "x"}],
        client=client,
    )
    assert result["content"] == expected
    assert result["reasoning_content"] == "r" * 2000
    client = FakeClient(events)
    with pytest.raises(IncompleteStreamError) as failure:
        await generate(
            provider=ProviderProfile(),
            model="demo",
            messages=[{"role": "user", "content": "x"}],
            client=client,
        )
    assert failure.value.partial_message["content"] == expected
