"""History feedback policies share image loss handling within native loops."""

from copy import deepcopy

import pytest

from my_llm import (
    AssistantMessage,
    ConfigurationError,
    MyCodec,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    ToolCall,
    ToolResult,
    UserMessage,
    omit,
)
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS


def reasoning_history(protocol):
    def reasoning(text):
        return ReasoningContent(
            text,
            signature="c2lnbmVk" if protocol in ("anthropic", "gemini") else None,
            opaque_protocol=protocol,
        )

    return [
        AssistantMessage(
            "assistant",
            [reasoning("plain reasoning"), TextContent("plain answer")],
            protocol=protocol,
        ),
        UserMessage("user", [TextContent("next question")]),
        AssistantMessage(
            "assistant",
            [reasoning("tool reasoning"), ToolCall("c", "f")],
            protocol=protocol,
        ),
        UserMessage("user", [ToolResult("c", [TextContent("tool result")])]),
        UserMessage("user", [TextContent("complete tail")]),
    ]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("mode", ["always", "when_tool_calls", "none", omit])
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_feedback_selects_history_without_semantic_loss(profile_factory, protocol, mode, policy):
    messages = reasoning_history(protocol)
    before = deepcopy(messages)
    observed = []
    codec = MyCodec(policy, input_semantic_loss_hook=lambda *args: observed.append(args))
    profile = profile_factory(
        protocol,
        feedback_reasoning_content=omit if mode is omit else config(mode),
        supports_function_calling=config(True),
    )

    def prepare():
        return codec.prepare(
            {"messages": messages, "tools": omit, "stream": False, "max_tokens": 128},
            profile,
            input_format="middleware",
            client={},
        )["request"]

    draft = prepare()
    assert not observed
    assert ("plain reasoning" in str(draft)) is (mode == "always")
    assert ("tool reasoning" in str(draft)) is (mode in ("always", "when_tool_calls"))
    assert "complete tail" in str(draft)
    assert messages == before


@pytest.mark.parametrize("target", ["openai_chat", "openai_responses", "ollama"])
def test_tool_feedback_keeps_split_responses_reasoning_and_native_item_ids(profile_factory, target):
    messages = [
        {
            "type": "reasoning",
            "id": "plain-id",
            "summary": [{"type": "summary_text", "text": "plain reasoning"}],
        },
        {"role": "user", "content": "next question"},
        {
            "type": "reasoning",
            "id": "tool-reasoning-id",
            "summary": [{"type": "summary_text", "text": "tool reasoning"}],
        },
        {
            "type": "message",
            "id": "message-id",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "checking", "annotations": [], "logprobs": []}
            ],
        },
        {
            "type": "function_call",
            "id": "function-id",
            "call_id": "c",
            "name": "f",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "c", "output": "tool result"},
    ]
    before = deepcopy(messages)
    draft = MyCodec().prepare(
        {"messages": messages, "tools": omit, "stream": False, "max_tokens": 128},
        profile_factory(
            target,
            feedback_reasoning_content=config("when_tool_calls"),
            supports_function_calling=config(True),
        ),
        input_format="openai_responses",
        client={},
    )["request"]
    assert "tool reasoning" in str(draft)
    assert "plain reasoning" not in str(draft)
    if target == "openai_responses":
        assert all(
            identity in str(draft)
            for identity in ("tool-reasoning-id", "message-id", "function-id")
        )
    assert messages == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_feedback_policy_does_not_filter_generated_reasoning(profile_factory, protocol):
    result = MyCodec().result(
        [reasoning_history(protocol)[0]],
        input_format=protocol,
        target_format=protocol,
        semantic_profile=profile_factory(protocol, feedback_reasoning_content=config("none")),
    )
    assert "plain reasoning" in str(result)


@pytest.mark.parametrize(
    "value", [True, False, "always", config(True), config(False), config("invalid"), config(None)]
)
def test_feedback_requires_a_wrapped_policy_value(profile_factory, value):
    with pytest.raises(ConfigurationError, match="feedback_reasoning_content"):
        profile_factory(feedback_reasoning_content=value)


@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_feedback_cache_commits_filtered_history_without_hook_edits(profile_factory, policy):
    def hook(draft, losses, context):
        draft["messages"].append({"role": "user", "content": "hook edit"})

    codec = MyCodec(policy, input_semantic_loss_hook=hook)
    profile = profile_factory(
        feedback_reasoning_content=config("when_tool_calls"), supports_function_calling=config(True)
    )
    inputs = {"messages": reasoning_history("openai_chat"), "tools": omit, "stream": False}
    for _ in range(2):

        def prepare():
            return codec.prepare(
                inputs,
                profile,
                input_format="middleware",
                client={},
                enable_message_cache=True,
                message_cache_key="feedback-test",
            )

        prepare()
    cached = codec._message_cache.get("feedback-test")
    assert len(cached) == 5
    assert "tool reasoning" in str(cached)
    assert "plain reasoning" not in str(cached)
    assert "hook edit" not in str(cached)


@pytest.mark.parametrize(
    "protocol,field", [("openai_chat", "reasoning_content"), ("ollama", "thinking")]
)
@pytest.mark.parametrize("mode", ["always", "when_tool_calls", "none"])
@pytest.mark.parametrize("has_calls", [True, False])
def test_native_scalar_reasoning_feedback_matches_message_policy(
    profile_factory, protocol, field, mode, has_calls
):
    message = {"role": "assistant", "content": "answer", field: "native reasoning"}
    if has_calls:
        message["tool_calls"] = [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            if protocol == "openai_chat"
            else {"function": {"name": "f", "arguments": {}}}
        ]
    request = {"messages": [message]}
    before = deepcopy(request)
    decoder = MyCodec().decoder(protocol)
    profile = profile_factory(
        protocol, feedback_reasoning_content=config(mode), supports_function_calling=config(True)
    )
    if mode == "always" or mode == "when_tool_calls" and has_calls:
        assert decoder.native_request(request, profile)["messages"][0][field] == "native reasoning"
    else:
        assert field not in decoder.native_request(request, profile)["messages"][0]
    assert request == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("mode", ["always", "when_tool_calls", "none", omit])
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_unsupported_reasoning_is_loss_regardless_of_feedback(
    profile_factory, protocol, mode, policy
):
    messages = reasoning_history(protocol)
    before = deepcopy(messages)
    observed = []
    codec = MyCodec(policy, input_semantic_loss_hook=lambda *args: observed.append(args))
    profile = profile_factory(
        protocol,
        supports_function_calling=config(True),
        reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=False),
        feedback_reasoning_content=omit if mode is omit else config(mode),
    )

    def prepare():
        return codec.prepare(
            {"messages": messages, "tools": omit, "stream": False, "max_tokens": 128},
            profile,
            input_format="middleware",
            client={},
        )["request"]

    losses = None
    if policy == "raise":
        with pytest.raises(SemanticLossError) as caught:
            prepare()
        draft, losses = caught.value.draft, caught.value.losses
    else:
        draft = prepare()
        if policy == "hook":
            assert observed[0][0] is draft
            losses = observed[0][1]
    assert "plain reasoning" not in str(draft)
    assert "tool reasoning" not in str(draft)
    assert "complete tail" in str(draft)
    if losses is not None:
        assert len(losses) == 2
        assert losses.items[0].origin_value is messages[0].content[0]
        assert losses.items[0].origin_path == "$.messages[0].content[0]"
        assert all("not support reasoning" in loss.reason for loss in losses)
    assert messages == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_reasoning_output_uses_output_hook(profile_factory, protocol):
    messages = [reasoning_history(protocol)[0]]
    observed = []
    codec = MyCodec("hook", output_semantic_loss_hook=lambda *args: observed.append(args))
    result = codec.result(
        messages,
        input_format=protocol,
        target_format=protocol,
        semantic_profile=profile_factory(
            protocol,
            reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=False),
        ),
    )
    assert "plain reasoning" not in str(result)
    assert "plain answer" in str(result)
    assert len(observed) == 1
    assert len(observed[0][1]) == 1
    assert "not support reasoning" in observed[0][1].items[0].reason
    assert observed[0][2].origin_source == "output"


@pytest.mark.parametrize(
    "protocol,field", [("openai_chat", "reasoning_content"), ("ollama", "thinking")]
)
@pytest.mark.parametrize("mode", ["always", "when_tool_calls", "none"])
def test_native_scalar_unsupported_reasoning_is_loss(profile_factory, protocol, field, mode):
    request = {"messages": [{"role": "assistant", "content": "answer", field: "reasoning"}]}
    before = deepcopy(request)
    profile = profile_factory(
        protocol,
        feedback_reasoning_content=config(mode),
        reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=False),
    )
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().decoder(protocol).native_request(request, profile)
    assert caught.value.losses.items[0].origin_path == f"$.messages[0].{field}"
    assert caught.value.losses.items[0].origin_value == "reasoning"
    assert "not support reasoning" in caught.value.losses.items[0].reason
    assert field not in caught.value.draft["messages"][0]
    assert request == before


def native_block_history(protocol, has_calls):
    if protocol == "anthropic":
        content = [
            {"type": "thinking", "thinking": "native reasoning", "signature": "signed"},
            {"type": "text", "text": "answer"},
        ]
        if has_calls:
            content.append({"type": "tool_use", "id": "c", "name": "f", "input": {}})
        return {"messages": [{"role": "assistant", "content": content}]}, content[0]
    if protocol == "gemini":
        parts = [{"text": "native reasoning", "thought": True}, {"text": "answer"}]
        if has_calls:
            parts.append({"function_call": {"name": "f", "args": {}}, "thought_signature": "c2ln"})
        return {"contents": [{"role": "model", "parts": parts}]}, parts[0]
    items = [
        {
            "type": "reasoning",
            "id": "rs1",
            "summary": [{"type": "summary_text", "text": "native reasoning"}],
        },
        {"role": "assistant", "content": "answer"},
    ]
    if has_calls:
        items.append({"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"})
    return {"input": items}, items[0]


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "openai_responses"])
@pytest.mark.parametrize("mode", ["always", "when_tool_calls", "none"])
@pytest.mark.parametrize("supports", [True, False])
@pytest.mark.parametrize("has_calls", [True, False])
def test_native_block_reasoning_uses_capability_and_feedback_separately(
    profile_factory, protocol, mode, supports, has_calls
):
    request, reasoning = native_block_history(protocol, has_calls)
    if protocol == "anthropic":
        request["max_tokens"] = 128
    before = deepcopy(request)
    decoder = MyCodec().decoder(protocol)
    profile = profile_factory(
        protocol,
        feedback_reasoning_content=config(mode),
        supports_function_calling=config(True),
        reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=supports),
    )
    if supports:
        draft = decoder.native_request(request, profile)
        assert ("native reasoning" in str(draft)) is (
            mode == "always" or mode == "when_tool_calls" and has_calls
        )
    else:
        with pytest.raises(SemanticLossError) as caught:
            decoder.native_request(request, profile)
        draft = caught.value.draft
        assert len(caught.value.losses) == 1
        assert caught.value.losses.items[0].origin_value is reasoning
        assert "not support reasoning" in caught.value.losses.items[0].reason
        assert "native reasoning" not in str(draft)
    assert "answer" in str(draft)
    if has_calls:
        assert "f" in str(draft)
        if protocol == "gemini":
            assert "thought_signature" in str(draft)
    assert request == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["delta", "block_start"])
def test_unsupported_reasoning_stream_records_loss_and_keeps_text(profile_factory, protocol, kind):
    profile = profile_factory(
        protocol,
        reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=False),
    )
    event = (
        {"type": "delta", "field": "thinking", "value": "disabled"}
        if kind == "delta"
        else {
            "type": "block_start",
            "block": {
                "type": "thinking",
                "thinking": "disabled",
                "signature": "signed",
                "mark": "raw",
            },
        }
    )
    events = [
        event,
        {"type": "delta", "field": "text", "value": "answer"},
    ]
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().decoder(protocol).chunks(
            events, semantic_profile=profile, source_protocol=protocol
        )
    assert len(caught.value.losses) == 1
    assert "not support reasoning" in caught.value.losses.items[0].reason
    assert "disabled" not in str(caught.value.draft)
    assert "answer" in str(caught.value.draft)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_history_tool_calls_and_results_are_losses_without_tool_definitions(
    profile_factory, protocol, policy
):
    messages = [
        AssistantMessage("assistant", [ToolCall("c", "f")]),
        UserMessage("user", [ToolResult("c", [TextContent("result")])]),
        UserMessage("user", [TextContent("tail")]),
    ]
    before = deepcopy(messages)
    observed = []
    codec = MyCodec(policy, input_semantic_loss_hook=lambda *args: observed.append(args))

    def prepare():
        return codec.prepare(
            {"messages": messages, "tools": omit, "stream": False, "max_tokens": 128},
            profile_factory(protocol, supports_function_calling=config(False)),
            input_format="middleware",
            client={},
        )["request"]

    losses = None
    if policy == "raise":
        with pytest.raises(SemanticLossError) as caught:
            prepare()
        draft, losses = caught.value.draft, caught.value.losses
    else:
        draft = prepare()
        if policy == "hook":
            losses = observed[0][1]
    assert "tail" in str(draft)
    assert "tool_calls" not in str(draft)
    assert "tool_use" not in str(draft)
    assert "'function_call':" not in str(draft)
    if losses is not None:
        assert len(losses) == 2
        assert [loss.origin_path for loss in losses] == [
            "$.messages[0].content[0]",
            "$.messages[1].content[0]",
        ]
        assert losses.items[0].origin_value is messages[0].content[0]
        assert losses.items[1].origin_value is messages[1].content[0]
        assert all("not support function calling" in loss.reason for loss in losses)
    assert messages == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_native_images_are_loss_when_vision_is_disabled(profile_factory, protocol):
    uri = "https://example.com/image.png"
    if protocol == "openai_chat":
        image = {"type": "image_url", "image_url": {"url": uri}}
        request = {
            "messages": [{"role": "user", "content": [image, {"type": "text", "text": "tail"}]}]
        }
    elif protocol == "openai_responses":
        image = {"type": "input_image", "image_url": uri, "detail": "auto"}
        request = {
            "input": [{"role": "user", "content": [image, {"type": "input_text", "text": "tail"}]}]
        }
    elif protocol == "anthropic":
        image = {"type": "image", "source": {"type": "url", "url": uri}}
        request = {
            "max_tokens": 128,
            "messages": [{"role": "user", "content": [image, {"type": "text", "text": "tail"}]}],
        }
    elif protocol == "gemini":
        image = {"inline_data": {"mime_type": "image/png", "data": "aW1hZ2U="}}
        request = {"contents": [{"role": "user", "parts": [image, {"text": "tail"}]}]}
    else:
        image = "aW1hZ2U="
        request = {"messages": [{"role": "user", "content": "tail", "images": [image]}]}
    before = deepcopy(request)
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().decoder(protocol).native_request(
            request, profile_factory(protocol, supports_vision=config(False))
        )
    assert len(caught.value.losses) == 1
    assert caught.value.losses.items[0].origin_value is image
    assert "not support image" in caught.value.losses.items[0].reason
    assert "tail" in str(caught.value.draft)
    assert "image.png" not in str(caught.value.draft)
    assert "aW1hZ2U=" not in str(caught.value.draft)
    assert request == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_native_history_tools_are_loss_when_function_calling_is_disabled(profile_factory, protocol):
    if protocol in ("openai_chat", "ollama"):
        call = (
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            if protocol == "openai_chat"
            else {"function": {"name": "f", "arguments": {}}}
        )
        result = {"role": "tool", "tool_call_id": "c", "tool_name": "f", "content": "result"}
        request = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [call]},
                result,
                {"role": "user", "content": "tail"},
            ]
        }
    elif protocol == "anthropic":
        call = {"type": "tool_use", "id": "c", "name": "f", "input": {}}
        result = {"type": "tool_result", "tool_use_id": "c", "content": "result"}
        request = {
            "max_tokens": 128,
            "messages": [
                {"role": "assistant", "content": [call]},
                {"role": "user", "content": [result]},
                {"role": "user", "content": "tail"},
            ],
        }
    elif protocol == "gemini":
        call = {"function_call": {"id": "c", "name": "f", "args": {}}}
        result = {"function_response": {"id": "c", "name": "f", "response": {"result": "result"}}}
        request = {
            "contents": [
                {"role": "model", "parts": [call]},
                {"role": "user", "parts": [result]},
                {"role": "user", "parts": [{"text": "tail"}]},
            ]
        }
    else:
        call = {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}
        result = {"type": "function_call_output", "call_id": "c", "output": "result"}
        request = {"input": [call, result, {"role": "user", "content": "tail"}]}
    before = deepcopy(request)
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().decoder(protocol).native_request(
            request, profile_factory(protocol, supports_function_calling=config(False))
        )
    assert len(caught.value.losses) == 2
    assert caught.value.losses.items[0].origin_value is call
    assert caught.value.losses.items[1].origin_value is result
    assert all("not support function calling" in loss.reason for loss in caught.value.losses)
    assert "tail" in str(caught.value.draft)
    assert "result" not in str(caught.value.draft)
    assert request == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_function_call_output_is_handled_by_output_hook(profile_factory, protocol):
    call = ToolCall("c", "f", {"x": 1})
    messages = [AssistantMessage("assistant", [call, TextContent("answer")])]
    observed = []
    result = MyCodec("hook", output_semantic_loss_hook=lambda *args: observed.append(args)).result(
        messages,
        input_format=protocol,
        target_format=protocol,
        semantic_profile=profile_factory(protocol, supports_function_calling=config(False)),
    )
    assert "answer" in str(result)
    assert len(observed) == 1
    assert len(observed[0][1]) == 1
    assert observed[0][1].items[0].origin_value == {
        "type": "tool_use",
        "id": "c",
        "name": "f",
        "input": {"x": 1},
    }
    assert "not support function calling" in observed[0][1].items[0].reason
    assert observed[0][2].origin_source == "output"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_tool_stream_event_is_loss_with_complete_text_draft(profile_factory, protocol):
    events = [
        {"type": "tool", "block": {"type": "tool_use", "id": "c", "name": "f", "input": {"x": 1}}},
        {"type": "delta", "field": "text", "value": "answer"},
    ]
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().decoder(protocol).chunks(
            events,
            source_protocol=protocol,
            semantic_profile=profile_factory(protocol, supports_function_calling=config(False)),
        )
    assert len(caught.value.losses) == 1
    assert "not support function calling" in caught.value.losses.items[0].reason
    assert "answer" in str(caught.value.draft)
