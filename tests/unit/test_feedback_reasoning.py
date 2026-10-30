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
from tests.helpers.profiles import config
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
def test_feedback_collects_all_excluded_reasoning_in_complete_drafts(
    profile_factory, protocol, mode, policy
):
    messages = reasoning_history(protocol)
    before = deepcopy(messages)
    observed = []
    codec = MyCodec(policy, input_semantic_loss_hook=lambda *args: observed.append(args))
    profile = profile_factory(
        protocol, feedback_reasoning_content=omit if mode is omit else config(mode)
    )

    def prepare():
        return codec.prepare(
            {"messages": messages, "tools": omit, "stream": False, "max_tokens": 128},
            profile,
            input_format="middleware",
            client={},
        )["request"]

    losses = None
    if mode != "always" and policy == "raise":
        with pytest.raises(SemanticLossError) as caught:
            prepare()
        draft, losses = caught.value.draft, caught.value.losses
    else:
        draft = prepare()
        if mode != "always" and policy == "hook":
            assert observed[0][0] is draft
            losses = observed[0][1]
        else:
            assert not observed
    assert ("plain reasoning" in str(draft)) is (mode == "always")
    assert ("tool reasoning" in str(draft)) is (mode in ("always", "when_tool_calls"))
    assert "complete tail" in str(draft)
    if losses is not None:
        assert len(losses) == (1 if mode == "when_tool_calls" else 2)
        assert losses.items[0].origin_value is messages[0].content[0]
        assert losses.items[0].origin_path == "$.messages[0].content[0]"
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
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().prepare(
            {"messages": messages, "tools": omit, "stream": False, "max_tokens": 128},
            profile_factory(target, feedback_reasoning_content=config("when_tool_calls")),
            input_format="openai_responses",
            client={},
        )
    assert len(caught.value.losses) == 1
    assert caught.value.losses.items[0].origin_value is messages[0]
    assert "tool reasoning" in str(caught.value.draft)
    assert "plain reasoning" not in str(caught.value.draft)
    if target == "openai_responses":
        assert all(
            identity in str(caught.value.draft)
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
    profile = profile_factory(feedback_reasoning_content=config("when_tool_calls"))
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

        if policy == "raise":
            with pytest.raises(SemanticLossError):
                prepare()
        else:
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
    profile = profile_factory(protocol, feedback_reasoning_content=config(mode))
    if mode == "always" or mode == "when_tool_calls" and has_calls:
        assert decoder.native_request(request, profile)["messages"][0][field] == "native reasoning"
    else:
        with pytest.raises(SemanticLossError) as caught:
            decoder.native_request(request, profile)
        assert caught.value.losses.items[0].origin_value == "native reasoning"
        assert caught.value.losses.items[0].origin_path == f"$.messages[0].{field}"
        assert field not in caught.value.draft["messages"][0]
    assert request == before
