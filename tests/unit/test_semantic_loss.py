"""End-to-end acceptance for the typed midpoint and complete-draft loss policy."""

from copy import deepcopy

import pytest

from my_llm import (
    AssistantMessage,
    BaseSemanticLossAdaptor,
    FunctionTool,
    ImageContent,
    MyCodec,
    ProtocolMiddleWare,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    ToolResult,
    UserMessage,
)
from my_llm.profile.types import omit


def picture_history():
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_a", "name": "look", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_a",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aW1hZ2U=",
                            },
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        },
        {"role": "user", "content": "complete tail"},
    ]


def complete_inputs(messages):
    return dict(
        messages=messages,
        tools=omit,
        stream=True,
        reasoning=None,
        reasoning_effort="none",
        extra_headers=None,
        extra_body=None,
        previous_response_id=None,
    )


def test_midpoint_is_concrete_and_models_have_independent_application_values():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"metadata": {"type": "string"}}},
                "strict": True,
            },
        }
    ]
    values = [
        {
            "role": "user",
            "metadata": {"drop": True},
            "content": [
                {"type": "text", "text": "text"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "input_audio", "input_audio": {"data": "ignored", "format": "wav"}},
                {"type": "file", "file": {"file_id": "ignored"}},
            ],
        },
        {"role": "assistant", "content": "answer", "reasoning_content": "reason"},
    ]
    encoded = MyCodec().encode(values, tools, input_format="openai_chat")
    assert isinstance(encoded, ProtocolMiddleWare)
    assert isinstance(encoded.messages[0], UserMessage)
    assert isinstance(encoded.messages[0].content[0], TextContent)
    assert isinstance(encoded.messages[0].content[1], ImageContent)
    assert isinstance(encoded.messages[1], AssistantMessage)
    assert any(isinstance(p, ReasoningContent) for p in encoded.messages[1].content)
    assert isinstance(encoded.tools[0], FunctionTool)
    assert "ignored" not in str(encoded.to_wire())
    tools[0]["function"]["parameters"]["properties"].clear()
    assert "metadata" in encoded.tools[0].parameters["properties"]


@pytest.mark.parametrize("target", ["openai_chat", "gemini", "ollama"])
def test_all_image_losses_are_collected_with_original_positions_and_complete_tail(target):
    values = picture_history()
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().convert(values, input_format="anthropic", target_format=target)
    error = caught.value
    assert len(error.losses.items) == 2
    assert error.losses.origin_source == "input"
    assert error.context.source_protocol == "anthropic"
    assert error.context.target_protocol == target
    assert [item.origin_path for item in error.losses] == [
        "$.messages[1].content[0].content[1]",
        "$.messages[1].content[0].content[2]",
    ]
    assert error.losses.items[0].origin_value == values[1]["content"][0]["content"][1]
    assert "complete tail" in str(error.draft)
    assert "beforeafter" in str(error.draft)
    assert "aW1hZ2U=" not in str(error.draft)


@pytest.mark.parametrize("target", ["openai_responses", "anthropic"])
def test_native_tool_images_remain_visible(target):
    converted = MyCodec().convert(picture_history(), input_format="anthropic", target_format=target)
    assert "https://example.com/a.png" in str(converted)
    assert "aW1hZ2U=" in str(converted)


def test_request_hook_can_move_images_in_place_on_the_final_sdk_request(profile_factory):
    saved = []

    def hook(draft, losses, context):
        saved.append(draft)
        assert context.origin_source == "input"
        assert context.stage == "request"
        assert "model" in draft
        draft["messages"].append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": losses.items[0].origin_value["source"]["url"]},
                    }
                ],
            }
        )
        return {"replacement": "ignored"}

    codec = MyCodec("hook", input_semantic_loss_hook=hook)
    profile = profile_factory("openai_chat")
    prepared = codec.prepare(
        complete_inputs(picture_history()),
        profile,
        input_format="anthropic",
        client=codec.client_parameters(profile),
    )
    assert saved == [prepared["request"]]
    assert saved[0] is prepared["request"]
    assert (
        prepared["request"]["messages"][-1]["content"][0]["image_url"]["url"]
        == "https://example.com/a.png"
    )


def test_output_hook_receives_the_same_final_result_and_separate_direction():
    seen = []

    def hook(draft, losses, context):
        seen.append(draft)
        assert losses.origin_source == context.origin_source == "output"
        assert context.source_protocol == "openai_chat"
        assert context.target_protocol == "anthropic"
        draft["content"].append({"type": "text", "text": "hook"})

    codec = MyCodec("hook", output_semantic_loss_hook=hook)
    encoded = codec.encode(
        [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}],
        input_format="openai_chat",
    )
    output = codec.result(encoded.messages, target_format="anthropic", input_format="openai_chat")
    assert output is seen[0]
    assert output["content"][-1]["text"] == "hook"


def test_ignore_and_empty_hook_continue_without_retaining_loss_history():
    values = picture_history()
    before = deepcopy(values)
    ignored = MyCodec("ignore").convert(
        values, input_format="anthropic", target_format="openai_chat"
    )
    empty_hook = MyCodec("hook").convert(
        values, input_format="anthropic", target_format="openai_chat"
    )
    assert ignored == empty_hook
    assert values == before
    assert not hasattr(MyCodec("ignore").semantic_loss_adaptor, "losses")


def test_hook_can_raise_and_reenter_an_independent_translation():
    def hook(draft, losses, context):
        nested = MyCodec().convert(
            [{"role": "user", "content": "nested"}],
            input_format="openai_chat",
            target_format="ollama",
        )
        assert nested["messages"][0]["content"] == "nested"
        raise LookupError("user decision")

    with pytest.raises(LookupError, match="user decision"):
        MyCodec("hook", input_semantic_loss_hook=hook).convert(
            picture_history(), input_format="anthropic", target_format="openai_chat"
        )


@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_adaptor_has_distinct_directional_hooks(policy):
    adaptor = BaseSemanticLossAdaptor(policy)
    assert callable(adaptor.input_semantic_loss_hook)
    assert callable(adaptor.output_semantic_loss_hook)


def test_typed_tool_result_contains_only_supported_parts():
    encoded = MyCodec().encode(picture_history(), input_format="anthropic")
    assert isinstance(encoded.messages[1].content[0], ToolResult)
    assert len(encoded.messages[1].content[0].content) == 4


@pytest.mark.parametrize("streaming", [False, True])
async def test_llm_delivers_the_exact_result_draft_modified_by_output_hook(monkeypatch, streaming):
    from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile
    from tests.helpers.protocols import chat_event
    from tests.helpers.transport import FakeClient

    client = FakeClient(
        [chat_event(content="answer", reasoning="reason"), chat_event(finish="stop")]
    )
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    final_drafts = []

    def hook(draft, losses, context):
        assert losses.origin_source == "output"
        if context.stage == "result":
            final_drafts.append(draft)
            draft["content"] += " resolved"

    async with MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value="openai_chat"), reasoning_content=omit
        ),
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
        semantic_loss_protocol="hook",
        output_semantic_loss_hook=hook,
    ) as llm:
        result = await llm.generate(
            messages=[{"role": "user", "content": "question"}], stream=streaming
        )
        if streaming:
            async with result:
                async for _ in result:
                    pass
            result = result.result
        assert final_drafts == [result]
        assert result is final_drafts[0]
        assert result["content"] == "answer resolved"
    assert client.closed and client.upstream.closed


def test_codec_handles_an_explicit_complete_error_from_custom_decoder():
    from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup, SemanticLossItem

    draft = [{"content": "complete"}]
    context = SemanticLossContext("custom", "openai_chat", "output")

    def decoder():
        raise SemanticLossError(
            draft, SemanticLossGroup([SemanticLossItem("value", "$.value")], "output"), context
        )

    result = MyCodec("ignore").codecs["openai_chat"].translate(decoder, context, draft_key="chunks")
    assert result["chunks"] is draft


def test_unused_protocol_factories_are_lazy_and_not_global():
    converter = MyCodec()
    assert converter.codecs.instances == {}
    chat = converter.codecs["openai_chat"]
    assert set(converter.codecs.instances) == {"openai_chat"}
    assert MyCodec().codecs["openai_chat"] is not chat
