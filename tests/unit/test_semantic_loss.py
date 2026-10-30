"""End-to-end acceptance for the typed midpoint and complete-draft loss policy."""

from copy import deepcopy

import pytest

from my_llm import (
    AssistantMessage,
    BaseSemanticLossAdaptor,
    FunctionTool,
    ImageContent,
    ImageSource,
    MyCodec,
    ProtocolMiddleWare,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    ToolResult,
    UserMessage,
)
from my_llm.middleware.models import content_from_wire
from my_llm.middleware.protocol import generation_input
from my_llm.profile.types import omit
from tests.helpers.profiles import config
from tests.helpers.protocols import PROTOCOLS


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


@pytest.mark.parametrize("target", ["openai_responses", "anthropic", "gemini", "ollama"])
def test_message_name_loss_keeps_all_positions_and_complete_draft(target):
    messages = [
        {"role": "user", "name": "alice", "content": "first"},
        {"role": "user", "name": "bob", "content": "complete tail"},
    ]
    before = deepcopy(messages)
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().convert(messages, input_format="openai_chat", target_format=target)
    error = caught.value
    assert [item.origin_path for item in error.losses] == [
        "$.messages[0].name",
        "$.messages[1].name",
    ]
    assert [item.origin_value for item in error.losses] == ["alice", "bob"]
    assert "complete tail" in str(error.draft)
    assert "alice" not in str(error.draft)
    assert messages == before


@pytest.mark.parametrize(
    "target", ["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"]
)
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
def test_message_name_request_policies_use_complete_sdk_draft(profile_factory, target, policy):
    observed = []

    def hook(draft, losses, context):
        assert context.stage == "request"
        assert context.source_protocol == "openai_chat"
        assert context.target_protocol == target
        assert context.origin_source == losses.origin_source == "input"
        assert [(item.origin_path, item.origin_value) for item in losses] == [
            ("$.messages[0].name", "alice")
        ]
        assert draft["model"]
        assert "complete tail" in str(draft)
        draft["hook_handled_name"] = True
        observed.append(draft)

    codec = MyCodec(policy, input_semantic_loss_hook=hook)
    profile = profile_factory(target, supports_message_name=config(False))
    messages = [
        {"role": "user", "name": "alice", "content": "first"},
        {"role": "user", "content": "complete tail"},
    ]
    before = deepcopy(messages)

    def prepare():
        return codec.prepare(
            {**complete_inputs(messages), "max_tokens": 128},
            profile,
            input_format="openai_chat",
            client={},
        )

    if policy == "raise":
        with pytest.raises(SemanticLossError) as caught:
            prepare()
        assert caught.value.draft["model"]
        assert "complete tail" in str(caught.value.draft)
        assert len(caught.value.losses) == 1
    else:
        prepared = prepare()
        assert "alice" not in str(prepared["request"])
        if policy == "hook":
            assert observed[0] is prepared["request"]
            assert prepared["request"]["hook_handled_name"] is True
        else:
            assert observed == []
    assert messages == before


def test_chat_supported_message_name_is_preserved_without_loss():
    messages = [{"role": "user", "name": "alice", "content": "hi"}]
    assert (
        MyCodec().convert(messages, input_format="openai_chat", target_format="openai_chat")[
            "messages"
        ]
        == messages
    )


@pytest.mark.parametrize("target", ["openai_responses", "anthropic", "gemini", "ollama"])
def test_named_typed_output_uses_output_hook_with_final_result_identity(target):
    observed = []

    def output_hook(draft, losses, context):
        assert context.origin_source == losses.origin_source == "output"
        assert [(item.origin_path, item.origin_value) for item in losses] == [
            ("$.messages[0].name", "assistant_name")
        ]
        observed.append(draft)

    codec = MyCodec(
        "hook",
        input_semantic_loss_hook=lambda *args: pytest.fail("input hook called for output"),
        output_semantic_loss_hook=output_hook,
    )
    message = AssistantMessage("assistant", [TextContent("answer")], name="assistant_name")
    result = codec.result([message], input_format="openai_chat", target_format=target)
    assert result is observed[0]
    assert "answer" in str(result)
    assert message.name == "assistant_name"


def test_chat_disabled_name_reports_loss_in_offline_and_native_request_paths(profile_factory):
    codec = MyCodec()
    profile = profile_factory("openai_chat", supports_message_name=config(False))
    messages = [{"role": "user", "name": "alice", "content": "hi"}]
    middle = codec.encode(messages, input_format="openai_chat")
    with pytest.raises(SemanticLossError) as converted:
        codec.decode(**middle, target_format="openai_chat", profile=profile)
    assert converted.value.losses.items[0].origin_value == "alice"
    assert "name" not in converted.value.draft["messages"][0]
    with pytest.raises(SemanticLossError) as native:
        codec.decoder("openai_chat").native_request({"messages": messages}, profile)
    assert native.value.losses.items[0].origin_path == "$.messages[0].name"
    assert "name" not in native.value.draft["messages"][0]
    assert messages[0]["name"] == "alice"


@pytest.mark.parametrize("mark", ["raw", "summary", "encrypted"])
def test_reasoning_mark_survives_typed_input_and_wire_roundtrips(mark):
    part = ReasoningContent(reasoning="reason", mark=mark)
    middle = generation_input([AssistantMessage("assistant", [part])], None)
    wire = middle.to_wire()["messages"][0]["content"][0]
    assert wire["mark"] == mark
    restored = content_from_wire(wire)
    assert restored.mark == mark
    assert restored.reasoning == part.reasoning
    assert restored.to_wire() == wire


@pytest.mark.parametrize(
    "protocol,native,mark",
    [
        (
            "openai_chat",
            {"role": "assistant", "content": "", "reasoning_content": "raw"},
            "raw",
        ),
        (
            "openai_responses",
            {
                "type": "reasoning",
                "id": "reason-1",
                "summary": [{"type": "summary_text", "text": "summary"}],
            },
            "summary",
        ),
        (
            "openai_responses",
            {"type": "reasoning", "id": "reason-1", "summary": [], "encrypted_content": "opaque"},
            "encrypted",
        ),
        (
            "anthropic",
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]},
            "encrypted",
        ),
    ],
)
def test_native_encoders_assign_reasoning_mark_before_middleware(protocol, native, mark):
    messages = MyCodec().codecs[protocol].encoder.messages([native])
    reasoning = next(part for part in messages[0].content if isinstance(part, ReasoningContent))
    assert reasoning.mark == mark


def test_reasoning_mark_reaches_decoder_without_becoming_a_native_field(monkeypatch):
    codec = MyCodec()
    decoder = codec.decoder("openai_responses")
    original = type(decoder)._messages
    observed = []

    def decode(self, values, profile=None, **parameters):
        observed.append(values[0]["content"][0]["mark"])
        return original(self, values, profile, **parameters)

    monkeypatch.setattr(type(decoder), "_messages", decode)
    result = codec.decode(
        messages=[AssistantMessage("assistant", [ReasoningContent("summary", mark="summary")])],
        target_format="openai_responses",
    )
    assert observed == ["summary"]
    assert result["messages"][0]["summary"] == [{"type": "summary_text", "text": "summary"}]
    assert "mark" not in result["messages"][0]


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
    profile = profile_factory("openai_chat", supports_function_calling=config(True))
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


def image_history():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                {"type": "text", "text": "after"},
            ],
        },
        {"role": "user", "content": "complete tail"},
    ]


@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("supports_vision", [None, config(False), omit])
@pytest.mark.parametrize("stage", ["convert", "prepare"])
def test_unsupported_images_report_original_values_and_complete_draft(
    profile_factory, target, supports_vision, stage
):
    messages = image_history()
    before = deepcopy(messages)
    profile = profile_factory(target, supports_vision=supports_vision)
    codec = MyCodec()
    with pytest.raises(SemanticLossError) as caught:
        if stage == "convert":
            codec.convert(
                messages, input_format="openai_chat", target_format=target, profile=profile
            )
        else:
            codec.prepare(
                {**complete_inputs(messages), "max_tokens": 128},
                profile,
                input_format="openai_chat",
                client={},
            )
    error = caught.value
    assert error.context.source_protocol == "openai_chat"
    assert error.context.target_protocol == target
    assert error.context.origin_source == "input"
    assert [item.origin_path for item in error.losses] == [
        "$.messages[0].content[1]",
        "$.messages[0].content[2]",
    ]
    assert [item.origin_value for item in error.losses] == messages[0]["content"][1:3]
    assert all(item.reason == "model does not support image input" for item in error.losses)
    assert "complete tail" in str(error.draft)
    assert "before" in str(error.draft) and "after" in str(error.draft)
    assert "aW1hZ2U=" not in str(error.draft) and "a.png" not in str(error.draft)
    assert messages == before


@pytest.mark.parametrize("target", PROTOCOLS)
def test_enabled_vision_preserves_supported_images_in_requests(profile_factory, target):
    messages = image_history()
    del messages[0]["content"][1]  # Ollama supports base64 images.
    codec = MyCodec()
    prepared = codec.prepare(
        {**complete_inputs(messages), "max_tokens": 128},
        profile_factory(target, supports_vision=config(True)),
        input_format="openai_chat",
        client={},
    )
    assert "aW1hZ2U=" in str(prepared["request"])


@pytest.mark.parametrize("target", PROTOCOLS)
def test_images_in_tool_results_are_model_capability_losses(profile_factory, target):
    messages = picture_history()
    before = deepcopy(messages)
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().prepare(
            {**complete_inputs(messages), "max_tokens": 128},
            profile_factory(target, supports_function_calling=config(True)),
            input_format="anthropic",
            client={},
        )
    assert [item.origin_path for item in caught.value.losses] == [
        "$.messages[1].content[0].content[1]",
        "$.messages[1].content[0].content[2]",
    ]
    assert all("model does not support image input" == item.reason for item in caught.value.losses)
    assert "complete tail" in str(caught.value.draft)
    assert messages == before


@pytest.mark.parametrize("policy", ["ignore", "hook"])
def test_vision_loss_policies_and_cache_keep_the_complete_filtered_translation(
    profile_factory, policy
):
    observed = []

    def hook(draft, losses, context):
        assert context.stage == "request"
        assert len(losses) == 2
        assert "complete tail" in str(draft)
        draft["messages"].append({"role": "user", "content": "hook edit"})
        observed.append(draft)

    codec = MyCodec(policy, input_semantic_loss_hook=hook)
    profile = profile_factory()
    messages = image_history()
    for _ in range(2):
        prepared = codec.prepare(
            complete_inputs(messages),
            profile,
            input_format="openai_chat",
            client={},
            enable_message_cache=True,
            message_cache_key="vision-test",
        )
        assert "aW1hZ2U=" not in str(prepared)
        if policy == "hook":
            assert observed[-1] is prepared["request"]
    cached = codec._message_cache.get("vision-test")
    assert len(cached) == 2
    assert "hook edit" not in str(cached) and "aW1hZ2U=" not in str(cached)


def test_vision_loss_raise_commits_filtered_cache_before_the_adaptor(profile_factory):
    codec = MyCodec()
    for _ in range(2):
        with pytest.raises(SemanticLossError):
            codec.prepare(
                complete_inputs(image_history()),
                profile_factory(),
                input_format="openai_chat",
                client={},
                enable_message_cache=True,
                message_cache_key="vision-test",
            )
    cached = codec._message_cache.get("vision-test")
    assert len(cached) == 2
    assert "aW1hZ2U=" not in str(cached)


def test_typed_images_keep_the_original_object_as_loss_value(profile_factory):
    image = ImageContent(ImageSource("url", url="https://example.com/a.png"))
    messages = [UserMessage("user", [image, TextContent("tail")])]
    with pytest.raises(SemanticLossError) as caught:
        MyCodec().prepare(
            complete_inputs(messages), profile_factory(), input_format="middleware", client={}
        )
    assert caught.value.losses.items[0].origin_value is image
    assert caught.value.losses.items[0].origin_path == "$.messages[0].content[0]"
    assert messages[0].content[0] is image


def test_profile_free_conversion_continues_to_translate_images():
    result = MyCodec().convert(
        image_history(), input_format="openai_chat", target_format="openai_responses"
    )
    assert "aW1hZ2U=" in str(result)
