"""Merged scans preserve validation boundaries and reject post-translation changes."""

from copy import deepcopy
from dataclasses import replace

import pytest

from my_llm import (
    ConfigurationError,
    FunctionTool,
    MessageValidationError,
    MyLLM,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    UserMessage,
)
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.shared import input as input_module
from my_llm.codec.shared.conversion import validate_history
from my_llm.middleware.protocol import generation_input
from my_llm.middleware.validation import HistoryValidator, validate_messages
from tests.helpers.protocols import PROTOCOLS, history
from tests.performance.workloads import profiles, user_message


class SinglePass(list):
    """Fail if a merged check attempts to iterate the same message/block batch twice."""

    def __iter__(self):
        if getattr(self, "visited", False):
            raise AssertionError("batch traversed more than once")
        self.visited = True
        return super().__iter__()


def test_structure_and_tool_associations_share_the_message_and_block_scan():
    values = SinglePass(
        [
            {
                "role": "assistant",
                "content": SinglePass(
                    [{"type": "tool_use", "id": "call", "name": "f", "input": {}}]
                ),
            },
            {
                "role": "user",
                "content": SinglePass(
                    [{"type": "tool_result", "tool_use_id": "call", "content": "ok"}]
                ),
            },
        ]
    )
    validator = HistoryValidator()
    validate_messages(values, history=validator)
    validator.finish()


def test_typed_content_and_tools_are_validated_and_located_in_one_scan():
    parts = SinglePass([TextContent("first"), TextContent("second")])
    tools = SinglePass([FunctionTool("first"), FunctionTool("second")])
    message = UserMessage("user", parts)
    result = generation_input([message], tools)
    assert result.messages[0].content[1].origin.path == "$.messages[0].content[1]"
    assert result.tools[1].origin.path == "$.tools[1]"
    assert message.origin is None and parts[0].origin is None and tools[0].origin is None


def test_standalone_association_validation_retains_first_error():
    with pytest.raises(MessageValidationError, match="invalid message role"):
        validate_history([{"role": "invalid", "content": []}, None])


def test_later_structural_error_precedes_an_earlier_association_error():
    values = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "unknown", "content": []}],
        },
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "missing mark"}]},
    ]
    with pytest.raises(MessageValidationError, match=r"messages\[1\].content\[0\].mark"):
        validate_messages(values, history=HistoryValidator())


@pytest.mark.parametrize("allow_external", [False, True])
def test_combined_validation_checks_external_ids_and_duplicate_results(allow_external):
    values = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "server", "content": []}],
        }
    ]
    if allow_external:
        values.append(deepcopy(values[0]))
    validator = HistoryValidator(allow_external_calls=allow_external)
    validate_messages(values, history=validator)
    with pytest.raises(
        MessageValidationError, match="duplicate tool result" if allow_external else "unknown call"
    ):
        validator.finish()


def test_tool_definition_error_keeps_precedence_over_association_error():
    provider, model = profiles("openai_chat")
    llm = MyLLM(provider=provider, model=model, stdio_protocol="middleware")
    with pytest.raises(MessageValidationError, match="functions"):
        llm.codec.decoder("openai_chat").request(
            [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "unknown", "content": []}],
                }
            ],
            [{}],
            llm.resolved_profile,
        )


def test_invalid_controls_keep_precedence_over_association_error():
    provider, model = profiles("openai_chat")
    llm = MyLLM(provider=provider, model=model, stdio_protocol="middleware")
    with pytest.raises(ConfigurationError, match="stream must be a bool"):
        llm.codec.decoder("openai_chat").request(
            [{"role": "invalid", "content": []}], None, llm.resolved_profile, stream="invalid"
        )


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("injected", ["tool_round", "image"])
def test_transformed_messages_keep_capability_checks_for_every_protocol(protocol, injected):
    provider, model = profiles(protocol)
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    decoder = llm.codec.decoder(protocol)
    if injected == "tool_round":
        additions = history(protocol)
        capability, reason = "supports_function_calling", "function calling"
    else:
        additions = [user_message(protocol, "injected", media=True)]
        capability, reason = "supports_vision", "image input"
    disabled = replace(
        llm.resolved_profile,
        values={**llm.resolved_profile.values, capability: False},
    )
    source = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    original = deepcopy(additions)
    with pytest.raises(SemanticLossError) as captured:
        decoder.request(
            source,
            None,
            disabled,
            message_transform=lambda messages: [*messages, *additions],
        )
    assert any(reason in loss.reason for loss in captured.value.losses)
    assert additions == original
    assert not llm._clients


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_protocols_share_filter_messages_and_keep_sdk_projection(protocol):
    provider, model = profiles(protocol)
    llm = MyLLM(provider=provider, model=model, stdio_protocol=protocol)
    decoder = llm.codec.decoder(protocol)
    assert type(decoder).filter_messages is BaseDecoder.filter_messages
    native = user_message(protocol, "hello", media=False)
    native["unknown_extension"] = {"ignored": True}
    original = deepcopy(native)
    filtered = decoder.filter_messages([native], llm.resolved_profile)
    assert "unknown_extension" not in filtered[0]
    assert native == original


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_filter_policy_reads_capabilities_once_for_the_entire_batch(monkeypatch, protocol):
    provider, model = profiles(protocol)
    llm = MyLLM(provider=provider, model=model, stdio_protocol=protocol)
    decoder = llm.codec.decoder(protocol)
    original = input_module.supports_content
    reads = []

    def supports(kind, profile):
        reads.append(kind)
        return original(kind, profile)

    monkeypatch.setattr(input_module, "supports_content", supports)
    decoder.filter_messages(
        [user_message(protocol, str(index), media=False) for index in range(5)],
        llm.resolved_profile,
    )
    assert reads == ["image", "thinking", "tool_use", "message_name"]


def test_public_request_revalidates_mutated_middleware_models():
    provider, model = profiles("openai_chat")
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    encoded = llm.codec.encode(
        [{"role": "assistant", "content": "hello", "reasoning_content": "why"}],
        input_format="openai_chat",
    )
    reasoning = next(
        part for part in encoded.messages[0].content if isinstance(part, ReasoningContent)
    )
    reasoning.mark = "invalid"
    with pytest.raises(MessageValidationError, match="reasoning mark"):
        llm.codec.request(encoded, llm.resolved_profile)


def test_cached_translation_is_filtered_again_without_readding_unsupported_content():
    from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder

    class ExtensionDecoder(ChatCompletionsDecoder):
        def _messages(self, messages, profile=None, **parameters):
            result = super()._messages(messages, profile, **parameters)
            for message in result:
                if message.get("content") == "initial":
                    message["content"] = [
                        {"type": "text", "text": "initial"},
                        {"type": "unsupported_extension"},
                    ]
            return result

    provider, model = profiles("openai_chat")
    llm = MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="openai_chat",
        semantic_loss_protocol="ignore",
        enable_message_cache=True,
    )
    llm.codec.codecs["openai_chat"].decoder = ExtensionDecoder()

    def prepare(text):
        return llm.codec.prepare(
            {"messages": [{"role": "user", "content": text}], "tools": None, "stream": True},
            llm.resolved_profile,
            input_format="openai_chat",
            client=llm.client_parameters,
            enable_message_cache=True,
            message_cache_key=llm._message_cache_key,
        )["request"]

    first = prepare("initial")
    expected = [{"type": "text", "text": "initial"}]
    assert first["messages"][0]["content"] == expected
    cached = llm.codec._message_cache.get(llm._message_cache_key)
    assert cached[0]["content"][1]["type"] == "unsupported_extension"
    second = prepare("next")
    assert second["messages"][0]["content"] == expected
    assert second == prepare("next")


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_public_filter_override_is_checked_after_returning(protocol):
    provider, model = profiles(protocol)
    llm = MyLLM(provider=provider, model=model, stdio_protocol=protocol)
    original_type = type(llm.codec.decoder(protocol))

    class InvalidFilter(original_type):
        def filter_messages(self, messages, profile, **parameters):
            result = super().filter_messages(messages, profile, **parameters)
            result[0]["role"] = {"invalid": True}
            return result

    decoder = InvalidFilter()
    message = user_message(protocol, "hello", media=False)
    request = {"model": model.model.default_value, decoder.message_path[0]: [message]}
    with pytest.raises(MessageValidationError, match="native protocol"):
        decoder.native_request(request, llm.resolved_profile)
