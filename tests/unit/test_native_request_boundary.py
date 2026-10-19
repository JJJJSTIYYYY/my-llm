"""Native transport boundaries validate and filter every request-bearing structure."""

from copy import deepcopy
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from my_llm import ConfigurationError, MessageValidationError, MyCodec, UnsupportedCapabilityError
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport


def test_open_model_unknown_fields_are_filtered_before_validation_and_repeated_requests(
    profile_factory,
):
    class Message(BaseModel):
        model_config = ConfigDict(extra="allow")
        role: Literal["user"]
        content: str
        __pydantic_extra__: dict[str, int]

    class Transport(ChatCompletionsTransport):
        __slots__ = ()
        message_type = Message

    bound = Transport()
    profile = profile_factory("openai_chat")
    for message in (
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second", "vendor_count": 2},
    ):
        original = deepcopy(message)
        bound.validate_input([message])
        request = bound.request({"messages": [message]}, profile)
        assert request["messages"] == [{"role": "user", "content": message["content"]}]
        assert message == original

    extra = {"role": "user", "content": "third", "vendor_count": {"invalid": True}}
    bound.validate_input([extra])
    assert bound.request({"messages": [extra]}, profile)["messages"] == [
        {"role": "user", "content": "third"}
    ]
    invalid = {"role": "user", "content": ["invalid"], "vendor_count": 1}
    with pytest.raises(MessageValidationError):
        bound.validate_input([invalid])
    with pytest.raises(MessageValidationError):
        bound.request({"messages": [invalid]}, profile)


@pytest.mark.parametrize("target", ["anthropic", "gemini"])
@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("source", ["openai_chat", "gemini"])
def test_system_unknown_fields_do_not_enter_intermediate_or_sdk_request(
    profile_factory, source, role, target
):
    system = (
        {
            "role": role,
            "parts": [{"text": "instructions", "vendor_block": {"nested": [1, None]}}],
        }
        if source == "gemini"
        else {
            "role": role,
            "content": [
                {"type": "text", "text": "instructions", "vendor_block": {"nested": [1, None]}}
            ],
        }
    )
    system["vendor_message"] = {"nested": [None, {"keep": True}]}
    user = (
        {"role": "user", "parts": [{"text": "hello"}]}
        if source == "gemini"
        else {"role": "user", "content": "hello"}
    )
    history = [system, user]
    original = deepcopy(history)
    codec = MyCodec()
    intermediate = codec.encode(history, input_format=source)
    assert "vendor_" not in str(intermediate)
    before_request = deepcopy(intermediate)
    request = codec.request(intermediate, profile_factory(target, max_tokens=2048))
    assert history == original
    assert intermediate == before_request
    assert "vendor_" not in str(request)
    if target == "anthropic":
        assert request["system"] == [{"type": "text", "text": "instructions"}]
    else:
        assert request["config"].system_instruction.parts[0].text == "instructions"


@pytest.mark.parametrize(
    "protocol,native,match",
    [
        (
            "anthropic",
            {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 4096}},
            "budget",
        ),
        (
            "anthropic",
            {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 256}},
            "budget",
        ),
        ("anthropic", {"output_config": {"effort": "unsupported"}}, "output_config"),
        ("openai_chat", {"max_completion_tokens": -1}, "max_tokens"),
        ("openai_chat", {"reasoning_effort": 5}, "reasoning_effort"),
        ("openai_chat", {"extra_body": {"n": 2}}, "candidate"),
        ("openai_responses", {"max_output_tokens": -1}, "max_tokens"),
        ("openai_responses", {"reasoning": {"effort": 5}}, "reasoning.effort"),
        ("openai_responses", {"reasoning": 5}, "reasoning"),
        ("gemini", {"config": {"max_output_tokens": -1}}, "max_tokens"),
        ("gemini", {"config": {"candidate_count": 2}}, "candidate"),
        (
            "gemini",
            {"config": {"thinking_config": {"include_thoughts": True, "thinking_budget": 0}}},
            "budget",
        ),
        ("ollama", {"think": {"invalid": True}}, "think"),
    ],
)
def test_direct_native_requests_validate_generation_settings(
    profile_factory, protocol, native, match
):
    messages = {
        "openai_chat": {"messages": [{"role": "user", "content": "hello"}]},
        "openai_responses": {"input": [{"role": "user", "content": "hello"}]},
        "anthropic": {"messages": [{"role": "user", "content": "hello"}]},
        "gemini": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        "ollama": {"messages": [{"role": "user", "content": "hello"}]},
    }[protocol]
    request = {"model": "demo", **messages, **native}
    original = deepcopy(request)
    with pytest.raises((ConfigurationError, UnsupportedCapabilityError), match=match):
        MyCodec().transport(protocol).request(request, profile_factory(protocol, max_tokens=2048))
    assert request == original


def test_anthropic_tool_json_schema_preserves_open_keywords(profile_factory):
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/item"}},
        "required": ["item"],
        "additionalProperties": False,
        "$defs": {"item": {"type": "string", "minLength": 1}},
        "minProperties": 1,
    }
    tool = {"name": "analyze", "input_schema": schema, "vendor_field": "filter this"}
    original = deepcopy(tool)
    request = (
        MyCodec()
        .transport("anthropic")
        .request(
            {"messages": [{"role": "user", "content": "hello"}], "tools": [tool]},
            profile_factory("anthropic", max_tokens=2048, supports_function_calling=True),
        )
    )
    assert request["tools"] == [{"name": "analyze", "input_schema": schema}]
    assert tool == original


@pytest.mark.parametrize("message", [None, "invalid", {"role": "user", "images": [None]}])
def test_ollama_invalid_messages_still_report_protocol_validation_errors(message):
    with pytest.raises(MessageValidationError):
        MyCodec().validate_input("ollama", [message])
