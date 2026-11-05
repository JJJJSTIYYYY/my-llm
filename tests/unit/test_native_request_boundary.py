"""Native transport boundaries validate and filter every request-bearing structure."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    MessageValidationError,
    MyCodec,
    SemanticLossError,
)
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.profile import ModelProfile, ProviderProfile, resolve_profiles
from my_llm.profile.types import InputParameter
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS, history


def test_open_model_unknown_fields_are_filtered_before_validation_and_repeated_requests(
    profile_factory,
):
    class Message(BaseModel):
        model_config = ConfigDict(extra="allow")
        role: Literal["user"]
        content: str
        __pydantic_extra__: dict[str, int]

    class Decoder(ChatCompletionsDecoder):
        __slots__ = ()
        message_type = Message

    bound = Decoder()
    profile = profile_factory("openai_chat")
    for message in (
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second", "vendor_count": 2},
    ):
        original = deepcopy(message)
        bound.validate_values([message], bound.message_type)
        request = bound.native_request({"messages": [message]}, profile)
        assert request["messages"] == [{"role": "user", "content": message["content"]}]
        assert message == original

    extra = {"role": "user", "content": "third", "vendor_count": {"invalid": True}}
    bound.validate_values([extra], bound.message_type)
    assert bound.native_request({"messages": [extra]}, profile)["messages"] == [
        {"role": "user", "content": "third"}
    ]
    invalid = {"role": "user", "content": ["invalid"], "vendor_count": 1}
    with pytest.raises(MessageValidationError):
        bound.validate_values([invalid], bound.message_type)
    with pytest.raises(MessageValidationError):
        bound.native_request({"messages": [invalid]}, profile)


@pytest.mark.parametrize("target", ["anthropic"])
@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("source", ["openai_chat"])
def test_system_unknown_fields_do_not_enter_intermediate_or_sdk_request(
    profile_factory, source, role, target
):
    system = {
        "role": role,
        "content": [
            {"type": "text", "text": "instructions", "vendor_block": {"nested": [1, None]}}
        ],
    }
    system["vendor_message"] = {"nested": [None, {"keep": True}]}
    user = {"role": "user", "content": "hello"}
    history = [system, user]
    original = deepcopy(history)
    codec = MyCodec()
    intermediate = codec.encode(history, input_format=source)
    assert "vendor_" not in str(intermediate)
    before_request = deepcopy(intermediate)
    request = codec.request(
        intermediate,
        profile_factory(target, max_tokens=parameter("max_tokens", 2048, protocol=target)),
    )
    assert history == original
    assert intermediate == before_request
    assert "vendor_" not in str(request)
    assert request["system"] == [{"type": "text", "text": "instructions"}]


@pytest.mark.parametrize(
    "protocol,native,match,expected",
    [
        (
            "anthropic",
            {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 4096}},
            "budget",
            ConfigurationError,
        ),
        (
            "anthropic",
            {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 256}},
            "budget",
            ConfigurationError,
        ),
        (
            "anthropic",
            {"output_config": {"effort": "unsupported"}},
            "output_config",
            SemanticLossError,
        ),
        ("openai_chat", {"max_completion_tokens": -1}, "max_tokens", ConfigurationError),
        ("openai_chat", {"reasoning_effort": 5}, "reasoning_effort", SemanticLossError),
        ("openai_chat", {"extra_body": {"n": 2}}, "candidate", SemanticLossError),
        ("openai_responses", {"max_output_tokens": -1}, "max_tokens", ConfigurationError),
        ("openai_responses", {"reasoning": {"effort": 5}}, "reasoning", ConfigurationError),
        ("openai_responses", {"reasoning": 5}, "reasoning", SemanticLossError),
    ],
)
def test_direct_native_requests_validate_generation_settings(
    profile_factory, protocol, native, match, expected
):
    messages = {
        "openai_chat": {"messages": [{"role": "user", "content": "hello"}]},
        "openai_responses": {"input": [{"role": "user", "content": "hello"}]},
        "anthropic": {"messages": [{"role": "user", "content": "hello"}]},
    }[protocol]
    request = {"model": "demo", **messages, **native}
    if protocol == "anthropic":
        request.setdefault("max_tokens", 2048)
    original = deepcopy(request)
    with pytest.raises(expected, match=match):
        MyCodec().decoder(protocol).native_request(
            request,
            profile_factory(protocol, max_tokens=parameter("max_tokens", 2048, protocol=protocol)),
        )
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
        .decoder("anthropic")
        .native_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [tool],
                "max_tokens": 2048,
            },
            profile_factory(
                "anthropic",
                max_tokens=parameter("max_tokens", 2048, protocol="anthropic"),
                supports_function_calling=config(True),
            ),
        )
    )
    assert request["tools"] == [{"name": "analyze", "input_schema": schema}]
    assert tool == original


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_request_projection_addresses_sdk_fields_without_reading_extensions(
    profile_factory, protocol
):
    class AddressedDict(dict):
        def __iter__(self):
            pytest.fail("decoder must address declared SDK fields")

        def items(self):
            pytest.fail("decoder must not enumerate unknown request fields")

        def keys(self):
            pytest.fail("decoder must not enumerate unknown request fields")

    class Unreadable:
        def __deepcopy__(self, memo):
            pytest.fail("unknown fields must not be read or copied")

    bound = MyCodec().decoder(protocol)
    messages = [AddressedDict(message, vendor=Unreadable()) for message in history(protocol)]
    request = AddressedDict(model="demo", stream=True, vendor=Unreadable())
    request[bound.message_path[0]] = messages
    if protocol == "anthropic":
        request["max_tokens"] = 2048
    else:
        request["temperature"] = None

    filtered = bound.native_request(
        request,
        profile_factory(
            protocol,
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
            supports_function_calling=config(True),
            supports_vision=config(True),
        ),
    )

    assert type(filtered) is dict
    assert "vendor" not in filtered
    assert all("vendor" not in message for message in filtered[bound.message_path[0]])
    container = filtered
    assert type(container) is dict
    if protocol != "anthropic":
        assert container["temperature"] is None


def test_profile_extension_restores_only_its_declared_unknown_sdk_leaf():
    @dataclass(kw_only=True)
    class ExtendedModel(ModelProfile):
        vendor_limit: InputParameter[int | None] = field(
            default_factory=lambda: InputParameter(
                path="$.reasoning.vendor_limit", default_value=None
            ),
        )

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_responses")),
        ExtendedModel(model=Attribute(default_value="demo")),
    )
    request = (
        MyCodec()
        .decoder("openai_responses")
        .native_request(
            {
                "model": "demo",
                "input": [{"role": "user", "content": "hello"}],
                "reasoning": {"effort": "low", "vendor_limit": None, "undeclared": True},
            },
            profile,
        )
    )

    assert request["reasoning"] == {"effort": "low", "vendor_limit": None}


@pytest.mark.parametrize("path", ["$.thinking", "$.extra_body.thinking"])
def test_anthropic_adaptive_profile_uses_actual_translated_thinking(path):
    profile = resolve_profiles(
        ProviderProfile(
            protocol=Config(default_value="anthropic"),
            reasoning_enable=parameter(
                "reasoning_enable", {"type": "adaptive"}, protocol="anthropic", path=path
            ),
        ),
        ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 512, protocol="anthropic"),
        ),
    )
    codec = MyCodec()
    intermediate = codec.encode([{"role": "user", "content": "hello"}], input_format="openai_chat")

    request = codec.request(intermediate, profile, reasoning=True)

    thinking = request["thinking"] if path == "$.thinking" else request["extra_body"]["thinking"]
    assert thinking == {"type": "adaptive"}
    assert request["max_tokens"] == 512


def test_nested_request_history_and_tools_own_their_parent_containers(profile_factory):
    class Envelope(TypedDict):
        messages: list[dict]
        tools: list[dict]

    class Decoder(ChatCompletionsDecoder):
        __slots__ = ()
        message_path = ("extra_body", "messages")
        tool_path = ("extra_body", "tools")
        request_schema = {**ChatCompletionsDecoder.request_schema, "extra_body": Envelope}

    request = {
        "extra_body": {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }
    }
    original = deepcopy(request)

    result = Decoder().native_request(request, profile_factory("openai_chat"))
    result["extra_body"]["messages"][0]["content"] = "changed"
    result["extra_body"]["tools"].append({"changed": True})

    assert request == original


def test_nested_request_paths_delegate_to_override_hooks_once_and_capture_payloads_once(
    profile_factory,
):
    class Envelope(TypedDict):
        messages: list[dict]
        tools: list[dict]
        metadata: dict

    captures = []
    hooks = []

    class Captured(dict):
        def __deepcopy__(self, memo):
            captures.append(self)
            return deepcopy(dict(self), memo)

    class Decoder(ChatCompletionsDecoder):
        __slots__ = ()
        message_path = ("extra_body", "messages")
        tool_path = ("extra_body", "tools")
        request_schema = {**ChatCompletionsDecoder.request_schema, "extra_body": Envelope}

        def filter_messages(self, messages, profile, **parameters):
            hooks.append(("messages", messages))
            return super().filter_messages(messages, profile, **parameters)

        def filter_tools(self, tools, profile):
            hooks.append(("tools", tools))
            return super().filter_tools(tools, profile)

    schema = Captured(type="object", properties={"name": {"type": "string"}})
    metadata = Captured(application={"nested": [1]})
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]
    request = {"extra_body": {"messages": messages, "tools": tools, "metadata": metadata}}

    result = Decoder().native_request(
        request,
        profile_factory("openai_chat", supports_function_calling=config(True)),
    )

    assert len(hooks) == 2
    assert hooks[0][0] == "messages" and hooks[0][1] is messages
    assert hooks[1][0] == "tools" and hooks[1][1] is tools
    assert len(captures) == 2
    assert sum(item is schema for item in captures) == 1
    assert sum(item is metadata for item in captures) == 1
    result["extra_body"]["messages"][0]["content"] = "changed"
    result["extra_body"]["tools"][0]["function"]["parameters"]["properties"]["name"] = {}
    result["extra_body"]["metadata"]["application"]["nested"].append(2)
    assert messages[0]["content"] == "hello"
    assert schema["properties"]["name"] == {"type": "string"}
    assert metadata["application"]["nested"] == [1]


def test_direct_chat_request_preserves_explicit_none_extra_body(profile_factory):
    request = (
        MyCodec()
        .decoder("openai_chat")
        .native_request(
            {"messages": [{"role": "user", "content": "hello"}], "extra_body": None},
            profile_factory("openai_chat"),
        )
    )

    assert "extra_body" in request
    assert request["extra_body"] is None


def test_declared_unknown_sdk_root_reports_configuration_error():
    @dataclass(kw_only=True)
    class ExtendedModel(ModelProfile):
        vendor: InputParameter[int] = field(
            default_factory=lambda: InputParameter(path="$.vendor", default_value=1),
        )

    profile = resolve_profiles(
        ProviderProfile(protocol=Config(default_value="openai_chat")),
        ExtendedModel(model=Attribute(default_value="demo")),
    )
    bound = MyCodec().decoder("openai_chat")

    for validate in (bound.validate_profile,):
        with pytest.raises(ConfigurationError, match="vendor.*SDK parameter 'vendor'"):
            validate(profile)
