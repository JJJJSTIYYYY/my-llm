import copy

import pytest

from my_llm import (
    ConfigurationError,
    ModelProfile,
    ProviderProfile,
    UnsupportedCapabilityError,
    omit,
)
from my_llm.profile.resolution import resolve_profiles
from tests.helpers.profiles import config, parameter
from tests.helpers.requests import codec, encode_request


@pytest.mark.parametrize(
    "protocol,token_key",
    [("openai_responses", "max_output_tokens"), ("openai_chat", "max_completion_tokens")],
)
def test_tokens_translate_and_upstream_always_streams(profile_factory, protocol, token_key):
    instance = codec(
        profile_factory(protocol, max_tokens=parameter("max_tokens", 100, protocol=protocol))
    )
    request = instance.encode([{"role": "user", "content": "hello"}], tools=omit)
    assert request[token_key] == 100
    assert request["stream"] is True
    assert "tools" not in request
    assert token_key not in instance.encode(
        [{"role": "user", "content": "hello"}], tools=omit, max_tokens=omit
    )
    assert (
        instance.encode([{"role": "user", "content": "hello"}], tools=omit, max_tokens=50)[
            token_key
        ]
        == 50
    )


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
def test_no_reasoning_parameters_by_default(profile_factory, protocol):
    request = codec(profile_factory(protocol)).encode(
        [{"role": "user", "content": "x"}], tools=omit
    )
    assert (
        "reasoning" not in request
        and "reasoning_effort" not in request
        and "extra_body" not in request
    )


def test_reasoning_enable_only_for_chat(profile_factory):
    request = codec(profile_factory()).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="high"
    )
    assert request["reasoning_effort"] == "medium"
    assert "extra_body" not in request


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_response_reasoning_mapping(profile_factory, effort):
    request = codec(profile_factory("openai_responses")).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort=effort
    )
    assert request["reasoning"] == {"summary": "detailed", "effort": effort}


def test_unsupported_reasoning_and_effort(profile_factory):
    with pytest.raises(UnsupportedCapabilityError, match="reasoning"):
        codec(profile_factory(supports_reasoning=config("supports_reasoning", False))).encode(
            [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="low"
        )
    limited = codec(
        profile_factory(
            "openai_responses",
            reasoning_effort_mapping=config(
                "reasoning_effort_mapping", {"xhigh": None, "max": None}
            ),
        )
    )
    with pytest.raises(UnsupportedCapabilityError, match="xhigh"):
        limited.encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="xhigh")


def test_custom_effort_body_and_generated_priority():
    provider = ProviderProfile(
        supports_reasoning_effort=config("supports_reasoning_effort", True),
        reasoning_enable=parameter(
            "reasoning_enable", True, protocol="openai_chat", path="$.extra_body.enable_thinking"
        ),
        reasoning_effort=parameter(
            "reasoning_effort", protocol="openai_chat", path="$.extra_body.thinking.budget"
        ),
        reasoning_effort_mapping=config("reasoning_effort_mapping", {"high": "large"}),
        extra_headers=config("extra_headers", {"a": "provider", "b": "provider"}),
        extra_body=config("extra_body", {"source": "provider"}),
    )
    model = ModelProfile(
        model="demo",
        extra_headers=config("extra_headers", {"a": "model"}),
        extra_body=config("extra_body", {"source": "model"}),
    )
    request = codec(resolve_profiles(provider, model)).encode(
        [{"role": "user", "content": "x"}],
        tools=omit,
        reasoning_effort="high",
        extra_headers={"a": "request", "b": omit},
        extra_body={"source": "request", "enable_thinking": False, "thinking": {"budget": "small"}},
    )
    assert request["extra_headers"] == {"a": "request"}
    assert request["extra_body"] == {
        "source": "request",
        "enable_thinking": False,
        "thinking": {"budget": "large"},
    }
    assert "reasoning_effort" not in request


def test_generated_named_parameters_win_over_body(profile_factory):
    request = codec(profile_factory("openai_responses")).encode(
        [{"role": "user", "content": "x"}],
        tools=omit,
        reasoning_effort="high",
        max_tokens=10,
        extra_body={"reasoning": {"effort": "low"}, "max_output_tokens": 1},
    )
    assert request["extra_body"]["reasoning"] == request["reasoning"]
    assert request["extra_body"]["max_output_tokens"] == 10


@pytest.mark.parametrize(
    "parameter", ["model", "stream", "input", "messages", "tools", "previous_response_id"]
)
def test_body_cannot_override_protocol_invariants(profile_factory, parameter):
    with pytest.raises(ConfigurationError):
        codec(profile_factory()).encode(
            [{"role": "user", "content": "x"}], tools=omit, extra_body={parameter: False}
        )


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
@pytest.mark.parametrize("input_format", ["openai_responses", "openai_chat"])
def test_tools_conversion_and_no_input_mutation(
    profile_factory, function_tool, protocol, input_format
):
    tools = [
        {"type": "function", **function_tool["function"]}
        if input_format == "openai_responses"
        else function_tool
    ]
    before = copy.deepcopy(tools)
    request = codec(
        profile_factory(
            protocol, supports_function_calling=config("supports_function_calling", True)
        )
    ).encode([{"role": "user", "content": "x"}], tools=tools, tools_format=input_format)
    assert request["tools"] == (
        [{"type": "function", **function_tool["function"]}]
        if protocol == "openai_responses"
        else [function_tool]
    )
    request["tools"][0]["type"] = "changed"
    assert tools == before


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
@pytest.mark.parametrize("input_format", ["openai_responses", "openai_chat"])
def test_tool_extensions_are_filtered_and_foreign_null_options_are_omitted(
    profile_factory, protocol, input_format
):
    function = {
        "name": "weather",
        "description": None,
        "parameters": None,
        "strict": None,
        "x_provider": {"flags": ["custom"]},
    }
    tool = (
        {"type": "function", **function}
        if input_format == "openai_responses"
        else {"type": "function", "function": function}
    )
    request = codec(
        profile_factory(
            protocol, supports_function_calling=config("supports_function_calling", True)
        )
    ).encode([{"role": "user", "content": "x"}], tools=[tool], tools_format=input_format)
    if protocol == "openai_responses":
        assert request["tools"] == [
            {"type": "function", "name": "weather", "parameters": None, "strict": False}
        ]
    else:
        assert request["tools"] == [{"type": "function", "function": {"name": "weather"}}]


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
def test_empty_tool_list_passes_through(profile_factory, protocol):
    request = codec(profile_factory(protocol)).encode([{"role": "user", "content": "x"}], tools=[])
    assert request["tools"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "invalid"},
        {"reasoning": False, "reasoning_effort": "high"},
        {"max_tokens": -1},
    ],
)
def test_invalid_request_parameters(profile_factory, kwargs):
    with pytest.raises(ConfigurationError):
        codec(profile_factory()).encode([{"role": "user", "content": "x"}], tools=omit, **kwargs)


@pytest.mark.parametrize(
    "protocol,key", [("openai_chat", "reasoning_content"), ("ollama", "thinking")]
)
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("replay", [True, False])
@pytest.mark.parametrize("has_calls", [True, False])
@pytest.mark.parametrize("visible", [None, "reason"])
def test_reasoning_replay_only_pads_tool_calls_when_thinking(
    protocol, key, thinking, replay, has_calls, visible, tool_call
):
    from my_llm import MyCodec

    message = {"role": "assistant", "content": "hello"}
    if has_calls:
        message["tool_calls"] = [tool_call]
    if visible:
        message["reasoning_content"] = visible
    profile = resolve_profiles(
        ProviderProfile(
            protocol=protocol,
            replay_reasoning_content_when_tool_calls=config(
                "replay_reasoning_content_when_tool_calls", replay
            ),
        ),
        ModelProfile(model="demo"),
    )
    result = MyCodec().convert(
        [message],
        input_format="openai_chat",
        target_format=protocol,
        profile=profile,
        reasoning=thinking,
    )["messages"][0]
    result = MyCodec().decoder(protocol).filter_messages([result], profile, reasoning=thinking)[0]
    assert (key in result) is (replay and (bool(visible) or (has_calls and thinking)))
    if key in result:
        assert result[key] == (visible or "")


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_native_generation_parameters_and_tools(profile_factory, protocol, function_tool):
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        supports_function_calling=config("supports_function_calling", True),
    )
    request = encode_request(profile, tools=[function_tool], reasoning=True)
    if protocol == "anthropic":
        assert request["max_tokens"] == 2048
        assert request["thinking"]["budget_tokens"] == 1024
        assert request["tools"][0]["input_schema"]["type"] == "object"
    elif protocol == "gemini":
        assert request["config"]["max_output_tokens"] == 2048
        assert request["config"]["automatic_function_calling"]["disable"]
        assert request["config"]["thinking_config"]["include_thoughts"]
    else:
        assert request["options"]["num_predict"] == 2048
        assert request["think"] is True


@pytest.mark.parametrize("protocol", ["openai_chat", "anthropic", "gemini", "ollama"])
def test_previous_response_id_is_responses_only(profile_factory, protocol):
    with pytest.raises(ConfigurationError):
        encode_request(
            profile_factory(protocol, max_tokens=parameter("max_tokens", 2048, protocol=protocol)),
            previous_response_id="resp_previous",
        )
