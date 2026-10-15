import copy

import pytest

from my_llm import (
    ConfigurationError,
    ModelProfile,
    ProviderProfile,
    UnsupportedCapabilityError,
    omit,
)
from my_llm.codec import ChatCompletionsCodec, OllamaCodec, ResponsesCodec
from my_llm.profile.utils.resolution import resolve_profiles
from my_llm.types.exceptions import MessageValidationError


def codec(profile):
    return (
        ResponsesCodec(profile)
        if profile.api_type == "responses"
        else ChatCompletionsCodec(profile)
    )


@pytest.mark.parametrize(
    "api_type,token_key",
    [("responses", "max_output_tokens"), ("chat_completions", "max_completion_tokens")],
)
def test_tokens_translate_and_upstream_always_streams(profile_factory, api_type, token_key):
    instance = codec(profile_factory(api_type, max_tokens=100))
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


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_no_reasoning_parameters_by_default(profile_factory, api_type):
    request = codec(profile_factory(api_type)).encode(
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
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in request


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_response_reasoning_mapping(profile_factory, effort):
    request = codec(profile_factory("responses")).encode(
        [{"role": "user", "content": "x"}], tools=omit, reasoning_effort=effort
    )
    assert request["reasoning"] == {"summary": "detailed", "effort": effort}


def test_unsupported_reasoning_and_effort(profile_factory):
    with pytest.raises(UnsupportedCapabilityError, match="reasoning"):
        codec(profile_factory(supports_reasoning=False)).encode(
            [{"role": "user", "content": "x"}], tools=omit, reasoning_effort="low"
        )
    limited = codec(
        profile_factory("responses", reasoning_effort_mapping={"xhigh": None, "max": None})
    )
    with pytest.raises(UnsupportedCapabilityError, match="xhigh"):
        limited.encode([{"role": "user", "content": "x"}], tools=omit, reasoning_effort="xhigh")


def test_custom_effort_body_and_generated_priority():
    provider = ProviderProfile(
        supports_reasoning_effort=True,
        reasoning_enable_parameter={"extra_body": {"enable_thinking": True}},
        reasoning_effort_parameter={
            "reasoning_effort": omit,
            "extra_body": {"thinking": {"budget": "{effort}"}},
        },
        reasoning_effort_mapping={"high": "large"},
        extra_headers={"a": "provider", "b": "provider"},
        extra_body={"source": "provider"},
    )
    model = ModelProfile(model="demo", extra_headers={"a": "model"}, extra_body={"source": "model"})
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
        "enable_thinking": True,
        "thinking": {"budget": "large"},
    }
    assert "reasoning_effort" not in request


def test_generated_named_parameters_win_over_body(profile_factory):
    request = codec(profile_factory("responses")).encode(
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


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
@pytest.mark.parametrize("input_format", ["responses", "chat_completions"])
def test_tools_conversion_and_no_input_mutation(
    profile_factory, function_tool, api_type, input_format
):
    tools = [
        {"type": "function", **function_tool["function"]}
        if input_format == "responses"
        else function_tool
    ]
    before = copy.deepcopy(tools)
    request = codec(profile_factory(api_type, supports_function_calling=True)).encode(
        [{"role": "user", "content": "x"}], tools=tools
    )
    assert request["tools"] == (
        [{"type": "function", **function_tool["function"]}]
        if api_type == "responses"
        else [function_tool]
    )
    request["tools"][0]["type"] = "changed"
    assert tools == before


@pytest.mark.parametrize("input_format", ["responses", "chat_completions"])
def test_tools_capability_and_cloud_tools_rejected(profile_factory, function_tool, input_format):
    tool = (
        {"type": "function", **function_tool["function"]}
        if input_format == "responses"
        else function_tool
    )
    with pytest.raises(UnsupportedCapabilityError):
        codec(profile_factory()).encode([{"role": "user", "content": "x"}], tools=[tool])
    with pytest.raises(MessageValidationError, match="local function"):
        codec(profile_factory(supports_function_calling=True)).encode(
            [{"role": "user", "content": "x"}], tools=[{"type": "web_search"}]
        )


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_mixed_tool_templates_are_converted_individually(profile_factory, function_tool, api_type):
    flat = {"type": "function", **copy.deepcopy(function_tool["function"])}
    flat["name"] = "second"
    tools = [function_tool, flat]
    original = copy.deepcopy(tools)
    request = codec(profile_factory(api_type, supports_function_calling=True)).encode(
        [{"role": "user", "content": "x"}], tools=tools
    )
    expected = (
        [{"type": "function", **function_tool["function"]}, flat]
        if api_type == "responses"
        else [
            function_tool,
            {
                "type": "function",
                "function": {key: value for key, value in flat.items() if key != "type"},
            },
        ]
    )
    assert request["tools"] == expected
    functions = (
        request["tools"]
        if api_type == "responses"
        else [tool["function"] for tool in request["tools"]]
    )
    for function in functions:
        function["parameters"]["properties"]["city"]["type"] = "changed"
    assert tools == original


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
@pytest.mark.parametrize("input_format", ["responses", "chat_completions"])
def test_tool_values_and_extension_fields_are_preserved(profile_factory, api_type, input_format):
    function = {
        "name": "weather",
        "description": None,
        "parameters": None,
        "strict": None,
        "x_provider": {"flags": ["custom"]},
    }
    tool = (
        {"type": "function", **function}
        if input_format == "responses"
        else {"type": "function", "function": function}
    )
    request = codec(profile_factory(api_type, supports_function_calling=True)).encode(
        [{"role": "user", "content": "x"}], tools=[tool]
    )
    assert request["tools"] == (
        [{"type": "function", **function}]
        if api_type == "responses"
        else [{"type": "function", "function": function}]
    )


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function"},
        {"type": "function", "name": ""},
        {"type": "function", "function": None},
        {"type": "function", "function": {"name": "nested"}, "name": "flat"},
        {"type": "function", "name": "weather", "parameters": []},
        {"type": "function", "name": "weather", "description": 1},
        {"type": "function", "name": "weather", "strict": "true"},
    ],
)
@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_invalid_or_ambiguous_tool_templates_are_rejected(profile_factory, tool, api_type):
    with pytest.raises(MessageValidationError):
        codec(profile_factory(api_type, supports_function_calling=True)).encode(
            [{"role": "user", "content": "x"}], tools=[tool]
        )


@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_empty_tool_list_passes_through(profile_factory, api_type):
    request = codec(profile_factory(api_type)).encode([{"role": "user", "content": "x"}], tools=[])
    assert request["tools"] == []


@pytest.mark.parametrize("replay", [True, False, omit])
def test_chat_reasoning_replay_and_message_names(profile_factory, tool_call, replay):
    history = [
        {
            "role": "assistant",
            "content": None,
            "name": "agent",
            "tool_calls": [tool_call],
            "reasoning_content": "reason",
            "finish_reason": "tool_calls",
            "usage": {},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    request = codec(
        profile_factory(
            supports_message_name=False, replay_reasoning_content_when_tool_calls=replay
        )
    ).encode(history, tools=omit)
    assert "name" not in request["messages"][0]
    assert ("reasoning_content" in request["messages"][0]) is (replay is True)
    assert "finish_reason" not in request["messages"][0]
    assert "usage" not in request["messages"][0]


@pytest.mark.parametrize("protocol", ["openai", "ollama"])
@pytest.mark.parametrize("replay", [None, True, False, omit])
@pytest.mark.parametrize("reasoning", [None, "", "reason"])
def test_reasoning_replay_fills_every_assistant_message(protocol, replay, reasoning, tool_call):
    history = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "checking"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    if reasoning is not None:
        for message in history:
            if message["role"] == "assistant":
                message["reasoning_content"] = reasoning
    before = copy.deepcopy(history)
    profile = resolve_profiles(
        ProviderProfile(protocol=protocol, replay_reasoning_content_when_tool_calls=replay),
        ModelProfile(model="demo"),
    )
    instance = ChatCompletionsCodec(profile) if protocol == "openai" else OllamaCodec(profile)
    request = instance.encode(history, tools=omit)
    key = "reasoning_content" if protocol == "openai" else "thinking"
    for message in request["messages"]:
        if message["role"] == "assistant" and (replay is None or replay is True):
            assert message[key] == (reasoning or "")
        else:
            assert key not in message
    assert history == before


def test_responses_history_native_replay(profile_factory, tool_call):
    native = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "encrypted"},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "weather",
            "arguments": tool_call["function"]["arguments"],
            "status": "completed",
        },
    ]
    history = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "tool_calls": [tool_call],
            "reasoning_content": "do not replay",
            "response_output": native,
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    request = codec(profile_factory("responses")).encode(history, tools=omit)
    assert request["input"][1:3] == native
    assert request["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "sunny",
    }
    assert "reasoning_content" not in str(request)


def test_responses_multimodal_conversion(profile_factory):
    request = codec(profile_factory("responses")).encode(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png", "detail": "low"},
                    },
                ],
            }
        ],
        tools=omit,
    )
    assert request["input"][0]["content"] == [
        {"type": "input_text", "text": "what?"},
        {"type": "input_image", "image_url": "https://example.test/image.png", "detail": "low"},
    ]


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
