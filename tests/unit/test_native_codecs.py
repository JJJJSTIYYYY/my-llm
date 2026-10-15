"""Request contracts, capability failures and malformed native streams."""

import json
from copy import deepcopy
from dataclasses import fields

import pytest
from google.genai import types

from my_llm import (
    ConfigurationError,
    IncompleteStreamError,
    MessageValidationError,
    ModelProfile,
    ProviderProfile,
    ResponseFailedError,
    UnsupportedCapabilityError,
    omit,
)
from my_llm.codec import AnthropicCodec, GeminiCodec, OllamaCodec
from my_llm.profile.utils.resolution import resolve_profiles

CODECS = {"anthropic": AnthropicCodec, "gemini": GeminiCodec, "ollama": OllamaCodec}
HISTORY = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


def codec(protocol, **model):
    return CODECS[protocol](
        resolve_profiles(
            ProviderProfile(protocol=protocol), ModelProfile(model="demo", max_tokens=4096, **model)
        )
    )


@pytest.mark.parametrize("protocol", CODECS)
@pytest.mark.parametrize("usage", ["none", "native", "stream_options"])
def test_protocol_inapplicable_fields_do_not_mutate_profiles(protocol, usage):
    body = {
        "ignored": {"nested": 1},
        "stream_options": {"include_usage": True},
        "reasoning": {"effort": "xhigh"},
    }
    provider = ProviderProfile(
        protocol=protocol,
        api_type="responses",
        usage_mode=usage,
        extra_body=body,
        extra_headers={"x-provider": "yes"},
    )
    model = ModelProfile(
        model="demo",
        max_tokens=4096,
        supports_message_name=True,
        extra_body={"ignored": omit},
        extra_headers={"x-model": "yes"},
    )
    profile = resolve_profiles(provider, model)
    before = deepcopy(profile.values)
    request = CODECS[protocol](profile).encode(HISTORY, tools=None, max_tokens=23)
    assert profile.values == before
    assert body["ignored"] == {"nested": 1}
    assert "reasoning" not in request and "stream_options" not in request
    assert not any(k in request for k in ("content_path", "api_type", "supports_message_name"))
    if protocol == "anthropic":
        assert request["max_tokens"] == 23 and isinstance(
            request["thinking"], type(__import__("anthropic").omit)
        )
        assert request["extra_headers"] == {"x-provider": "yes", "x-model": "yes"}
    elif protocol == "gemini":
        assert request["config"].max_output_tokens == 23
        assert request["config"].automatic_function_calling.disable is True
    else:
        assert request["options"]["num_predict"] == 23


@pytest.mark.parametrize("protocol", CODECS)
def test_native_token_omission_and_required_anthropic_limit(protocol):
    instance = codec(protocol)
    if protocol == "anthropic":
        with pytest.raises(ConfigurationError, match="requires max_tokens"):
            instance.encode(HISTORY, tools=None, max_tokens=omit)
    else:
        request = instance.encode(HISTORY, tools=None, max_tokens=omit)
        assert (
            ("options" not in request)
            if protocol == "ollama"
            else request["config"].max_output_tokens is None
        )


@pytest.mark.parametrize("protocol", CODECS)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning": False, "reasoning_effort": "high"},
        {"reasoning": "yes"},
        {"reasoning_effort": "unknown"},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"previous_response_id": "unsupported"},
        {"extra_body": {"tools": []}},
    ],
)
def test_native_invalid_generation_parameters(protocol, kwargs):
    with pytest.raises(ConfigurationError):
        codec(protocol).encode(HISTORY, tools=None, **kwargs)


@pytest.mark.parametrize("protocol", CODECS)
@pytest.mark.parametrize("capability", ["reasoning", "function_calling"])
@pytest.mark.parametrize("disabled", [False, omit])
def test_native_disabled_capabilities_are_errors(protocol, capability, disabled, function_tool):
    instance = codec(protocol, **{f"supports_{capability}": disabled})
    with pytest.raises(UnsupportedCapabilityError):
        instance.encode(
            HISTORY,
            tools=[function_tool] if capability == "function_calling" else None,
            reasoning=capability == "reasoning",
        )


@pytest.mark.parametrize("protocol", CODECS)
def test_native_effort_mapping_omission_and_unrepresentable_levels(protocol):
    instance = codec(
        protocol, supports_reasoning_effort=True, reasoning_effort_mapping={"high": omit}
    )
    with pytest.raises(UnsupportedCapabilityError):
        instance.encode(HISTORY, tools=None, reasoning_effort="high")
    with pytest.raises(UnsupportedCapabilityError, match="supported level"):
        codec(
            protocol,
            supports_reasoning_effort=True,
            reasoning_effort_mapping={"xhigh": "unrepresentable"},
        ).encode(HISTORY, tools=None, reasoning_effort="xhigh")


@pytest.mark.parametrize("protocol", CODECS)
def test_native_effort_bindings_and_mapping(protocol):
    binding = {
        "anthropic": {"extra_body": {"thinking": {"type": "adaptive"}}},
        "gemini": {"thinking_config": {"thinking_budget": 1234}},
        "ollama": {"think": True},
    }[protocol]
    instance = codec(
        protocol,
        supports_reasoning_effort=True,
        reasoning_enable_parameter=binding,
        reasoning_effort_mapping={"xhigh": "high"},
    )
    request = instance.encode(HISTORY, tools=None, reasoning_effort="xhigh")
    if protocol == "anthropic":
        assert request["thinking"]["type"] == "adaptive"
        assert request["output_config"]["effort"] == "high"
    elif protocol == "gemini":
        assert request["config"].thinking_config.thinking_level == types.ThinkingLevel.HIGH
        assert request["config"].thinking_config.thinking_budget is None
    else:
        assert request["think"] == "high"


@pytest.mark.parametrize("protocol", CODECS)
def test_native_local_function_templates_only(protocol, function_tool):
    instance = codec(protocol, supports_function_calling=True)
    flat = {"type": "function", **function_tool["function"]}
    a = instance.encode(HISTORY, tools=[function_tool])
    b = instance.encode(HISTORY, tools=[flat])
    if protocol == "gemini":
        assert a["config"].tools == b["config"].tools
    else:
        assert a["tools"] == b["tools"]
    with pytest.raises(MessageValidationError, match="local function"):
        instance.encode(HISTORY, tools=[{"type": "web_search"}])
    with pytest.raises(ConfigurationError, match="protocol parameters"):
        codec(protocol, reasoning_enable_parameter={"extra_body": {"tools": []}}).encode(
            HISTORY, tools=None, reasoning=True
        )


@pytest.mark.parametrize("protocol", CODECS)
def test_native_mixed_response_history_and_tools(protocol, tool_call):
    call = {"type": "function_call", "call_id": tool_call["id"], **tool_call["function"]}
    history = [
        *HISTORY,
        call,
        {"type": "function_call_output", "call_id": tool_call["id"], "output": "sunny"},
    ]
    request = codec(protocol).encode(history, tools=None)
    if protocol == "gemini":
        assert request["contents"][-1].parts[0].function_response.name == "weather"
    elif protocol == "anthropic":
        assert request["messages"][-1]["content"][0]["tool_use_id"] == tool_call["id"]
    else:
        assert request["messages"][-1]["tool_name"] == "weather"
    with pytest.raises(MessageValidationError, match="item_reference"):
        codec(protocol).encode([{"type": "item_reference", "id": "server-only"}], tools=None)
    with pytest.raises(MessageValidationError, match="preceding"):
        codec(protocol).encode(
            [{"role": "tool", "tool_call_id": "unknown", "content": "x"}], tools=None
        )
    with pytest.raises(MessageValidationError, match="duplicate"):
        codec(protocol).encode(
            [{"role": "assistant", "tool_calls": [tool_call, tool_call]}], tools=None
        )


@pytest.mark.parametrize("protocol", CODECS)
def test_native_empty_stream_and_duplicate_finish(protocol):
    decoder = codec(protocol).decoder()
    with pytest.raises(IncompleteStreamError):
        decoder.finish()
    decoder = codec(protocol).decoder()
    for event in {
        "anthropic": [
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ],
        "gemini": [{"candidates": [{"finish_reason": "STOP"}]}],
        "ollama": [{"done": True, "done_reason": "stop"}],
    }[protocol]:
        assert not decoder.feed(event)
    assert decoder.finish().is_finished
    with pytest.raises(IncompleteStreamError):
        decoder.finish()
    with pytest.raises(IncompleteStreamError):
        decoder.feed(event)


@pytest.mark.parametrize("protocol", CODECS)
def test_native_error_events(protocol):
    decoder = codec(protocol).decoder()
    with pytest.raises(ResponseFailedError):
        decoder.feed({"type": "error", "error": {"message": "bad"}})


def test_anthropic_interleaved_tools_and_json_failures():
    decoder = codec("anthropic").decoder()
    for index in (0, 1):
        decoder.feed(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": f"call_{index}",
                    "name": "weather",
                    "input": {},
                },
            }
        )
    for text in ('{"city":', '"上海"}'):
        for index in (1, 0):
            decoder.feed(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": text},
                }
            )
    for index in (0, 1):
        decoder.feed({"type": "content_block_stop", "index": index})
    decoder.feed({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})
    decoder.feed({"type": "message_stop"})
    result = decoder.finish()
    assert [c["id"] for c in result.tool_calls] == ["call_0", "call_1"]
    assert all(
        json.loads(c["function"]["arguments"]) == {"city": "上海"} for c in result.tool_calls
    )
    decoder = codec("anthropic").decoder()
    with pytest.raises(IncompleteStreamError, match="active content block"):
        decoder.feed(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x"},
            }
        )
    decoder.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "call_0", "name": "weather", "input": {}},
        }
    )
    decoder.feed(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{"},
        }
    )
    with pytest.raises(IncompleteStreamError, match="invalid JSON"):
        decoder.feed({"type": "content_block_stop", "index": 0})


@pytest.mark.parametrize("protocol", ["gemini", "ollama"])
def test_missing_native_ids_are_unique_across_turns(protocol):
    event = {
        "gemini": {
            "candidates": [
                {
                    "content": {"parts": [{"function_call": {"name": "f", "args": {}}}]},
                    "finish_reason": "STOP",
                }
            ]
        },
        "ollama": {
            "message": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            "done": True,
        },
    }[protocol]
    ids = []
    for _ in range(2):
        decoder = codec(protocol).decoder()
        decoder.feed(event)
        ids.append(decoder.finish().tool_calls[0]["id"])
    assert ids[0] != ids[1]


@pytest.mark.parametrize(
    "event",
    [
        {"prompt_feedback": {"block_reason": "SAFETY"}},
        {"candidates": [{"finish_reason": "MALFORMED_FUNCTION_CALL"}]},
        {"candidates": [{"finish_reason": "SAFETY"}]},
    ],
)
def test_gemini_blocked_and_invalid_function_responses(event):
    with pytest.raises(ResponseFailedError):
        codec("gemini").decoder().feed(event)


def test_profile_dataclass_contract_unchanged():
    assert [
        (f.name, f.default) for f in fields(ProviderProfile) if f.name in ("protocol", "api_type")
    ] == [("api_type", "chat_completions"), ("protocol", "openai")]
    assert ModelProfile(model="demo").max_context_window == 256000


@pytest.mark.parametrize("protocol", CODECS)
@pytest.mark.parametrize("kind", ["chat", "responses"])
def test_native_image_content_conversion(protocol, kind):
    uri = "data:image/png;base64,aW1hZ2U="
    image = (
        {"type": "image_url", "image_url": {"url": uri}}
        if kind == "chat"
        else {"type": "input_image", "image_url": uri}
    )
    history = [{"role": "user", "content": [{"type": "text", "text": "describe"}, image]}]
    original = deepcopy(history)
    request = codec(protocol).encode(history, tools=None)
    assert history == original
    if protocol == "anthropic":
        assert request["messages"][0]["content"][1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
        }
    elif protocol == "gemini":
        assert request["contents"][0].parts[1].inline_data.data == b"image"
        assert request["contents"][0].parts[1].inline_data.mime_type == "image/png"
    else:
        assert request["messages"][0]["content"] == "describe"
        assert request["messages"][0]["images"] == ["aW1hZ2U="]


@pytest.mark.parametrize("protocol", CODECS)
def test_native_remote_media_and_file_ids(protocol):
    history = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "https://images.test/a.png"}],
        }
    ]
    if protocol == "ollama":
        with pytest.raises(MessageValidationError, match="data URI"):
            codec(protocol).encode(history, tools=None)
    elif protocol == "anthropic":
        assert codec(protocol).encode(history, tools=None)["messages"][0]["content"][0][
            "source"
        ] == {"type": "url", "url": "https://images.test/a.png"}
    else:
        assert (
            codec(protocol).encode(history, tools=None)["contents"][0].parts[0].file_data.file_uri
            == "https://images.test/a.png"
        )
    with pytest.raises(MessageValidationError):
        codec(protocol).encode(
            [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "other-provider-file"}],
                }
            ],
            tools=None,
        )


@pytest.mark.parametrize("protocol", ["anthropic", "gemini"])
def test_native_document_content_conversion(protocol):
    request = codec(protocol).encode(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_data": "data:application/pdf;base64,cGRm"}
                ],
            }
        ],
        tools=None,
    )
    if protocol == "anthropic":
        assert request["messages"][0]["content"][0] == {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRm"},
        }
    else:
        assert request["contents"][0].parts[0].inline_data.data == b"pdf"


@pytest.mark.parametrize("protocol", CODECS)
def test_native_tools_and_configuration_are_independent_snapshots(protocol, function_tool):
    body = {"options": {"temperature": 0.1}, "thinking_config": {"thinking_budget": 1234}}
    instance = codec(protocol, supports_function_calling=True, extra_body=body)
    request = instance.encode(HISTORY, tools=[function_tool])
    function_tool["function"]["parameters"]["properties"]["city"]["type"] = "integer"
    if protocol == "gemini":
        assert (
            request["config"]
            .tools[0]
            .function_declarations[0]
            .parameters_json_schema["properties"]["city"]["type"]
            == "string"
        )
        request["config"].thinking_config.thinking_budget = 999
    elif protocol == "anthropic":
        assert request["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    else:
        assert (
            request["tools"][0]["function"]["parameters"]["properties"]["city"]["type"] == "string"
        )
        request["options"]["temperature"] = 999
    assert body["options"]["temperature"] == 0.1
    assert body["thinking_config"]["thinking_budget"] == 1234


@pytest.mark.parametrize("protocol", CODECS)
async def test_native_history_snapshots_only_once(protocol):
    from copy import deepcopy

    from my_llm import generate

    copies = []

    class Tracked(dict):
        def __deepcopy__(self, memo):
            copies.append(self["text"])
            return deepcopy(dict(self), memo)

    history = [{"role": "user", "content": [Tracked(type="text", text="original")]}]
    result = await generate(
        provider=ProviderProfile(protocol=protocol),
        model=ModelProfile(model="demo", max_tokens=2048),
        messages=history,
        stream=True,
    )
    history[0]["content"][0]["text"] = "changed"
    assert copies == ["original"]
    await result.aclose()  # An unconsumed stream must not create an SDK client.


@pytest.mark.parametrize("protocol", CODECS)
def test_native_ignores_unused_fields_without_copying_them(protocol):
    class Unused:
        def __deepcopy__(self, memo):
            raise AssertionError("unused configuration must not be copied for each request")

    instance = codec(protocol)
    instance.profile.values["extra_body"]["not_a_native_parameter"] = Unused()
    instance.encode(HISTORY, tools=None)


@pytest.mark.parametrize("protocol", ["anthropic", "gemini"])
def test_native_replay_detects_conflicting_unified_tools(protocol, tool_call):
    native = (
        {"type": "tool_use", "id": "call_1", "name": "different", "input": {}}
        if protocol == "anthropic"
        else {
            "type": "gemini_part",
            "part": {"function_call": {"id": "call_1", "name": "different", "args": {}}},
        }
    )
    with pytest.raises(MessageValidationError, match="must match native"):
        codec(protocol).encode(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                    "response_output": [native],
                }
            ],
            tools=None,
        )


def test_anthropic_supports_declared_xhigh_effort():
    request = codec("anthropic", supports_reasoning_effort=True).encode(
        HISTORY, tools=None, reasoning_effort="xhigh"
    )
    assert request["output_config"]["effort"] == "xhigh"


def test_gemini_rejects_nonobject_arguments_even_when_empty():
    with pytest.raises(IncompleteStreamError, match="JSON object"):
        codec("gemini").decoder().feed(
            {
                "candidates": [
                    {"content": {"parts": [{"function_call": {"name": "tool", "args": []}}]}}
                ]
            }
        )


@pytest.mark.parametrize(
    "protocol,binding",
    [
        ("anthropic", {"thinking": {"type": "disabled"}}),
        ("gemini", {"thinking_config": {"thinking_budget": 0}}),
        ("ollama", {"think": False}),
    ],
)
def test_native_reasoning_cannot_silently_disable_requested_capability(protocol, binding):
    with pytest.raises(ConfigurationError, match="reasoning"):
        codec(protocol, reasoning_enable_parameter=binding).encode(
            HISTORY, tools=None, reasoning=True
        )


@pytest.mark.parametrize("protocol", CODECS)
def test_native_omit_disables_usage(protocol):
    instance = CODECS[protocol](
        resolve_profiles(
            ProviderProfile(protocol=protocol, usage_mode=omit),
            ModelProfile(model="demo", max_tokens=2048),
        )
    )
    decoder = instance.decoder()
    for event in {
        "anthropic": [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ],
        "gemini": [
            {"candidates": [{"finish_reason": "STOP"}], "usage_metadata": {"total_token_count": 4}}
        ],
        "ollama": [{"done": True, "eval_count": 4}],
    }[protocol]:
        decoder.feed(event)
    assert decoder.finish().usage is None


def test_gemini_unspecified_statuses_are_not_completion_or_blocking():
    decoder = codec("gemini").decoder()
    chunks = decoder.feed(
        {
            "prompt_feedback": {"block_reason": "BLOCK_REASON_UNSPECIFIED"},
            "candidates": [
                {
                    "content": {"parts": [{"text": "x"}]},
                    "finish_reason": "FINISH_REASON_UNSPECIFIED",
                }
            ],
        }
    )
    assert chunks[0].content == "x"
    with pytest.raises(IncompleteStreamError):
        decoder.finish()


@pytest.mark.parametrize("protocol", CODECS)
def test_native_invalid_role_with_output_metadata_is_validation_error(protocol):
    with pytest.raises(MessageValidationError):
        codec(protocol).encode([{"content": "missing role", "response_output": []}], tools=None)
