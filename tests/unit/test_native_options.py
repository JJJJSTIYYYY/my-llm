"""Native option handling must preserve intent or fail before opening a stream."""

from copy import deepcopy

import pytest

from my_llm import ConfigurationError, ModelProfile, MyCodec, MyLLM, ProviderProfile
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history
from tests.helpers.requests import encode_request


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_unsupported_native_option_is_skipped(profile_factory, protocol):
    request = encode_request(
        profile_factory(protocol, max_tokens=parameter("max_tokens", 2048, protocol=protocol)),
        extra_body={"unexpected": {"value": True}},
    )
    body = request["config"] if protocol == "gemini" else request
    assert "unexpected" not in body


@pytest.mark.parametrize("key", ["automatic_function_calling", "http_options"])
def test_gemini_cannot_override_owned_configuration(profile_factory, key):
    with pytest.raises(ConfigurationError, match="cannot override protocol parameters"):
        encode_request(profile_factory("gemini"), extra_body={key: {}})


@pytest.mark.parametrize("protocol", ["anthropic", "gemini", "ollama"])
def test_native_request_options_are_owned(profile_factory, protocol):
    values = {
        "anthropic": {"metadata": {"user_id": "original"}},
        "gemini": {
            "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"}]
        },
        "ollama": {"options": {"temperature": 0.5}},
    }[protocol]
    before = deepcopy(values)
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        extra_body=config("extra_body", values),
    )
    request = encode_request(profile)
    if protocol == "gemini":
        request["config"]["safety_settings"][0]["threshold"] = "BLOCK_ONLY_HIGH"
    elif protocol == "anthropic":
        request["metadata"]["user_id"] = "changed"
    else:
        request["options"]["temperature"] = 1
    assert values == before
    assert profile.get("extra_body") == before


def test_unknown_anthropic_option_is_not_a_validation_error(profile_factory):
    request = encode_request(
        profile_factory("anthropic", max_tokens=parameter("max_tokens", 64, protocol="anthropic")),
        extra_body={"unexpected": 1},
    )
    assert "unexpected" not in request


@pytest.mark.parametrize(
    "source", ["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"]
)
def test_official_chat_requests_do_not_pad_compatible_reasoning_fields(profile_factory, source):
    codec = MyCodec()
    intermediate = codec.encode(history(source), input_format=source)
    profile = profile_factory(
        "openai_chat", supports_function_calling=config("supports_function_calling", True)
    )
    request = codec.request(intermediate, profile, reasoning=True)
    assert request["reasoning_effort"] == "medium"
    assert all("reasoning_content" not in message for message in request["messages"])


def test_compatible_reasoning_replay_remains_explicitly_available(profile_factory):
    codec = MyCodec()
    profile = profile_factory(
        "openai_chat",
        replay_reasoning_content_when_tool_calls=config(
            "replay_reasoning_content_when_tool_calls", True
        ),
    )
    request = codec.request(
        codec.encode(history("openai_chat"), input_format="openai_chat"), profile, reasoning=True
    )
    assistant = next(message for message in request["messages"] if message.get("tool_calls"))
    assert assistant["reasoning_content"] == ""


def test_ollama_effort_mapping_can_use_model_defined_strings(profile_factory):
    profile = profile_factory(
        "ollama",
        supports_reasoning_effort=config("supports_reasoning_effort", True),
        reasoning_effort_mapping=config("reasoning_effort_mapping", {"high": "deep"}),
    )
    request = encode_request(profile, reasoning_effort="high")
    assert request["think"] == "deep"


@pytest.mark.parametrize("retries", [1, 3])
def test_ollama_cannot_silently_ignore_requested_retries(retries):
    from my_llm import UnsupportedCapabilityError

    with pytest.raises(UnsupportedCapabilityError, match="max_retries"):
        MyLLM(
            stdio_protocol="openai_chat",
            provider=ProviderProfile(protocol="ollama", max_retries=config("max_retries", retries)),
            model=ModelProfile(model="demo"),
        )


@pytest.mark.parametrize("protocol,field", [("openai_chat", "n"), ("gemini", "candidate_count")])
def test_multiple_candidates_are_rejected_before_native_execution(profile_factory, protocol, field):
    from my_llm import UnsupportedCapabilityError

    with pytest.raises(UnsupportedCapabilityError, match="one generated candidate"):
        encode_request(profile_factory(protocol), extra_body={field: 2})


def test_native_sdk_configuration_objects_remain_usable_with_generated_parameters(profile_factory):
    from google.genai import types
    from ollama import Options

    thinking = types.ThinkingConfig(thinking_budget=512, include_thoughts=False)
    request = encode_request(
        profile_factory("gemini"), reasoning=True, extra_body={"thinking_config": thinking}
    )
    assert request["config"]["thinking_config"]["thinking_budget"] == 512
    assert request["config"]["thinking_config"]["include_thoughts"] is False
    assert thinking.include_thoughts is False
    options = Options(num_predict=7, temperature=0.2)
    request = encode_request(
        profile_factory("ollama"), max_tokens=10, extra_body={"options": options}
    )
    assert request["options"] == {"num_predict": 10, "temperature": 0.2}
    assert options.num_predict == 7
