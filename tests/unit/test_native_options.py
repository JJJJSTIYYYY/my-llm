"""Native option handling must preserve intent or fail before opening a stream."""

from copy import deepcopy

import pytest

from my_llm import (
    ConfigurationError,
    MyCodec,
    SemanticLossError,
)
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history
from tests.helpers.requests import encode_request


@pytest.mark.parametrize("protocol", ["anthropic", "gemini"])
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


@pytest.mark.parametrize("protocol", ["anthropic", "gemini"])
def test_native_request_options_are_owned(profile_factory, protocol):
    values = {
        "anthropic": {"metadata": {"user_id": "original"}},
        "gemini": {
            "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"}]
        },
    }[protocol]
    before = deepcopy(values)
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        extra_body=config(values),
    )
    request = encode_request(profile)
    if protocol == "gemini":
        request["config"]["safety_settings"][0]["threshold"] = "BLOCK_ONLY_HIGH"
    elif protocol == "anthropic":
        request["metadata"]["user_id"] = "changed"
    assert values == before
    assert profile.get("extra_body") == before


def test_unknown_anthropic_option_is_not_a_validation_error(profile_factory):
    request = encode_request(
        profile_factory("anthropic", max_tokens=parameter("max_tokens", 64, protocol="anthropic")),
        extra_body={"unexpected": 1},
    )
    assert "unexpected" not in request


@pytest.mark.parametrize("source", ["openai_chat", "openai_responses", "anthropic", "gemini"])
def test_chat_can_disable_compatible_reasoning_replay(profile_factory, source):
    codec = MyCodec()
    intermediate = codec.encode(history(source), input_format=source)
    profile = profile_factory(
        "openai_chat",
        supports_function_calling=config(True),
        should_feedback_reasoning_item=config("none"),
    )
    request = codec.request(intermediate, profile, reasoning=True)
    assert request["reasoning_effort"] == "medium"
    assert all("reasoning_content" not in message for message in request["messages"])


@pytest.mark.parametrize("replay", [None, config("always")])
def test_chat_replays_compatible_reasoning_by_default_and_when_enabled(profile_factory, replay):
    codec = MyCodec()
    profile = profile_factory(
        "openai_chat",
        should_feedback_reasoning_item=replay,
        supports_function_calling=config(True),
    )
    request = codec.request(
        codec.encode(history("openai_chat"), input_format="openai_chat"), profile, reasoning=True
    )
    assistant = next(message for message in request["messages"] if message.get("tool_calls"))
    assert assistant["reasoning_content"] == ""


@pytest.mark.parametrize("protocol,field", [("openai_chat", "n"), ("gemini", "candidate_count")])
def test_multiple_candidates_are_losses_before_native_execution(profile_factory, protocol, field):
    with pytest.raises(SemanticLossError, match="candidate") as caught:
        encode_request(profile_factory(protocol), extra_body={field: 2})
    body = caught.value.draft.get("config", caught.value.draft.get("extra_body", {}))
    assert body[field] == 1


def test_native_sdk_configuration_objects_remain_usable_with_generated_parameters(profile_factory):
    from google.genai import types

    thinking = types.ThinkingConfig(thinking_budget=512, include_thoughts=False)
    request = encode_request(
        profile_factory("gemini"), reasoning=True, extra_body={"thinking_config": thinking}
    )
    assert request["config"]["thinking_config"]["thinking_budget"] == -1
    assert request["config"]["thinking_config"]["include_thoughts"] is True
    assert thinking.include_thoughts is False
