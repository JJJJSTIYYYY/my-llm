"""Native option handling must preserve intent or fail before opening a stream."""

from copy import deepcopy

import pytest

from my_llm import (
    MyCodec,
    SemanticLossError,
)
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import history
from tests.helpers.requests import encode_request


@pytest.mark.parametrize("protocol", ["anthropic"])
def test_unsupported_native_option_is_skipped(profile_factory, protocol):
    request = encode_request(
        profile_factory(protocol, max_tokens=parameter("max_tokens", 2048, protocol=protocol)),
        extra_body={"unexpected": {"value": True}},
    )
    body = request
    assert "unexpected" not in body


@pytest.mark.parametrize("protocol", ["anthropic"])
def test_native_request_options_are_owned(profile_factory, protocol):
    values = {"anthropic": {"metadata": {"user_id": "original"}}}[protocol]
    before = deepcopy(values)
    profile = profile_factory(
        protocol,
        max_tokens=parameter("max_tokens", 2048, protocol=protocol),
        extra_body=config(values),
    )
    request = encode_request(profile)
    request["metadata"]["user_id"] = "changed"
    assert values == before
    assert profile.get("extra_body") == before


def test_unknown_anthropic_option_is_not_a_validation_error(profile_factory):
    request = encode_request(
        profile_factory("anthropic", max_tokens=parameter("max_tokens", 64, protocol="anthropic")),
        extra_body={"unexpected": 1},
    )
    assert "unexpected" not in request


@pytest.mark.parametrize("source", ["openai_chat", "openai_responses", "anthropic"])
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


@pytest.mark.parametrize("protocol,field", [("openai_chat", "n")])
def test_multiple_candidates_are_losses_before_native_execution(profile_factory, protocol, field):
    with pytest.raises(SemanticLossError, match="candidate") as caught:
        encode_request(profile_factory(protocol), extra_body={field: 2})
    body = caught.value.draft.get("extra_body", {})
    assert body[field] == 1
