"""Native transport boundaries validate and filter every request-bearing structure."""

from copy import deepcopy

import pytest

from my_llm import ConfigurationError, MyCodec, UnsupportedCapabilityError
from my_llm.codec.shared.conversion import REPLAY


@pytest.mark.parametrize("target", ["anthropic", "gemini"])
@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("source", ["openai_chat", "gemini"])
def test_system_extensions_are_preserved_in_history_and_filtered_at_the_sdk_boundary(
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
    before_request = deepcopy(intermediate)
    request = codec.request(intermediate, profile_factory(target, max_tokens=2048))
    assert history == original
    assert intermediate == before_request
    assert "vendor_" not in str(request)
    assert REPLAY not in str(request)
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
