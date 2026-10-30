"""Shared decoder policies borrow unchanged data and isolate filtered branches."""

from copy import deepcopy

import pytest

from my_llm import MyCodec, SemanticLossError, omit
from my_llm.basic.semantic_loss import SemanticLossContext, collect_losses, originated
from my_llm.codec.shared.output import output_chunks, output_messages
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools


def test_omitted_name_capability_does_not_restore_the_protocol_default(profile_factory):
    codec = MyCodec()
    inputs = codec.encode(
        [{"role": "user", "name": "speaker", "content": "complete tail"}],
        input_format="openai_chat",
    )
    with pytest.raises(SemanticLossError) as caught:
        codec.request(inputs, profile_factory(supports_message_name=omit))
    assert caught.value.draft["messages"] == [{"role": "user", "content": "complete tail"}]
    loss = caught.value.losses.items[0]
    assert loss.origin_path == "$.messages[0].name"
    assert loss.origin_value == "speaker"
    assert loss.reason == "model does not support message names"


@pytest.mark.parametrize("events", [False, True])
def test_output_without_losses_reuses_internal_containers(profile_factory, events):
    messages = [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]
    values = (
        [
            {"type": "delta", "field": "text", "value": "answer"},
            {"type": "finish", "messages": messages, "output_blocks": []},
        ]
        if events
        else messages
    )
    before = deepcopy(values)
    project = output_chunks if events else output_messages
    result = project(values, profile_factory())

    assert result is values
    assert values == before


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_output_loss_only_replaces_affected_branches(profile_factory, events, position):
    thought = originated(
        {"type": "thinking", "thinking": "unsupported"}, f"$.messages[{position}].content[0]"
    )
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": str(index)}]}
        for index in range(3)
    ]
    messages[position]["content"].insert(0, thought)
    event = {
        "type": "finish",
        "messages": messages,
        "output_blocks": [
            {"block": messages[0]["content"][-1]},
            {"block": thought},
        ],
    }
    unchanged_event = {"type": "delta", "field": "text", "value": "answer"}
    values = [unchanged_event, event] if events else messages
    before = deepcopy(values)
    profile = profile_factory(reasoning_enable=parameter("reasoning_enable", supports=False))

    with collect_losses(SemanticLossContext("openai_chat", "openai_chat", "output")) as (losses, _):
        result = output_chunks(values, profile) if events else output_messages(values, profile)

    assert result is not values
    projected = result[1]["messages"] if events else result
    for index, message in enumerate(messages):
        assert (projected[index] is message) is (index != position)
        assert projected[index]["content"][-1] is message["content"][-1]
    if events:
        assert result[0] is unchanged_event
        assert result[1] is not event
        assert result[1]["output_blocks"] == event["output_blocks"][:1]
    assert len(losses.items) == 1
    assert values == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_encoder_does_not_apply_decoder_capability_policy(monkeypatch, profile_factory, protocol):
    def fail(*args, **kwargs):
        pytest.fail("encoder entered semantic loss handling")

    codec = MyCodec("hook", input_semantic_loss_hook=fail, output_semantic_loss_hook=fail)
    profile = profile_factory(
        protocol,
        supports_function_calling=config(False),
        supports_vision=config(False),
        supports_message_name=config(False),
        reasoning_enable=parameter("reasoning_enable", protocol=protocol, supports=False),
        feedback_reasoning_content=config("none"),
    )
    monkeypatch.setattr("my_llm.codec.shared.capabilities.report_loss", fail)
    monkeypatch.setattr("my_llm.cache.compiled_cache.report_loss", fail)
    monkeypatch.setattr("my_llm.codec.shared.conversion.report_loss", fail)
    monkeypatch.setattr("my_llm.codec.base.codec.BaseCodec.translate", fail)
    source = history(protocol)
    definitions = tools(protocol)
    before = deepcopy(source)

    result = codec.encode(
        source, definitions, chunks=stream_events(protocol), input_format=protocol, profile=profile
    )
    envelope = codec.codecs[protocol].encoder.request(
        {"messages": source, "tools": definitions}, profile
    )

    assert result.messages and result.tools and result.stream
    assert envelope["middleware"].messages and envelope["middleware"].tools
    assert source == before
