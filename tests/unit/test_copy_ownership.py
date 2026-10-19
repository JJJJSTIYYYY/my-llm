"""Mutable public conversion results own their nested protocol payloads."""

from copy import deepcopy

import pytest
from google.genai import types
from pydantic import BaseModel

from my_llm import MyCodec
from my_llm.codec.shared.conversion import REPLAY, SOURCE
from my_llm.codec.shared.options import resolve_options
from tests.helpers.protocols import PROTOCOLS, history, tools


def edit_containers(value):
    """Edit mappings and sequences, including payloads inside SDK request models."""
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            edit_containers(getattr(value, name))
    elif isinstance(value, dict):
        for child in tuple(value.values()):
            edit_containers(child)
        value["x_test_edit"] = ["changed"]
    elif isinstance(value, list):
        for child in tuple(value):
            edit_containers(child)
        value.append({"x_test_edit": "changed"})


def native_values(protocol, kind):
    values = history(protocol) if kind == "messages" else tools(protocol)
    values[0]["x_vendor"] = {"nested": [{"labels": ["original"]}]}
    return values


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_encoder_owns_nested_output(protocol, kind):
    codec = MyCodec().codecs[protocol]
    values = native_values(protocol, kind)
    before = deepcopy(values)

    converted = getattr(codec, f"encode_{kind}")(values)
    edit_containers(converted)

    assert values == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_encoder_captures_nested_input(protocol, kind):
    codec = MyCodec().codecs[protocol]
    values = native_values(protocol, kind)
    converted = getattr(codec, f"encode_{kind}")(values)
    before = deepcopy(converted)

    edit_containers(values)

    assert converted == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_decoder_owns_nested_output(source, target, kind):
    codec = MyCodec()
    intermediate = codec.encode(**{kind: native_values(source, kind)}, input_format=source)[kind]
    before = deepcopy(intermediate)

    converted = getattr(codec.codecs[target], f"decode_{kind}")(intermediate)
    edit_containers(converted)

    assert intermediate == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_decoder_captures_nested_input(source, target, kind):
    codec = MyCodec()
    intermediate = codec.encode(**{kind: native_values(source, kind)}, input_format=source)[kind]
    converted = getattr(codec.codecs[target], f"decode_{kind}")(intermediate)
    before = deepcopy(converted)

    edit_containers(intermediate)

    assert converted == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_decoder_without_native_snapshot_owns_nested_output(protocol, kind):
    codec = MyCodec()
    intermediate = codec.encode(**{kind: native_values(protocol, kind)}, input_format=protocol)[
        kind
    ]
    for value in intermediate:
        value[SOURCE].pop("value", None)
    before = deepcopy(intermediate)

    converted = getattr(codec.codecs[protocol], f"decode_{kind}")(intermediate)
    assert intermediate == before
    edit_containers(converted)

    assert intermediate == before


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_public_decoder_without_native_snapshot_captures_nested_input(protocol, kind):
    codec = MyCodec()
    intermediate = codec.encode(**{kind: native_values(protocol, kind)}, input_format=protocol)[
        kind
    ]
    for value in intermediate:
        value[SOURCE].pop("value", None)
    converted = getattr(codec.codecs[protocol], f"decode_{kind}")(intermediate)
    before = deepcopy(converted)

    edit_containers(intermediate)

    assert converted == before


def test_saved_replay_is_independent_of_later_intermediate_edits():
    codec = MyCodec()
    native = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "reason", "signature": "signed"},
                {"type": "text", "text": "answer"},
            ],
            "x_vendor": {"labels": ["original"]},
        }
    ]
    intermediate = codec.encode(native, input_format="anthropic")["messages"]
    converted = codec.decode(intermediate, target_format="openai_chat")["messages"]
    assert REPLAY in converted[0]
    before = deepcopy(converted)

    edit_containers(intermediate)

    assert converted == before
    restored = codec.convert(converted, input_format="openai_chat", target_format="anthropic")[
        "messages"
    ]
    assert restored == native


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_repeated_public_requests_do_not_share_mutable_input(profile_factory, protocol):
    codec = MyCodec()
    intermediate = codec.encode(history("gemini"), tools("gemini"), input_format="gemini")
    profile = profile_factory(protocol, max_tokens=2048, supports_function_calling=True)
    before = deepcopy(intermediate)
    first = codec.request(intermediate, profile)
    expected = deepcopy(first)

    edit_containers(first)

    assert intermediate == before
    assert codec.request(intermediate, profile) == expected


def test_expanded_binding_owns_native_sdk_option_model(profile_factory):
    profile = profile_factory(
        "gemini",
        reasoning_enable_parameter={"thinking_config": types.ThinkingConfig(include_thoughts=True)},
    )
    before = deepcopy(profile.values)
    options = resolve_options(profile, reasoning=True)

    options.bindings[0]["thinking_config"].include_thoughts = False

    assert profile.values == before


def test_expanded_binding_owns_mutable_values_inside_tuple(profile_factory):
    profile = profile_factory(
        "gemini",
        reasoning_enable_parameter={
            "safety_settings": (
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            )
        },
    )
    before = deepcopy(profile.values)
    options = resolve_options(profile, reasoning=True)

    options.bindings[0]["safety_settings"][0].threshold = "BLOCK_ONLY_HIGH"

    assert profile.values == before
