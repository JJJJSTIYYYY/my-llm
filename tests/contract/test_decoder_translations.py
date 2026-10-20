"""Decoder implementations return ordinary lists without restoration metadata."""

from copy import deepcopy

import pytest

from my_llm import MyCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from tests.helpers.profiles import parameter
from tests.helpers.protocols import PROTOCOLS, history, tools


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_builtin_decoder_hooks_and_public_methods_return_plain_lists(protocol, kind):
    converter = MyCodec()
    native = history(protocol) if kind == "messages" else tools(protocol)
    values = converter.encode(input_format=protocol, **{kind: native})[kind]
    decoder = converter.decoder(protocol)
    before = deepcopy(values)

    assert type(getattr(decoder, f"_{kind}")(values)) is list
    assert type(getattr(decoder, kind)(values)) is list
    assert values == before


def test_decoder_message_expansion_needs_no_source_groups_or_recovery_records():
    class SplittingDecoder(ChatCompletionsDecoder):
        __slots__ = ()
        protocol = "splitting"

        def _messages(self, values, profile=None, **parameters):
            return [
                {"role": value["role"], "content": piece}
                for value in values
                for piece in value["content"][0]["text"].split("|")
            ]

    values = [
        {"role": "user", "content": [{"type": "text", "text": "first|second"}], "vendor": 1},
        {"role": "user", "content": [{"type": "text", "text": "third"}], "vendor": 2},
    ]
    before = deepcopy(values)

    assert SplittingDecoder().messages(values) == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    assert values == before


DELTA_PATHS = {
    "openai_chat": ("$.choices[0].delta.content", "$.choices[0].delta.reasoning_content"),
    "openai_responses": ("$.delta", "$.delta"),
    "anthropic": ("$.delta.text", "$.delta.thinking"),
    "gemini": (
        "$.candidates[0].content.parts[0].text",
        "$.candidates[0].content.parts[0].text",
    ),
    "ollama": ("$.message.content", "$.message.thinking"),
}


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "semantic,field,index", [("content", "text", 0), ("reasoning_content", "thinking", 1)]
)
@pytest.mark.parametrize("phase", ["encode", "decode"])
@pytest.mark.parametrize("value", ["translated", None])
def test_parameter_translation_does_not_rewrite_standard_chunk_output(
    protocol, semantic, field, index, phase, value, profile_factory
):
    from my_llm.codec.shared.options import path_value

    profile = profile_factory(
        protocol,
        **{
            semantic: parameter(
                semantic, "fallback", protocol=protocol, path="$.vendor.payload", phase=phase
            )
        },
    )
    result = (
        MyCodec()
        .decoder(protocol)
        .chunks(
            [{"type": "delta", "field": field, "value": value}],
            profile,
            source_protocol=protocol,
        )
    )

    assert len(result) == 1
    assert path_value(result[0], DELTA_PATHS[protocol][index], "missing") == value
    assert "vendor" not in result[0]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "semantic,field,index",
    [
        ("content", "text", 0),
        ("reasoning_content", "thinking", 1),
    ],
)
@pytest.mark.parametrize("custom_path", [False, True])
def test_encoder_missing_text_uses_default_at_standard_and_custom_paths(
    protocol,
    semantic,
    field,
    index,
    custom_path,
    profile_factory,
):
    from my_llm.codec.shared.options import delete_path, set_path

    native_path = DELTA_PATHS[protocol][index]
    overrides = {
        semantic: parameter(
            semantic,
            "fallback",
            protocol=protocol,
            path="$.vendor.payload" if custom_path else None,
        )
    }
    profile = profile_factory(protocol, **overrides)
    codec = MyCodec().codecs[protocol]
    event = codec.decoder.chunks(
        [{"type": "delta", "field": field, "value": "original"}],
        source_protocol=protocol,
    )[0]
    delete_path(event, native_path)
    if custom_path:
        set_path(event, "$.vendor", {})
    encoded = codec.encoder.chunks([event], profile)
    operations = [op for batch in encoded for op in batch["events"]]
    values = [
        op["value"] if op["type"] == "block_delta" else op["block"][field]
        for op in operations
        if op["type"] == "block_delta"
        and op.get("field") == field
        or op["type"] == "block_append"
        and op["block"]["type"] == field
    ]
    assert values == ["fallback"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "semantic,field",
    [
        ("content", "text"),
        ("reasoning_content", "thinking"),
    ],
)
def test_encoder_text_omit_skips_existing_payload(protocol, semantic, field, profile_factory):
    from my_llm.profile.types import omit

    overrides = {semantic: omit}
    profile = profile_factory(protocol, **overrides)
    codec = MyCodec().codecs[protocol]
    event = codec.decoder.chunks(
        [{"type": "delta", "field": field, "value": "remove me"}],
        source_protocol=protocol,
    )[0]
    operations = [op for batch in codec.encoder.chunks([event], profile) for op in batch["events"]]

    assert not any(
        op["type"] == "block_delta"
        and op.get("field") == field
        or op["type"] == "block_append"
        and op["block"]["type"] == field
        for op in operations
    )


def test_responses_known_item_header_does_not_copy_unknown_metadata():
    class OnlyKnownKeys(dict):
        def keys(self):
            raise AssertionError("scanned an item header")

        def __deepcopy__(self, memo):
            raise AssertionError("copied an unprojected item header")

    item = OnlyKnownKeys(type="message", id="message_1", role="assistant", vendor=object())
    result = (
        MyCodec()
        .decoder("openai_responses")
        .chunks(
            [{"type": "block_start", "block": {"type": "text"}, "source_ref": ["text", 0, 0]}],
            source_protocol="openai_responses",
            response_items={"0": item},
        )
    )
    assert result[0]["item"] == {
        "type": "message",
        "id": "message_1",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }


@pytest.mark.parametrize("custom_path", [False, True])
def test_gemini_defaults_do_not_create_text_on_usage_or_finish_only_events(
    custom_path, profile_factory
):
    overrides = {
        "content": parameter(
            "content",
            "fallback",
            protocol="gemini",
            path="$.vendor.answer" if custom_path else None,
        ),
        "reasoning_content": parameter(
            "reasoning_content",
            "fallback thought",
            protocol="gemini",
            path="$.vendor.thought" if custom_path else None,
        ),
    }
    profile = profile_factory("gemini", **overrides)
    events = [
        {"usage_metadata": {"prompt_token_count": 1}},
        {"candidates": [{"finish_reason": "STOP"}]},
    ]
    operations = [
        op
        for batch in MyCodec().codecs["gemini"].encoder.chunks(events, profile)
        for op in batch["events"]
    ]
    assert not any(op["type"] == "block_append" for op in operations)


@pytest.mark.parametrize("protocol", ["openai_chat", "ollama"])
@pytest.mark.parametrize("custom_path", [False, True])
@pytest.mark.parametrize("actual_payload", [False, True])
def test_terminal_events_do_not_append_missing_text_defaults(
    protocol,
    custom_path,
    actual_payload,
    profile_factory,
):
    from my_llm.codec.shared.options import set_path

    overrides = {
        "content": parameter(
            "content",
            "fallback",
            protocol=protocol,
            path="$.vendor.answer" if custom_path else None,
        ),
        "reasoning_content": parameter(
            "reasoning_content",
            "fallback thought",
            protocol=protocol,
            path="$.vendor.thought" if custom_path else None,
        ),
    }
    profile = profile_factory(protocol, **overrides)
    event = (
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        if protocol == "openai_chat"
        else {"done": True, "message": {}}
    )
    if actual_payload:
        set_path(event, "$.vendor.answer" if custom_path else DELTA_PATHS[protocol][0], "last")
    operations = [
        op
        for batch in MyCodec().codecs[protocol].encoder.chunks([event], profile)
        for op in batch["events"]
    ]
    payloads = [op["value"] for op in operations if op["type"] == "block_delta"]

    assert payloads == (["last"] if actual_payload else [])


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "semantic,field,index", [("content", "text", 0), ("reasoning_content", "thinking", 1)]
)
@pytest.mark.parametrize("custom_path", [False, True])
def test_present_null_response_does_not_use_configured_text_default(
    protocol, semantic, field, index, custom_path, profile_factory
):
    from my_llm.codec.shared.options import set_path

    declaration = parameter(
        semantic,
        "must not appear",
        protocol=protocol,
        path="$.vendor.payload" if custom_path else None,
    )
    profile = profile_factory(protocol, **{semantic: declaration})
    codec = MyCodec().codecs[protocol]
    native = codec.decoder.chunks(
        [{"type": "delta", "field": field, "value": "native"}], source_protocol=protocol
    )[0]
    set_path(native, declaration.path, None)
    operations = [
        event for batch in codec.encoder.chunks([native], profile) for event in batch["events"]
    ]
    payloads = [
        event["value"] if event["type"] == "block_delta" else event["block"].get(field)
        for event in operations
        if event["type"] == "block_delta" or event["type"] == "block_append"
    ]
    assert "must not appear" not in payloads
