"""Decoder implementations return ordinary lists without restoration metadata."""

from copy import deepcopy

import pytest

from my_llm import MyCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
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
