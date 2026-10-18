"""Decoder source groups are part of the protocol implementation contract."""

from copy import deepcopy

import pytest

from my_llm import MyCodec, Translations
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.shared.conversion import FUNCTIONS, REPLAY
from tests.helpers.protocols import PROTOCOLS, history, tools


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_builtin_decoder_hooks_declare_sources_and_public_methods_return_lists(protocol, kind):
    converter = MyCodec()
    native = history(protocol) if kind == "messages" else tools(protocol)
    values = converter.encode(input_format=protocol, **{kind: native})[kind]
    decoder = converter.decoder(protocol)

    translated = getattr(decoder, f"_{kind}")(values)

    assert isinstance(translated, Translations)
    assert [source for _, sources, _ in translated.groups for source in sources] == values
    assert type(getattr(decoder, kind)(values)) is list


def test_message_extensions_follow_source_groups_after_one_message_expands():
    class SplittingDecoder(ChatCompletionsDecoder):
        __slots__ = ()
        protocol = "splitting"

        def _messages(self, values, profile=None, **parameters) -> Translations:
            result = Translations()
            for message in values:
                result.group([message], structure={"content": [{}]})
                for text in message["content"][0]["text"].split("|"):
                    result.append(
                        {"role": message["role"], "content": [{"type": "text", "text": text}]}
                    )
            return result

    values = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "first|second", "future": {"source": 0}}],
            "future": {"source": 0},
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "third", "future": {"source": 1}}],
            "future": {"source": 1},
        },
    ]
    before = deepcopy(values)

    result = SplittingDecoder().messages(values)

    assert type(result) is list
    assert len(result) == 3
    assert result[0]["future"] == {"source": 0}
    assert "future" not in result[1]
    assert result[2]["future"] == {"source": 1}
    # A split has no unambiguous target content index. Only the one-to-one group
    # can merge an unknown nested extension into the target content list.
    assert all("future" not in item["content"][0] for item in result[:2])
    assert result[2]["content"][0]["future"] == {"source": 1}
    assert result[0][REPLAY]["values"] == before
    assert values == before


def test_tool_extensions_follow_declared_sources_when_output_order_changes():
    class ReorderingDecoder(ChatCompletionsDecoder):
        __slots__ = ()
        protocol = "reordering"

        def _tools(self, values, profile=None, **parameters) -> Translations:
            result = Translations()
            for tool in reversed(values):
                result.group([tool], structure={"function": {}})
                result.append({"type": "function", "function": tool[FUNCTIONS][0]})
            return result

    values = [
        {FUNCTIONS: [{"name": "first"}], "function": {"future": {"source": 0}}},
        {FUNCTIONS: [{"name": "second"}], "function": {"future": {"source": 1}}},
    ]
    before = deepcopy(values)

    result = ReorderingDecoder().tools(values)

    assert type(result) is list
    assert result[0]["function"] == {"name": "second", "future": {"source": 1}}
    assert result[1]["function"] == {"name": "first", "future": {"source": 0}}
    assert values == before
