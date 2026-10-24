"""An omitted I/O protocol must be inferred uniquely, never by priority."""

import pytest

from my_llm import (
    Attribute,
    Config,
    MessageValidationError,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    generate,
)
from tests.helpers.profiles import parameter
from tests.helpers.protocols import PROTOCOLS, history, tools


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_complete_native_history_identifies_one_protocol(protocol):
    codec = MyCodec()
    assert codec.match_protocol(history(protocol)) == protocol


@pytest.mark.parametrize(
    "values",
    [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        tools("openai_chat"),
        [],
        [{"unknown": "value"}],
        [{"role": "user", "content": "hello"}, {"role": "model", "parts": [{"text": "hi"}]}],
    ],
)
def test_non_unique_protocol_has_no_registration_order_fallback(values):
    codec = MyCodec()
    assert codec.match_protocol(values) is None
    codec.codecs = dict(reversed(list(codec.codecs.items())))
    assert codec.match_protocol(values) is None


@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
async def test_ambiguous_input_fails_before_request_without_binding(monkeypatch, upstream, stream):
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value=upstream)),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            max_tokens=parameter("max_tokens", 4096, protocol=upstream),
        ),
    )

    def must_not_request(*args, **kwargs):
        pytest.fail("ambiguous input reached native request construction")

    monkeypatch.setattr(MyCodec, "request", must_not_request)
    with pytest.raises(
        MessageValidationError,
        match="stdio_protocol is not specified and cannot be inferred uniquely",
    ):
        await llm.generate(messages=[{"role": "user", "content": "hello"}], stream=stream)
    assert llm.stdio_protocol is None


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_failed_inference_leaves_a_later_unique_binding_available(protocol):
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(model=Attribute(default_value="demo")),
    )
    for values in ([], [{"unknown": "value"}], [{"role": "user", "content": "hello"}]):
        with pytest.raises(MessageValidationError, match="stdio_protocol"):
            llm.guess_stdio_protocol(values)
        assert llm.stdio_protocol is None
    assert llm.guess_stdio_protocol(history(protocol)) == protocol
    assert llm.stdio_protocol == protocol


async def test_one_shot_generation_also_requires_a_unique_or_explicit_protocol():
    with pytest.raises(MessageValidationError, match="stdio_protocol"):
        await generate(
            provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
            model="demo",
            messages=[{"role": "user", "content": "hello"}],
        )
