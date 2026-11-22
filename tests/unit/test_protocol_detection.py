"""An omitted I/O protocol must be inferred uniquely, never by priority."""

import pytest

from my_llm import (
    Attribute,
    Config,
    ConfigurationError,
    MessageValidationError,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    UnsupportedProviderError,
)
from tests.helpers.profiles import parameter
from tests.helpers.protocols import PROTOCOLS, chat_event, history, tools
from tests.helpers.transport import FakeClient


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_complete_native_history_identifies_one_protocol(protocol):
    codec = MyCodec()
    assert codec._match_protocol(history(protocol)) == protocol


@pytest.mark.parametrize(
    "values",
    [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        [],
        [{"unknown": "value"}],
        [{"role": "user", "content": "hello"}, {"role": "model", "parts": [{"text": "hi"}]}],
    ],
)
def test_non_unique_protocol_has_no_registration_order_fallback(values):
    codec = MyCodec()
    assert codec._match_protocol(values) is None
    codec._codecs = dict(reversed(list(codec._codecs.items())))
    assert codec._match_protocol(values) is None


def test_chat_tool_declarations_uniquely_identify_chat():
    assert MyCodec()._match_protocol(tools("openai_chat")) == "openai_chat"


def test_removed_ollama_protocol_is_rejected_as_upstream_and_stdio():
    model = ModelProfile(model=Attribute(default_value="demo"))
    with pytest.raises(UnsupportedProviderError, match="ollama"):
        MyLLM(provider=ProviderProfile(protocol=Config(default_value="ollama")), model=model)
    with pytest.raises(ConfigurationError, match="ollama"):
        MyLLM(
            provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
            model=model,
            stdio_protocol="ollama",
        )


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

    monkeypatch.setattr(MyCodec, "_request", must_not_request)
    with pytest.raises(
        MessageValidationError,
        match="stdio_protocol is not specified and cannot be inferred uniquely",
    ):
        await llm.generate(messages=[{"role": "user", "content": "hello"}], stream=stream)
    assert llm.stdio_protocol is None


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_failed_generation_inference_leaves_a_later_unique_binding_available(
    protocol, monkeypatch
):
    client = FakeClient([chat_event(content="ok"), chat_event(finish="stop")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=True),
        ),
    )
    for values in ([], [{"unknown": "value"}], [{"role": "user", "content": "hello"}]):
        with pytest.raises(MessageValidationError, match="stdio_protocol"):
            await llm.generate(messages=values)
        assert llm.stdio_protocol is None
    await llm.generate(messages=history(protocol))
    assert llm.stdio_protocol == protocol
    assert len(client.requests) == 1


async def test_generation_requires_a_unique_or_explicit_protocol():
    with pytest.raises(MessageValidationError, match="stdio_protocol"):
        await MyLLM(
            provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
            model=ModelProfile(model=Attribute(default_value="demo")),
        ).generate(messages=[{"role": "user", "content": "hello"}])
