"""Optional accounting and metadata never prevent a valid assistant response."""

from contextlib import asynccontextmanager
from copy import deepcopy

import pytest

from my_llm import MessageValidationError, MyCodec, MyLLM, Usage
from my_llm.codec.shared.stream import metadata
from my_llm.runtime.state import StreamState
from tests.helpers.protocols import PROTOCOLS, chat_event, stream_events
from tests.helpers.transport import FakeClient, FakeStream
from tests.performance.workloads import profiles


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("value", [None, "bad", [], 1, True, {}])
def test_native_usage_ignores_unparseable_values(protocol, value):
    assert MyCodec().codecs[protocol].encoder.usage(value) is None


@pytest.mark.parametrize(
    "protocol,usage",
    [
        ("openai_chat", {"prompt_tokens": "bad", "completion_tokens": 2}),
        ("openai_chat", {"prompt_tokens_details": "bad"}),
        ("openai_responses", {"input_tokens_details": []}),
        ("anthropic", {"input_tokens": 2, "cache_read_input_tokens": "bad"}),
        ("gemini", {"prompt_token_count": True}),
        ("ollama", {"eval_count": -1}),
    ],
)
def test_invalid_native_counter_or_nested_details_return_empty(protocol, usage):
    assert MyCodec().codecs[protocol].encoder.usage(usage) is None


@pytest.mark.parametrize(
    "value",
    [
        "bad",
        [],
        {"input": "bad"},
        {"input": {"base": 2, "add": {"cache": "bad"}}},
        {"input": {"base": True}},
        {"input": {"base": 2, "add": []}},
        {"total": -1},
        {"cached": False},
    ],
)
def test_invalid_canonical_usage_is_empty_at_all_public_projections(value):
    assert Usage.from_record(value) is None
    for protocol in PROTOCOLS:
        assert MyCodec().codecs[protocol].decoder.usage(value) is None
    state = StreamState("demo")
    state.feed([{"type": "usage", "usage": value}])
    assert state.usage is None


def test_bad_metadata_does_not_partially_update_response_state():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("auxiliary data cannot be copied")

    state = StreamState("demo")
    state.feed([metadata({"scalar": "existing"})])
    original = state.metadata
    for operation in (
        {"type": "metadata", "path": None, "values": {"model": "wrong"}},
        {"type": "metadata", "path": (["unhashable"],), "values": {}},
        {"type": "metadata", "path": ("scalar", "child"), "values": {}},
        {"type": "metadata", "path": ("new",), "values": {"bad": Uncopyable()}},
        {"type": "metadata", "path": (), "values": "bad"},
    ):
        state.feed([operation])
        assert state.metadata is original
    assert metadata({"bad": Uncopyable()}) == {"type": "metadata", "path": (), "values": {}}


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_malformed_native_usage_and_auxiliary_metadata_keep_the_response(
    monkeypatch, protocol, streaming
):
    incoming = deepcopy(stream_events(protocol))
    for event in incoming:
        roots = [
            event,
            *(event[k] for k in ("message", "response") if isinstance(event.get(k), dict)),
        ]
        for root in roots:
            for name in ("usage", "usage_metadata"):
                if name in root:
                    root[name] = {"malformed": True} if protocol == "anthropic" else "bad"
            for name in ("id", "model", "model_version", "response_id", "created", "created_at"):
                if name in root:
                    root[name] = ["invalid auxiliary value"]
        if protocol == "ollama":
            event.update(prompt_eval_count="bad", eval_count=[])
    upstream = FakeStream(incoming)

    @asynccontextmanager
    async def open_stream(*args):
        try:
            yield upstream
        finally:
            await upstream.aclose()

    provider, model = profiles(protocol)
    async with MyLLM(provider=provider, model=model, stdio_protocol=protocol) as llm:
        monkeypatch.setattr(llm, "_get_client", lambda _: object())
        monkeypatch.setattr(type(llm.codec.transport(protocol)), "upstream", open_stream)
        result = await llm.generate(
            messages=(
                [{"role": "user", "parts": [{"text": "hello"}]}]
                if protocol == "gemini"
                else [{"role": "user", "content": "hello"}]
            ),
            stream=streaming,
        )
        if streaming:
            async with result:
                async for _ in result:
                    pass
            result = result.result
        assert "Hello" in str(result)
        assert llm.get_last_usage() is None
        assert upstream.closed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        {"type": "metadata", "path": None, "values": {}},
        {"type": "metadata", "path": (), "values": "bad"},
        {"type": "metadata"},
        {"type": "usage"},
        {"type": "usage", "usage": {"input": "bad"}},
    ],
)
async def test_auxiliary_fallback_covers_the_encoder_validation_entry(
    monkeypatch, streaming, operation
):
    from my_llm.codec.openai.chat.codec import ChatCompletionsCodec
    from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder

    class CustomEncoder(ChatCompletionsEncoder):
        def _chunks(self, values, profile=None, **parameters):
            batches = super()._chunks(values, profile, **parameters)
            batches[0]["events"].insert(0, deepcopy(operation))
            return batches

    client = FakeClient([chat_event(content="valid answer"), chat_event(finish="stop")])
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kw: client)
    provider, model = profiles("openai_chat")
    async with MyLLM(provider=provider, model=model, stdio_protocol="openai_chat") as llm:
        codec = ChatCompletionsCodec()
        codec.encoder = CustomEncoder()
        llm.codec.codecs["openai_chat"] = codec
        result = await llm.generate(
            messages=[{"role": "user", "content": "hello"}], stream=streaming
        )
        if streaming:
            async with result:
                async for _ in result:
                    pass
            result = result.result
        assert result["content"] == "valid answer"
        assert llm.get_last_usage() is None
        assert client.upstream.closed


def test_explicit_validator_stays_strict_and_unknown_operations_are_not_swallowed():
    from my_llm.middleware.stream import stream_from_wire
    from my_llm.middleware.validation import validate_operations

    invalid_metadata = [
        {"protocol": "example", "events": [{"type": "metadata", "path": None, "values": {}}]}
    ]
    with pytest.raises(MessageValidationError, match="path"):
        validate_operations(invalid_metadata)
    assert stream_from_wire(invalid_metadata)[0].events == invalid_metadata[0]["events"]
    with pytest.raises(MessageValidationError, match="operation"):
        stream_from_wire([{"protocol": "example", "events": [{"type": "unknown"}]}])


def test_offline_decode_also_ignores_malformed_auxiliary_data():
    chunks = [
        {
            "type": "batch",
            "protocol": "openai_chat",
            "events": [
                {"type": "metadata", "path": None, "values": {}},
                {"type": "usage", "usage": "bad"},
                {"type": "block_append", "block": {"type": "text", "text": "valid answer"}},
                {"type": "finish", "reason": "stop"},
            ],
        }
    ]
    rendered = MyCodec().decode(
        chunks=chunks, target_format="openai_chat", input_format="openai_chat"
    )
    assert "valid answer" in str(rendered)


def test_invalid_auxiliary_tails_do_not_invalidate_completion_but_primary_updates_do():
    from my_llm import IncompleteStreamError

    state = StreamState("demo")
    state.feed(
        [
            {"type": "block_append", "block": {"type": "text", "text": "valid answer"}},
            {"type": "finish", "reason": "stop"},
        ]
    )
    assert (
        state.feed(
            [
                {"type": "metadata", "path": None, "values": {}},
                {"type": "usage", "usage": "bad"},
            ]
        )
        == []
    )
    with pytest.raises(IncompleteStreamError, match="after protocol terminal"):
        state.feed([{"type": "block_append", "block": {"type": "text", "text": "invalid tail"}}])
    assert state.finish()[-1]["messages"][0]["content"][0]["text"] == "valid answer"
