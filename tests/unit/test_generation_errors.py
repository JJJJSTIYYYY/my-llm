"""Generation failures retain their meaning across rendering and SDK validation."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from my_llm import ConfigurationError, MyLLM, ResponseFailedError
from tests.helpers.transport import FakeClient
from tests.performance.workloads import profiles


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("render_failure", [None, "signature", "hook"])
async def test_partial_rendering_preserves_the_original_stream_error(
    monkeypatch, streaming, render_failure
):
    error = ResponseFailedError("original provider failure", details={"code": "offline"})
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "partial answer"},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call", "name": "f", "input": {}},
        },
    ]
    if render_failure is not None:
        # The pending tool defers this block until EOF, so both return modes
        # first attempt to render its signature while handling the source error.
        events.append(
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "thinking",
                    "thinking": "working",
                    "signature": "offline-signature",
                },
            }
        )
    client = FakeClient(events, failure=error)
    monkeypatch.setattr("my_llm.codec.anthropic.transport.AsyncAnthropic", lambda **kw: client)

    def fail_hook(draft, losses, context):
        raise LookupError("partial hook failed")

    provider, model = profiles("anthropic")
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="openai_chat",
        semantic_loss_protocol="hook" if render_failure == "hook" else "raise",
        output_semantic_loss_hook=fail_hook if render_failure == "hook" else None,
    ) as llm:
        with pytest.raises(ResponseFailedError) as caught:
            result = await llm.generate(
                messages=[{"role": "user", "content": "hello"}], stream=streaming
            )
            if streaming:
                async with result:
                    async for _ in result:
                        pass
        assert caught.value is error
        assert error.details == {"code": "offline"}
        if render_failure is None:
            assert error.partial_message == {"role": "assistant", "content": "partial answer"}
        else:
            assert error.partial_message is None
            expected = (
                "LookupError('partial hook failed')"
                if render_failure == "hook"
                else "SemanticLossError("
            )
            assert any(expected in note for note in error.__notes__)
        assert client.upstream.closed and not client.closed
        assert llm.get_last_usage() is None
        if streaming:
            assert result.result is None
    assert client.closed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "body", [{"temperature": "invalid"}, {"thinking_config": {"thinking_budget": "invalid"}}]
)
async def test_invalid_gemini_config_is_a_configuration_error_before_client_creation(
    body, streaming
):
    original = deepcopy(body)
    provider, model = profiles("gemini")
    async with MyLLM(provider=provider, model=model, stdio_protocol="openai_chat") as llm:
        with pytest.raises(ConfigurationError, match="invalid Gemini config") as caught:
            await llm.generate(
                messages=[{"role": "user", "content": "hello"}],
                extra_body=body,
                stream=streaming,
            )
        assert isinstance(caught.value.__cause__, ValidationError)
        assert llm._clients == {}
    assert body == original
