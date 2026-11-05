"""Exercise live example controls and repairs at the official SDK's HTTP boundary."""

import importlib
import json
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from tests.helpers.protocols import chat_event, response_completed, stream_events
from tests.helpers.transport import sse

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PROTOCOLS = ("openai_chat", "openai_responses", "anthropic")
STDIO = (*PROTOCOLS, "middleware")
CASES = (
    "reasoning_missing",
    "reasoning_null",
    "reasoning_empty",
    "missing_tool_result",
    "orphan_tool_result",
    "tool_result_images",
    "tool_result_image_placeholders",
)
MATRIX = [
    (upstream, stdio, case)
    for upstream in PROTOCOLS
    for stdio in STDIO
    for case in CASES
    if (
        case in ("missing_tool_result", "orphan_tool_result")
        or upstream == "openai_chat"
        and (
            case == "reasoning_missing"
            or stdio == "openai_chat"
            and case in ("reasoning_null", "reasoning_empty")
            or stdio in ("openai_responses", "anthropic")
            and case.startswith("tool_result_image")
        )
    )
]


def replay_events(protocol, vendor, value=None, *, empty=False):
    thought = "" if empty else "Original thought\nsecond line"
    if protocol == "openai_chat":
        if value is not None:
            return [chat_event(content=value), chat_event(finish="stop")]
        first = chat_event()
        first["choices"][0]["delta"][
            "reasoning" if vendor == "sensenova_e2e" else "reasoning_content"
        ] = thought
        return [
            first,
            chat_event(
                calls=[
                    {
                        "index": 0,
                        "id": "call_original",
                        "type": "function",
                        "function": {"name": "audit_value", "arguments": "{}"},
                    }
                ]
            ),
            chat_event(finish="tool_calls"),
        ]
    if protocol == "openai_responses":
        items = (
            [
                {
                    "type": "message",
                    "id": "msg_final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": value, "annotations": [], "logprobs": []}
                    ],
                }
            ]
            if value is not None
            else [
                {
                    "type": "reasoning",
                    "id": "rs_original",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": thought}],
                },
                {
                    "type": "function_call",
                    "id": "fc_original",
                    "status": "completed",
                    "call_id": "call_original",
                    "name": "audit_value",
                    "arguments": "{}",
                },
            ]
        )
        return [response_completed(items)]
    content = (
        [{"type": "text", "text": value}]
        if value is not None
        else [
            {"type": "thinking", "thinking": thought, "signature": "original-signature"},
            {"type": "tool_use", "id": "call_original", "name": "audit_value", "input": {}},
        ]
    )
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_original",
                "type": "message",
                "role": "assistant",
                "model": "local-example",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        }
    ]
    for index, part in enumerate(content):
        events.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": part},
                {"type": "content_block_stop", "index": index},
            ]
        )
    return [
        *events,
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn" if value is not None else "tool_use",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


@pytest.mark.parametrize("vendor", ["deepseek_reasoning_e2e", "sensenova_e2e"])
@pytest.mark.parametrize("upstream", PROTOCOLS)
@pytest.mark.parametrize("middleware", [False, True])
async def test_reasoning_replay_matches_original_response_and_actual_http_body(
    monkeypatch, vendor, upstream, middleware
):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module(vendor)
    inputs = importlib.import_module("e2e_inputs")
    requests, clients = [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        calls, results = calls_and_results(upstream, body)
        value = None
        if calls:
            assert len(results) == 1 and calls[0] == results[0][0]
            value = results[0][1]
            if isinstance(value, list):
                value = "".join(part["text"] for part in value)
            if upstream == "openai_chat":
                assert body["messages"][1]["reasoning_content"] == "Original thought\nsecond line"
        return httpx.Response(
            200,
            text=sse(
                replay_events(upstream, vendor, value),
                anthropic=upstream == "anthropic",
                done=upstream == "openai_chat",
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    def observed_client(**kwargs):
        client = httpx.AsyncClient(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(inputs, "AsyncClient", observed_client)
    result = await inputs.replay_scenario(
        example.provider(upstream, "local-example-key", reasoning=True),
        example.model_profile("local-example", "reasoning_replay"),
        "middleware" if middleware else upstream,
    )
    assert result["requests"] == len(requests) == 4
    assert len(result["roundtrips"]) == 2
    assert all(
        row["returned_exact"]
        and row["prepared_exact"]
        and row["wire_exact"]
        and row["nonempty_replay"]
        for row in result["roundtrips"]
    )
    assert all(client.is_closed for client in clients)
    assert "Original thought" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["empty_source", "feedback_disabled"])
async def test_replay_rejects_empty_or_dropped_reasoning_before_a_second_api_call(
    monkeypatch, failure
):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module("deepseek_reasoning_e2e")
    inputs = importlib.import_module("e2e_inputs")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            text=sse(
                replay_events(
                    "openai_chat", "deepseek_reasoning_e2e", empty=failure == "empty_source"
                )
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr(
        inputs,
        "AsyncClient",
        lambda **kwargs: httpx.AsyncClient(**kwargs, transport=httpx.MockTransport(handler)),
    )
    model = example.model_profile("local-example", "reasoning_replay")
    if failure == "feedback_disabled":
        model.should_feedback_reasoning_item.default_value = "none"
    evidence = {}
    with pytest.raises(AssertionError, match="no nonempty reasoning|request preparation changed"):
        await inputs.replay_scenario(
            example.provider("openai_chat", "local-example-key", reasoning=True),
            model,
            "openai_chat",
            evidence=evidence,
        )
    assert evidence["requests"] == len(requests) == 1


def calls_and_results(upstream, body):
    if upstream == "openai_chat":
        messages = body["messages"]
        calls = [c["id"] for m in messages for c in m.get("tool_calls", [])]
        results = [(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"]
    elif upstream == "openai_responses":
        calls = [m["call_id"] for m in body["input"] if m.get("type") == "function_call"]
        results = [
            (m["call_id"], m["output"])
            for m in body["input"]
            if m.get("type") == "function_call_output"
        ]
    else:
        parts = [p for m in body["messages"] for p in m["content"] if isinstance(p, dict)]
        calls = [p["id"] for p in parts if p["type"] == "tool_use"]
        results = [(p["tool_use_id"], p["content"]) for p in parts if p["type"] == "tool_result"]
    return calls, results


@pytest.mark.parametrize("vendor", ["sensenova_e2e", "deepseek_reasoning_e2e"])
@pytest.mark.parametrize("upstream,stdio,case", MATRIX)
async def test_input_matrix_sends_invalid_controls_and_profile_repaired_history(
    monkeypatch, vendor, upstream, stdio, case
):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module(vendor)
    inputs = importlib.import_module("e2e_inputs")
    requests, clients = [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "local-example" and body["stream"] is True
        token_field = "max_output_tokens" if upstream == "openai_responses" else "max_tokens"
        assert body[token_field] == (2048 if case in inputs.REASONING_CASES else 1024)
        calls, results = calls_and_results(upstream, body)
        assert len(calls) == 1
        if calls[0].startswith("call_raw_"):
            if case in ("reasoning_missing", "reasoning_null", "reasoning_empty"):
                assistant = body["messages"][1]
                if case == "reasoning_missing":
                    assert "reasoning_content" not in assistant
                else:
                    assert assistant["reasoning_content"] == (
                        None if case == "reasoning_null" else ""
                    )
            elif case == "missing_tool_result":
                assert not results
            else:
                assert len(results) == 2 and inputs.ORPHAN_TEXT in json.dumps(body)
            if case != "reasoning_empty":
                return httpx.Response(
                    400, json={"error": {"type": "invalid_request_error", "message": case}}
                )
        else:
            assert len(results) == 1 and results[0][0] == calls[0]
            assert inputs.ORPHAN_TEXT not in json.dumps(body)
            if case == "missing_tool_result":
                assert results[0][1] in ("", [])
            if upstream == "openai_chat":
                assistant = next(m for m in body["messages"] if m.get("tool_calls"))
                assert assistant["reasoning_content"] == ""
            if case in inputs.IMAGE_CASES:
                assert results[0][1].startswith(inputs.RESULT_TEXT)
                reference = results[0][1][len(inputs.RESULT_TEXT) :]
                assert reference.startswith("[Image: image_") and reference.endswith("]")
                last = body["messages"][-1]
                assert last["role"] == "user"
                if case == "tool_result_images":
                    assert last["content"][0] == {"type": "text", "text": reference + "\n"}
                    assert last["content"][1]["image_url"]["url"] == inputs.DATA_URL
                else:
                    assert last["content"] == reference + "\n[Unsupported Image]\n"
        events = json.loads(json.dumps(stream_events(upstream)).replace('"Hello"', '"E2E_OK"'))
        return httpx.Response(
            200,
            text=sse(events, anthropic=upstream == "anthropic", done=upstream == "openai_chat"),
            headers={"Content-Type": "text/event-stream"},
        )

    def factory(**kwargs):
        sdk = AsyncAnthropic if upstream == "anthropic" else AsyncOpenAI
        client = sdk(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    target = {
        "openai_chat": "my_llm.codec.openai.chat.transport.AsyncOpenAI",
        "openai_responses": "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        "anthropic": "my_llm.codec.anthropic.transport.AsyncAnthropic",
    }[upstream]
    monkeypatch.setattr(target, factory)
    result = await inputs.input_scenario(
        example.provider(upstream, "local-example-key", reasoning=case in inputs.REASONING_CASES),
        example.model_profile("local-example", case),
        stdio,
        case,
    )
    assert result["requests"] == len(requests) == (2 if case in inputs.IMAGE_CASES else 3)
    assert [row["stream"] for row in result["completions"]] == [False, True]
    assert all(row["text"] == "E2E_OK" for row in result["completions"])
    expected = (
        "local_semantic_loss"
        if case in inputs.IMAGE_CASES
        else "accepted"
        if case == "reasoning_empty"
        else "rejected"
    )
    assert result["raw_api"]["outcome"] == expected
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
async def test_raw_control_does_not_treat_operational_errors_as_history_rejections(
    monkeypatch, status
):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module("deepseek_reasoning_e2e")
    inputs = importlib.import_module("e2e_inputs")
    from my_llm import MyLLM

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        status, json={"error": {"message": "operational error"}}
                    )
                )
            ),
        )

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=example.provider("openai_chat", "local-example-key"),
        model=example.model_profile("local-example"),
        stdio_protocol="openai_chat",
    ) as llm:
        with pytest.raises(Exception) as caught:
            await inputs.raw_control(llm, "missing_tool_result", request_interval=0)
        assert caught.value.status_code == status


async def test_live_report_keeps_control_evidence_and_redacts_failed_requests(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module("deepseek_reasoning_e2e")
    from types import SimpleNamespace

    key = "local-example-secret"
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        raw = body["messages"][1]["tool_calls"][0]["id"].startswith("call_raw_")
        message = f"tool result missing; {key}" if raw else f"rpm exhausted; {key}"
        return httpx.Response(400 if raw else 429, json={"error": {"message": message}})

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    report_path = tmp_path / "results.json"
    args = SimpleNamespace(
        model="local-example",
        upstream=["openai_chat"],
        stdio=["openai_chat"],
        case=["missing_tool_result"],
        report=report_path,
        retry_failed=False,
        request_interval=0,
        semantic_loss_protocol="ignore",
        validate=False,
    )
    assert await example.run(
        args,
        key,
        provider_factory=example.provider,
        model_factory=example.model_profile,
        endpoint="https://api.deepseek.com",
    )
    saved = report_path.read_text()
    assert key not in saved and key not in capsys.readouterr().out
    row = json.loads(saved)["results"][0]
    assert row["raw_api"]["outcome"] == "rejected"
    assert row["prepared_request"]["ids_paired"]
    assert row["status"] == "failed" and row["error_type"] == "RateLimitError"
    assert row["requests"] == len(requests) == 2


@pytest.mark.parametrize("field,value", [("mode", "offline"), ("endpoint", "https://other.test")])
async def test_retry_does_not_mix_offline_or_other_provider_evidence(
    monkeypatch, tmp_path, field, value
):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module("deepseek_reasoning_e2e")
    from types import SimpleNamespace

    path = tmp_path / "results.json"
    report = {
        "model": "local-example",
        "mode": "live",
        "endpoint": "https://api.deepseek.com",
        "results": [],
    }
    report[field] = value
    path.write_text(json.dumps(report))
    args = SimpleNamespace(model="local-example", report=path, retry_failed=True, validate=False)
    with pytest.raises(ValueError, match=f"retry {field}"):
        await example.run(args, "local-example-key", endpoint="https://api.deepseek.com")
