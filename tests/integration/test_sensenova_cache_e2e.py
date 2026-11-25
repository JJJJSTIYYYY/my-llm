"""Validate the live cache probe using real SDKs and local HTTP responses."""

import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.helpers.protocols import PROTOCOLS, stream_events
from tests.performance.transport import Body, wire_frames
from tests.performance.workloads import Size, response_events


@pytest.fixture
def probe(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "examples"))
    return runpy.run_path(str(root / "examples/sensenova_cache_e2e.py"))


@dataclass
class Options:
    model: str = "local-cache-probe"
    records: int = 2
    request_interval: float = 0


@pytest.mark.parametrize(
    "upstream,stdio,first_answer",
    [(upstream, stdio, "READY") for upstream in PROTOCOLS for stdio in [*PROTOCOLS, "middleware"]]
    + [("openai_responses", "middleware", "OTHER")],
)
async def test_live_probe_collects_real_sdk_usage_and_replays_result(
    probe, monkeypatch, upstream, stdio, first_answer
):
    requests = []
    bodies = []

    async def respond(transport, request):
        index = len(requests)
        requests.append(request)
        if index == 2:
            events = response_events(
                upstream,
                Size(fragment_chars=3),
                False,
                "",
                "",
                [{"id": "call_add", "name": "add", "input": {"a": 19, "b": 23}}],
            )
        else:
            events = stream_events(upstream)
            events = json.loads(
                json.dumps(events).replace("Hello", (first_answer, "CACHE_OK", "", "42")[index])
            )
        if upstream == "openai_chat":
            events[-1]["usage"] = {
                "prompt_tokens": 4096,
                "completion_tokens": 2,
                "total_tokens": 4098,
                "prompt_tokens_details": {"cached_tokens": 3072},
            }
        elif upstream == "openai_responses":
            events[-1]["response"]["usage"] = {
                "input_tokens": 4096,
                "output_tokens": 2,
                "total_tokens": 4098,
                "input_tokens_details": {"cached_tokens": 3072},
            }
        else:
            events[0]["message"]["usage"] = {
                "input_tokens": 1024,
                "output_tokens": 0,
                "cache_read_input_tokens": 3072,
            }
        body = Body(wire_frames(upstream, events, 17))
        bodies.append(body)
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    row = {"index": 0, "rounds": []}
    await probe["session"](upstream, stdio, "local-test-key", Options(), row)
    assert row["status"] == ("passed" if first_answer == "READY" else "failed"), row
    assert row["measurement_complete"] is True
    assert len(requests) == len(row["rounds"]) == 4
    assert all(
        turn["cached_tokens"] == 3072 and turn["input_tokens"] == 4096 for turn in row["rounds"]
    )
    assert all(turn["prefix_equal"] for turn in row["rounds"][1:])
    assert all(body.closed for body in bodies)


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({}, (None, None)),
        ({"prompt_tokens": 10}, (10, None)),
        ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}, (10, 0)),
    ],
)
def test_unreported_cache_counters_are_not_counted_as_zero(probe, usage, expected):
    assert probe["counters"]("openai_chat", usage) == expected


async def test_retry_preserves_successes_and_failed_attempts(probe, monkeypatch, tmp_path):
    passed = {"upstream": "openai_chat", "stdio": "middleware", "status": "passed", "rounds": []}
    failed = {
        "upstream": "openai_responses",
        "stdio": "middleware",
        "status": "failed",
        "rounds": [],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"model": "local", "records": 2, "results": [passed, failed]}))
    called = []

    async def session(upstream, stdio, key, args, row):
        called.append((upstream, stdio))
        row["status"] = "passed"

    monkeypatch.setitem(probe["run"].__globals__, "session", session)
    args = SimpleNamespace(
        model="local",
        records=2,
        request_interval=0,
        report=path,
        retry_failed=True,
        upstream=["openai_chat", "openai_responses"],
        stdio=["middleware"],
    )
    report = await probe["run"](args, "local-test-key")
    assert called == [("openai_responses", "middleware")]
    assert report["results"][0] == passed
    assert report["results"][1]["index"] == 7
    assert report["results"][1]["previous_attempts"] == [failed]
    assert "local-test-key" not in path.read_text()
