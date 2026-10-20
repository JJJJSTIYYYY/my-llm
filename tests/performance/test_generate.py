"""Keep benchmark fixtures, SDK execution and complete timing boundaries honest."""

import json
import socket
from copy import deepcopy

import pytest

from my_llm import MyLLM
from tests.helpers.protocols import PROTOCOLS
from tests.performance.request_chain import probe
from tests.performance.runner import benchmark, measure
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.validation import validate_exchange, validate_workload
from tests.performance.workloads import Size, build, controls, profiles

SMALL = Size(
    rounds=1,
    tools=2,
    parallel_calls=2,
    text_chars=64,
    output_chars=128,
    fragment_chars=19,
    argument_items=1,
    schema_depth=2,
)
CASES = [
    *[(protocol, protocol, True) for protocol in PROTOCOLS],
    *[(upstream, stdio, False) for upstream in PROTOCOLS for stdio in PROTOCOLS],
]


@pytest.mark.parametrize("upstream,stdio,native", CASES)
@pytest.mark.parametrize("stream", [False, True])
async def test_complex_workload_uses_real_sdk_and_completes_without_network(
    monkeypatch, upstream, stdio, native, stream
):
    def network_forbidden(*args, **kwargs):
        pytest.fail("the performance harness attempted a socket connection")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    workload = build(upstream, stdio, SMALL, native=native)
    before = deepcopy((workload.messages, workload.tools, workload.events))
    report = await benchmark(workload, stream=stream, iterations=2, warmup=1, wire_chunk_bytes=37)

    assert report["warm"]["iterations"] == 2
    assert len(report["warm"]["samples"]) == 2
    assert report["warm"]["p95_ms"] >= report["warm"]["median_ms"] > 0
    assert report["request_bytes"] > 0 and report["response_bytes"] > 0
    assert report["http_chunks"] > report["upstream_events"]
    assert (workload.messages, workload.tools, workload.events) == before
    for sample in [report["cold"], *report["warm"]["samples"]]:
        if stream:
            assert 0 < sample["prepare_ms"] < sample["first_event_ms"] <= sample["total_ms"]
            assert sample["output_events"] > 0
        else:
            assert sample["prepare_ms"] is sample["first_event_ms"] is None
    json.dumps(report)


@pytest.mark.parametrize("stream", [False, True])
async def test_ollama_base64_images_survive_input_and_request_validation(stream):
    workload = build("ollama", "ollama", SMALL, native=True)
    validate_workload(workload)
    frames = wire_frames("ollama", workload.events, 37)
    with LocalSDK("ollama", frames) as sdk:
        provider, model = profiles("ollama")
        async with MyLLM(provider=provider, model=model, stdio_protocol="ollama") as llm:
            _, result = await measure(llm, workload, stream=stream, parameters=controls("ollama"))
            validate_exchange(workload, "ollama", result, sdk)
        request = json.loads(sdk.requests[-1].content)
        assert request["messages"][0]["role"] == "system"
        images = [image for message in request["messages"] for image in message.get("images", [])]
        expected = [image for message in workload.messages for image in message.get("images", [])]
        assert images == expected


@pytest.mark.parametrize("upstream", PROTOCOLS)
async def test_chain_probe_measures_memory_cache_and_actual_overlap(monkeypatch, upstream):
    def network_forbidden(*args, **kwargs):
        pytest.fail("the request-chain benchmark attempted a socket connection")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    result = await probe(build(upstream, "openai_chat", SMALL, native=False), True, 4, 1)
    assert result["memory"]["peak_bytes"] > 0
    assert result["concurrent"]["peak_bytes"] > 0
    assert result["concurrent"]["peak_active"] == 4
    assert len(result["concurrent"]["samples_ms"]) == 1
    for name, before in result["cache_before"].items():
        after = result["cache_after"][name]
        assert after["currsize"] == before["currsize"]
        assert after["misses"] == before["misses"]
