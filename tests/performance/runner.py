"""Measure the public generate call through complete result retrieval."""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import ceil
from statistics import median
from time import perf_counter_ns, process_time_ns

from my_llm import MyLLM
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.validation import validate_exchange, validate_result, validate_workload
from tests.performance.workloads import Workload, controls, profiles


@dataclass(frozen=True)
class Sample:
    total_ms: float
    cpu_ms: float
    prepare_ms: float | None
    first_event_ms: float | None
    output_events: int


async def measure(llm: MyLLM, workload: Workload, *, stream: bool, parameters: dict):
    """Time only generate + full consumption + result copy, including response close."""
    first_event = None
    prepared = None
    count = 0
    cpu_start = process_time_ns()
    start = perf_counter_ns()
    result = await llm.generate(
        messages=workload.messages, tools=workload.tools, stream=stream, **parameters
    )
    if stream:
        prepared = perf_counter_ns()
        async with result:
            async for _ in result:
                if first_event is None:
                    first_event = perf_counter_ns()
                count += 1
        # Include public result retrieval; each runtime owns its accessor semantics.
        result = result.result
    end = perf_counter_ns()
    cpu_end = process_time_ns()
    return Sample(
        total_ms=(end - start) / 1_000_000,
        cpu_ms=(cpu_end - cpu_start) / 1_000_000,
        prepare_ms=(prepared - start) / 1_000_000 if prepared is not None else None,
        first_event_ms=(first_event - start) / 1_000_000 if first_event is not None else None,
        output_events=count,
    ), result


def summarize(samples: list[Sample]):
    values = sorted(sample.total_ms for sample in samples)
    return {
        "iterations": len(samples),
        "median_ms": median(values),
        "p95_ms": values[ceil(len(values) * 0.95) - 1],
        "min_ms": values[0],
        "max_ms": values[-1],
        "cpu_median_ms": median(sample.cpu_ms for sample in samples),
        "requests_per_second": 1000 * len(values) / sum(values),
        "prepare_median_ms": median(sample.prepare_ms for sample in samples)
        if samples[0].prepare_ms is not None
        else None,
        "first_event_median_ms": median(sample.first_event_ms for sample in samples)
        if samples[0].first_event_ms is not None
        else None,
        "output_events": samples[0].output_events,
        "samples": [asdict(sample) for sample in samples],
    }


async def benchmark(
    workload: Workload,
    *,
    stream: bool,
    iterations: int,
    warmup: int,
    wire_chunk_bytes: int,
):
    if iterations < 1 or warmup < 0 or wire_chunk_bytes < 1:
        raise ValueError("iterations/chunk bytes must be positive; warmup must be non-negative")
    stdio = workload.stdio
    validate_workload(workload)
    frames = wire_frames(workload.protocol, workload.events, wire_chunk_bytes)
    provider, model = profiles(workload.protocol)
    parameters = controls(workload.protocol)
    with LocalSDK(workload.protocol, frames) as sdk:
        # Validate a full exchange before measuring. Cold below means a new
        # instance's lazy SDK client, not a fresh Python process/schema cache.
        async with MyLLM(provider=provider, model=model, stdio_protocol=stdio) as llm:
            _, result = await measure(llm, workload, stream=stream, parameters=parameters)
            validate_exchange(workload, stdio, result, sdk)
            request_bytes = len(sdk.requests[-1].content)
            request_sha256 = hashlib.sha256(sdk.requests[-1].content).hexdigest()
        sdk.release_samples()
        async with MyLLM(provider=provider, model=model, stdio_protocol=stdio) as llm:
            cold, result = await measure(llm, workload, stream=stream, parameters=parameters)
            validate_result(workload, stdio, result)
            for _ in range(warmup):
                _, result = await measure(llm, workload, stream=stream, parameters=parameters)
                validate_result(workload, stdio, result)
                sdk.release_samples()
            samples = []
            for _ in range(iterations):
                sdk.release_samples()
                sample, result = await measure(llm, workload, stream=stream, parameters=parameters)
                samples.append(sample)
                validate_exchange(workload, stdio, result, sdk)
        assert sdk.created_clients == 2, "requests must reuse their instance's SDK client"
        assert all(client.is_closed for client in sdk.http_clients)
    return {
        "suite": "native" if workload.native else "cross",
        "upstream": workload.protocol,
        "stdio": stdio,
        "stream": stream,
        "request_bytes": request_bytes,
        "request_sha256": request_sha256,
        "response_bytes": sum(map(len, frames)),
        "response_sha256": hashlib.sha256(b"".join(frames)).hexdigest(),
        "workload_sha256": hashlib.sha256(
            json.dumps(asdict(workload), sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest(),
        "upstream_events": len(workload.events),
        "http_chunks": len(frames),
        "input_messages": len(workload.messages),
        "tool_definitions": workload.size.tools,
        "parallel_calls": workload.size.parallel_calls,
        "cold": asdict(cold),
        "warm": summarize(samples),
    }
