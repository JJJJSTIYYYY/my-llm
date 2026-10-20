"""Comparable request-chain timing, separate memory/cache probes and real overlap.

Run with python -m tests.performance.request_chain --json /tmp/chain.json.
PYTHONPATH can select a saved source tree without changing this harness.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import platform
import subprocess
import sys
import tracemalloc
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import my_llm
from my_llm import MyLLM
from my_llm.codec.shared import transport as schema
from tests.helpers.protocols import PROTOCOLS
from tests.performance.runner import benchmark, measure
from tests.performance.transport import Body, LocalSDK, wire_frames
from tests.performance.validation import validate_exchange, validate_result
from tests.performance.workloads import Size, build, controls, profiles


def caches():
    return {
        name: getattr(schema, name).cache_info()._asdict()
        for name in (
            "fields",
            "_cached_adapter",
            "parameter_types",
            "_compiled_plan",
            "_declared_path",
        )
    }


async def probe(workload, stream, concurrency, iterations):
    frames = wire_frames(workload.protocol, workload.events, 1024)
    provider, model = profiles(workload.protocol)
    parameters = controls(workload.protocol)
    with LocalSDK(workload.protocol, frames) as sdk:
        async with MyLLM(provider=provider, model=model, stdio_protocol=workload.stdio) as llm:
            await measure(llm, workload, stream=stream, parameters=parameters)
            sdk.release_samples()
            before_cache = caches()
            gc.collect()
            tracemalloc.start()
            try:
                _, result = await measure(llm, workload, stream=stream, parameters=parameters)
                current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            validate_exchange(workload, workload.stdio, result, sdk)
            after_cache = caches()
            sdk.release_samples()
            # Every response waits for the other requests before yielding bytes.
            # This proves overlap even though the in-memory SDK has no I/O wait.
            barrier = asyncio.Barrier(concurrency)
            active = 0
            peak_active = 0

            class ConcurrentBody(Body):
                async def __aiter__(self):
                    nonlocal active, peak_active
                    active += 1
                    peak_active = max(peak_active, active)
                    await barrier.wait()
                    async for frame in super().__aiter__():
                        yield frame

                async def aclose(self):
                    nonlocal active
                    if not self.closed:
                        active -= 1
                    await super().aclose()

            def respond(request):
                response = original(request)
                body = ConcurrentBody(frames)
                sdk.bodies[-1] = body
                response.stream = body
                return response

            original = sdk.respond
            sdk.respond = respond
            # MockTransport captured the old bound method; switch its handler.
            # Build a fresh instance so every SDK binds the concurrent handler.
        sdk.release_samples()
        samples = []
        async with MyLLM(provider=provider, model=model, stdio_protocol=workload.stdio) as llm:
            for index in range(iterations + 3):
                sdk.release_samples()
                memory_probe = index == iterations + 2
                if memory_probe:
                    gc.collect()
                    tracemalloc.start()
                start = perf_counter_ns()
                try:
                    async with asyncio.timeout(60):
                        results = await asyncio.gather(
                            *(
                                measure(
                                    llm,
                                    workload,
                                    stream=stream,
                                    parameters={
                                        **parameters,
                                        "extra_headers": {
                                            "x-benchmark": "request",
                                            "x-request": str(request_index),
                                        },
                                    },
                                )
                                for request_index in range(concurrency)
                            )
                        )
                    elapsed = (perf_counter_ns() - start) / 1_000_000
                    if memory_probe:
                        concurrent_current, concurrent_peak = tracemalloc.get_traced_memory()
                finally:
                    if memory_probe:
                        tracemalloc.stop()
                if index >= 2 and not memory_probe:
                    samples.append(elapsed)
                for _, result in results:
                    validate_result(workload, workload.stdio, result)
                assert all(body.closed for body in sdk.bodies)
                assert {request.headers["x-request"] for request in sdk.requests} == {
                    str(index) for index in range(concurrency)
                }
            client_count = sdk.created_clients - 1
    assert peak_active == concurrency
    assert client_count == (concurrency if workload.protocol == "ollama" else 1)
    return {
        "memory": {"current_bytes": current, "peak_bytes": peak},
        "cache_before": before_cache,
        "cache_after": after_cache,
        "concurrent": {
            "width": concurrency,
            "peak_active": peak_active,
            "clients": client_count,
            "batch_median_ms": median(samples),
            "samples_ms": samples,
            "current_bytes": concurrent_current,
            "peak_bytes": concurrent_peak,
        },
    }


async def run(options):
    size = Size()
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(
            b"".join(
                path.read_bytes() for path in sorted(Path(my_llm.__file__).parent.rglob("*.py"))
            )
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: version(name)
            for name in ("openai", "anthropic", "google-genai", "ollama", "httpx", "pydantic")
        },
        "memory_definition": "separate tracemalloc probes after warmup and gc.collect; excludes fixtures and client construction; includes returned result and captured HTTP observations; not RSS",
        "timing_definition": "real SDK over immediate in-memory HTTP; serial benchmark boundary unchanged; concurrent batches await all complete results, two warmup batches; default GC; no tracing during timing",
        "size": asdict(size),
        "iterations": options.iterations,
        "warmup": 2,
        "concurrency": options.concurrency,
        "cases": [],
    }
    for native in (True, False):
        for upstream in PROTOCOLS:
            for stdio in [upstream] if native else PROTOCOLS:
                workload = build(upstream, stdio, size, native=native)
                for stream in (False, True):
                    timing = await benchmark(
                        workload,
                        stream=stream,
                        iterations=options.iterations,
                        warmup=2,
                        wire_chunk_bytes=1024,
                    )
                    timing.update(
                        await probe(workload, stream, options.concurrency, options.iterations)
                    )
                    report["cases"].append(timing)
                    print(
                        timing["suite"],
                        upstream,
                        stdio,
                        stream,
                        round(timing["warm"]["median_ms"], 3),
                        flush=True,
                    )
                    options.json.parent.mkdir(parents=True, exist_ok=True)
                    options.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=4)
    options = parser.parse_args()
    if options.iterations < 1 or options.concurrency < 2:
        parser.error("iterations must be positive; concurrency must be >=2")
    asyncio.run(run(options))
