"""Diagnose collected-response tails with GC enabled, using the existing benchmark.

Run: python -m tests.performance.gc_diagnostic /tmp/responses-gc.json
The callbacks add overhead: this is a diagnostic, not the primary timing run.
"""

import argparse
import asyncio
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

from tests.performance import middleware
from tests.performance.fingerprint import harness, source
from tests.performance.request_chain import caches
from tests.performance.workloads import Size, build


async def run(
    destination, *, protocol="openai_responses", stdio="native", iterations=150, warmup=2
):
    samples = []
    original = middleware.measure
    workload = build(protocol, protocol, Size(), native=True)
    cache_before = None

    async def measure(*args, **kwargs):
        nonlocal cache_before
        if len(samples) == warmup * 2:
            cache_before = caches()
        started, pauses = {}, []

        def observe(phase, info):
            generation = info["generation"]
            if phase == "start":
                started[generation] = perf_counter_ns()
            elif generation in started:
                pauses.append(
                    {
                        "generation": generation,
                        "ms": (perf_counter_ns() - started.pop(generation)) / 1e6,
                    }
                )

        gc.callbacks.append(observe)
        try:
            sample, result = await original(*args, **kwargs)
        finally:
            gc.callbacks.remove(observe)
        samples.append(
            {
                "stdio": args[1].stdio,
                "total_ms": sample.total_ms,
                "cpu_ms": sample.cpu_ms,
                "gc_ms": sum(pause["ms"] for pause in pauses),
                "collections": pauses,
            }
        )
        return sample, result

    middleware.measure = measure
    try:
        report, _ = await middleware.paired(
            workload,
            streaming=False,
            iterations=iterations,
            warmup=warmup,
            chunk_bytes=1024,
        )
    finally:
        middleware.measure = original
    measured = samples[warmup * 2 :]
    selected = "middleware" if stdio == "middleware" else protocol
    report.update(
        stdio=selected,
        warm=report[stdio],
        gc_samples=[sample for sample in measured if sample["stdio"] == selected],
        gc_by_stdio={
            name: [sample for sample in measured if sample["stdio"] == name]
            for name in (protocol, "middleware")
        },
        gc_enabled=gc.isenabled(),
        gc_threshold=gc.get_threshold(),
        cache_before=cache_before,
        cache_after=caches(),
        source_sha256=source(),
        harness_sha256=harness(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        size=asdict(workload.size),
        warmup=warmup,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument(
        "--protocol",
        choices=("openai_chat", "openai_responses", "anthropic", "gemini", "ollama"),
        default="openai_responses",
    )
    parser.add_argument("--stdio", choices=("native", "middleware"), default="native")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=2)
    options = parser.parse_args()
    if options.iterations < 1 or options.warmup < 0:
        parser.error("iterations must be positive; warmup must be non-negative")
    asyncio.run(
        run(
            options.json,
            protocol=options.protocol,
            stdio=options.stdio,
            iterations=options.iterations,
            warmup=options.warmup,
        )
    )
