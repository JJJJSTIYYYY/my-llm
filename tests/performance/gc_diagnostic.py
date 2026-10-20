"""Diagnose Responses collect tails with GC enabled, using the existing benchmark.

Run: python -m tests.performance.gc_diagnostic /tmp/responses-gc.json
The callbacks add overhead: this is a diagnostic, not the primary timing run.
"""

import argparse
import asyncio
import gc
import json
from pathlib import Path
from time import perf_counter_ns

from tests.performance import runner
from tests.performance.workloads import Size, build


async def run(destination):
    samples = []
    original = runner.measure

    async def measure(*args, **kwargs):
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
                "total_ms": sample.total_ms,
                "gc_ms": sum(pause["ms"] for pause in pauses),
                "collections": pauses,
            }
        )
        return sample, result

    runner.measure = measure
    try:
        report = await runner.benchmark(
            build("openai_responses", "openai_responses", Size(), native=True),
            stream=False,
            iterations=150,
            warmup=2,
            wire_chunk_bytes=1024,
        )
    finally:
        runner.measure = original
    report.update(gc_samples=samples[-150:], gc_enabled=gc.isenabled())
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    asyncio.run(run(parser.parse_args().json))
