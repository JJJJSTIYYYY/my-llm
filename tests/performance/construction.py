"""Measure instance preparation separately from generate and SDK construction."""

import argparse
import asyncio
import gc
import hashlib
import json
import platform
import sys
import tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from my_llm import MyLLM
from tests.helpers.protocols import PROTOCOLS
from tests.performance.fingerprint import source
from tests.performance.transport import LocalSDK
from tests.performance.workloads import profiles


async def benchmark(protocol, iterations):
    provider, model = profiles(protocol)
    samples = []
    with LocalSDK(protocol, ()) as sdk:
        for index in range(iterations + 10):
            start = perf_counter_ns()
            llm = MyLLM(provider=provider, model=model, stdio_protocol=protocol)
            elapsed = (perf_counter_ns() - start) / 1_000_000
            await llm.aclose()
            if index >= 10:
                samples.append(elapsed)
        assert sdk.created_clients == 0
        gc.collect()
        tracemalloc.start()
        try:
            retained = [
                MyLLM(provider=provider, model=model, stdio_protocol=protocol) for _ in range(100)
            ]
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        for llm in retained:
            await llm.aclose()
        assert sdk.created_clients == 0
    return {
        "protocol": protocol,
        "median_ms": median(samples),
        "samples_ms": samples,
        "memory": {"retained_instances": 100, "current_bytes": current, "peak_bytes": peak},
    }


async def run(options):
    report = {
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source(),
        "python": sys.version,
        "platform": platform.platform(),
        "definition": "MyLLM constructor only; profile fixtures and instance close excluded; 10 warmup instances; no SDK clients constructed; default GC; separate tracemalloc probe retains 100 instances after gc.collect",
        "cases": [await benchmark(protocol, options.iterations) for protocol in PROTOCOLS],
    }
    options.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    options = parser.parse_args()
    if options.iterations < 1:
        parser.error("iterations must be positive")
    asyncio.run(run(options))
