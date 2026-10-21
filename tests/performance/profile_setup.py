"""Measure profile declarations separately from resolution and instance setup.

Run this file with PYTHONPATH selecting either source and test fixture snapshot.
No SDK clients or network requests are created. Default GC remains enabled.
"""

import argparse
import gc
import hashlib
import json
import tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import my_llm
from my_llm import resolve_profiles
from tests.helpers.protocols import PROTOCOLS
from tests.performance.workloads import profiles


def measure(factory, batches, iterations):
    for _ in range(100):
        factory()
    samples = []
    for _ in range(batches):
        start = perf_counter_ns()
        for _ in range(iterations):
            factory()
        samples.append((perf_counter_ns() - start) / iterations / 1000)
    return {"median_us": median(samples), "batch_samples_us": samples}


def memory(factory, count=100):
    gc.collect()
    tracemalloc.start()
    try:
        retained = [factory() for _ in range(count)]
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(retained) == count
    return {"retained_count": count, "current_bytes": current, "peak_bytes": peak}


def run(options):
    cases = []
    for protocol in PROTOCOLS:
        provider, model = profiles(protocol)
        cases.append(
            {
                "protocol": protocol,
                "declarations": measure(
                    lambda: profiles(protocol), options.batches, options.iterations
                ),
                "resolution": measure(
                    lambda: resolve_profiles(provider, model), options.batches, options.iterations
                ),
                "declaration_memory": memory(lambda: profiles(protocol)),
                "resolution_memory": memory(lambda: resolve_profiles(provider, model)),
            }
        )
    report = {
        "definition": "Profile/Parameter/Config creation and resolve_profiles measured separately; prebuilt fixtures excluded from resolution; 100 warmups; default GC; per-operation microseconds averaged within each batch; separate tracemalloc probes retain 100 values after gc.collect, tracing is off during timing; no SDK construction or requests",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            b"".join(
                path.read_bytes() for path in sorted(Path(my_llm.__file__).parent.rglob("*.py"))
            )
        ).hexdigest(),
        "batches": options.batches,
        "iterations_per_batch": options.iterations,
        "cases": cases,
    }
    options.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args()
    if options.batches < 1 or options.iterations < 1:
        parser.error("batches and iterations must be positive")
    run(options)
