"""Measure full, incremental and repeated-batch preparation without network I/O.

Run: python -m tests.performance.preparation --json /tmp/preparation.json
Use PYTHONPATH to select a saved runtime while retaining this same driver.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import platform
import re
import sys
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns, process_time_ns

from my_llm import MyLLM
from tests.helpers.protocols import PROTOCOLS
from tests.performance.fingerprint import harness, source
from tests.performance.middleware import direct_workload
from tests.performance.workloads import Size, build, controls, profiles


def fingerprint(value):
    """Ignore only randomly generated Responses item prefixes in request digests."""
    return hashlib.sha256(json.dumps(fingerprint_value(value), sort_keys=True).encode()).hexdigest()


def fingerprint_value(value):
    """Normalize nested generated item IDs without changing application values."""
    if isinstance(value, list):
        return [fingerprint_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: re.sub(r"^(msg|fc)_[0-9a-f]{32}_", r"\1_", item)
            if key == "id" and isinstance(item, str)
            else fingerprint_value(item)
            for key, item in value.items()
        }
    return value


async def measure(workload, mode, *, iterations, warmup):
    provider, model = profiles(workload.protocol)
    parameters = controls(workload.protocol)
    cached = mode != "uncached"
    samples = []
    request_hash = None
    history, delta = workload.messages[:-1], workload.messages[-1:]
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol=workload.stdio,
        enable_message_cache=cached,
    ) as llm:

        async def prepare(messages):
            return await llm.codec.prepare(
                {"messages": messages, "tools": workload.tools, "stream": True, **parameters},
                llm.resolved_profile,
                input_format=workload.stdio,
                client=llm.client_parameters,
                enable_message_cache=cached,
                message_cache_key=llm._message_cache_key,
            )

        if mode == "repeat":
            await prepare(history)
            await prepare(delta)
        for index in range(warmup + iterations):
            if mode == "incremental":
                llm.codec.expire_message_cache(llm._message_cache_key)
                await prepare(history)
            collections = sum(item["collections"] for item in gc.get_stats())
            cpu_start = process_time_ns()
            start = perf_counter_ns()
            prepared = await prepare(delta if cached else workload.messages)
            elapsed = perf_counter_ns() - start
            cpu_elapsed = process_time_ns() - cpu_start
            observed = fingerprint(prepared["request"])
            if request_hash is None:
                request_hash = observed
            assert request_hash == observed
            if index >= warmup:
                samples.append(
                    {
                        "ms": elapsed / 1_000_000,
                        "cpu_ms": cpu_elapsed / 1_000_000,
                        "gc_collections": sum(item["collections"] for item in gc.get_stats())
                        - collections,
                    }
                )
        assert llm._client is None
    values = sorted(sample["ms"] for sample in samples)
    return {
        "upstream": workload.protocol,
        "stdio": workload.stdio,
        "rounds": workload.size.rounds,
        "input_messages": len(workload.messages),
        "delta_messages": len(delta),
        "mode": mode,
        "request_sha256": request_hash,
        "median_ms": median(values),
        "p95_ms": values[(len(values) * 95 + 99) // 100 - 1],
        "cpu_median_ms": median(sample["cpu_ms"] for sample in samples),
        "samples": samples,
    }


async def run(options):
    report = {
        "source_sha256": source(),
        "harness_sha256": harness(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("openai", "anthropic", "pydantic")},
        "iterations": options.iterations,
        "warmup": options.warmup,
        "definition": (
            "Warm MyCodec.prepare, default GC, no SDK client or network. Fixtures, instance "
            "construction, cache resets/seeding and assertions excluded. Incremental rebuilds "
            "prior history then adds the last user message. Repeat seeds once and repeatedly "
            "prepares that same last batch without rebuilding or growing history. "
            "Digests normalize only UUID prefixes in generated Responses item IDs."
        ),
        "cases": [],
    }
    for rounds in options.rounds:
        size = replace(
            Size(), rounds=rounds, tools=4, parallel_calls=2, text_chars=256, argument_items=3
        )
        for upstream in PROTOCOLS:
            for stdio in (*PROTOCOLS, "middleware"):
                workload = build(
                    upstream,
                    upstream if stdio == "middleware" else stdio,
                    size,
                    native=stdio in (upstream, "middleware"),
                )
                fixture_hash = hashlib.sha256(
                    json.dumps(asdict(workload), sort_keys=True).encode()
                ).hexdigest()
                if stdio == "middleware":
                    workload = direct_workload(workload)
                for mode in ("uncached", "incremental", "repeat"):
                    case = await measure(
                        workload, mode, iterations=options.iterations, warmup=options.warmup
                    )
                    case["workload_sha256"] = fixture_hash
                    report["cases"].append(case)
        options.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"rounds={rounds}: {len(report['cases'])} cases complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--rounds", type=int, nargs="+", default=[2, 24])
    options = parser.parse_args()
    if options.iterations < 1 or options.warmup < 0 or min(options.rounds) < 1:
        parser.error("iterations/rounds must be positive and warmup nonnegative")
    asyncio.run(run(options))
