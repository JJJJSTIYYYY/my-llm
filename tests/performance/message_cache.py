"""Compare full-history translation with cached incremental Chat request preparation.

Run: python -m tests.performance.message_cache --json /tmp/message-cache.json
"""

import argparse
import gc
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter_ns, process_time_ns
from uuid import uuid4

from my_llm import MyLLM, omit
from tests.performance.fingerprint import source
from tests.performance.workloads import profiles, text


def messages(count, chars, *, offset=0):
    """Distinct batches, with assistant/user deltas and fixed-size text leaves."""
    return [
        {
            "role": (
                "system"
                if offset == 0 and count % 2 == 0 and index == 0
                else "assistant"
                if (count - index) % 2 == 0
                else "user"
            ),
            "content": f"{index + offset:08d}:" + text(chars - 9),
        }
        for index in range(count)
    ]


def prepare(llm, batch, *, cached, key):
    return llm.codec.prepare(
        {
            "messages": batch,
            "tools": omit,
            "stream": True,
            "max_tokens": 512,
            "reasoning": None,
            "reasoning_effort": "none",
            "extra_headers": None,
            "extra_body": None,
            "previous_response_id": None,
        },
        llm.resolved_profile,
        input_format="openai_chat",
        client=llm.client_parameters,
        enable_message_cache=cached,
        message_cache_key=key,
    )


def measure(operation):
    collections = sum(item["collections"] for item in gc.get_stats())
    cpu = process_time_ns()
    start = perf_counter_ns()
    result = operation()
    elapsed = perf_counter_ns() - start
    cpu = process_time_ns() - cpu
    return result, {
        "ms": elapsed / 1_000_000,
        "cpu_ms": cpu / 1_000_000,
        "gc_collections": sum(item["collections"] for item in gc.get_stats()) - collections,
    }


def summary(samples):
    values = sorted(item["ms"] for item in samples)
    return {
        "median_ms": median(values),
        "p95_ms": values[ceil(len(values) * 0.95) - 1],
        "cpu_median_ms": median(item["cpu_ms"] for item in samples),
        "samples_with_gc": sum(item["gc_collections"] > 0 for item in samples),
        "samples": samples,
    }


def benchmark(history_count, chars, *, delta_count, delta_chars, iterations, warmup):
    provider, model = profiles("openai_chat")
    # Keep instance/factory/schema setup outside the timed request preparation.
    llm = MyLLM(provider=provider, model=model, stdio_protocol="openai_chat")
    key = str(uuid4())
    history = messages(history_count, chars)
    delta = messages(delta_count, delta_chars, offset=history_count + 10000)
    full = [*history, *delta]
    original = json.dumps(full, ensure_ascii=False)
    samples = {"direct": [], "cached": []}
    try:
        for index in range(warmup + iterations):
            # Reconstruct the same already-translated history before every pair.
            # Otherwise repeated deltas would measure dedup hits or growing histories.
            llm.codec.expire_message_cache(key)
            if history:
                prepare(llm, history, cached=True, key=key)
            operations = {
                "direct": lambda: prepare(llm, full, cached=False, key=key),
                "cached": lambda: prepare(llm, delta, cached=True, key=key),
            }
            order = ("direct", "cached") if index % 2 == 0 else ("cached", "direct")
            results = {}
            for name in order:
                result, sample = measure(operations[name])
                results[name] = result
                if index >= warmup:
                    samples[name].append(sample)
            # Assertions and caller-input serialization are outside both timings.
            assert results["direct"] == results["cached"]
            assert len(results["cached"]["request"]["messages"]) == len(full)
            assert json.dumps(full, ensure_ascii=False) == original
        assert not llm._clients  # No SDK client or network I/O is used.
    finally:
        llm.codec.expire_message_cache(key)
    direct, cached = (summary(samples[name]) for name in ("direct", "cached"))
    return {
        "history_messages": history_count,
        "delta_messages": delta_count,
        "total_messages": len(full),
        "chars_per_message": chars,
        "delta_chars_per_message": delta_chars,
        "total_content_chars": history_count * chars + delta_count * delta_chars,
        "direct": direct,
        "cached": cached,
        "speedup": direct["median_ms"] / cached["median_ms"],
        "paired_wins": sum(
            cached["ms"] < direct["ms"]
            for direct, cached in zip(samples["direct"], samples["cached"], strict=True)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--chars", type=int, nargs="+", default=[32, 256, 2048, 16384])
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--delta-chars", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--json", type=Path, required=True)
    options = parser.parse_args()
    if min(options.history) < 0 or min(options.chars) < 9:
        parser.error("history must be nonnegative; chars must be at least 9")
    if options.delta_chars is not None and options.delta_chars < 9:
        parser.error("delta-chars must be at least 9")
    if options.delta < 1 or options.iterations < 1 or options.warmup < 0:
        parser.error("delta/iterations must be positive; warmup must be nonnegative")
    cases = [(count, chars) for chars in options.chars for count in options.history]
    random.Random(options.seed).shuffle(cases)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source_sha256": source(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {name: version(name) for name in ("openai", "pydantic")},
        "gc": {"enabled": gc.isenabled(), "threshold": gc.get_threshold()},
        "settings": {**vars(options), "json": str(options.json)},
        "definition": (
            "Warm MyCodec.prepare, openai_chat -> openai_chat. Direct translates full "
            "history + delta; cached translates delta and appends to previously translated "
            "history. Both include target request assembly/filtering/validation; cached "
            "includes fingerprint, copying and commit. Seed/reset, fixtures, construction, "
            "assertions and imports excluded. No SDK, network or model inference. "
            "Alternating AB/BA pairs; shuffled cases; Python default GC. One cache append "
            "per pair, never repeated-batch dedup. Samples are for one subsequent request, "
            "not cumulative conversation cost."
        ),
        "cases": [],
    }
    for count, chars in cases:
        case = benchmark(
            count,
            chars,
            delta_count=options.delta,
            delta_chars=options.delta_chars or chars,
            iterations=options.iterations,
            warmup=options.warmup,
        )
        report["cases"].append(case)
        print(
            f"history={count:4d} chars={chars:6d} "
            f"direct={case['direct']['median_ms']:.4f}ms "
            f"cached={case['cached']['median_ms']:.4f}ms "
            f"speedup={case['speedup']:.3f}x wins={case['paired_wins']}/{options.iterations}",
            flush=True,
        )
        options.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
