"""Compare actual request preparation with cached or freshly compiled SDK rules.

Run: python -m tests.performance.compiled_cache --json /tmp/compiled-cache.json
"""

import argparse
import asyncio
import json
import platform
import sys
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from my_llm import MyLLM
from my_llm.cache import compiled_cache
from tests.helpers.protocols import PROTOCOLS
from tests.performance.fingerprint import harness, source
from tests.performance.message_cache import measure, summary
from tests.performance.request_chain import caches
from tests.performance.workloads import Size, build, controls, profiles


async def benchmark(protocol, iterations, warmup):
    workload = build(protocol, protocol, Size(), native=True)
    original = deepcopy((workload.messages, workload.tools))
    provider, model = profiles(protocol)
    samples = {"cached": [], "uncached": []}
    async with MyLLM(provider=provider, model=model, stdio_protocol=protocol) as llm:
        inputs = {
            "messages": workload.messages,
            "tools": workload.tools,
            "stream": True,
            **controls(protocol),
        }

        def prepare():
            return llm.codec.prepare(
                inputs, llm.resolved_profile, input_format=protocol, client=llm.client_parameters
            )

        prepare()
        before = caches()
        for index in range(warmup + iterations):
            order = ("cached", "uncached") if index % 2 == 0 else ("uncached", "cached")
            results = {}
            for name in order:
                with ExitStack() as patches:
                    if name == "uncached":
                        # Keep every compiler's behavior, bypass only its LRU.
                        # This affects existing imports of the same wrapper too.
                        for compiler_name in before:
                            compiler = getattr(compiled_cache, compiler_name)
                            patches.enter_context(
                                patch.object(compiler, "_compiled", compiler.__wrapped__)
                            )
                    results[name], sample = measure(prepare)
                if index >= warmup:
                    samples[name].append(sample)
            assert results["cached"] == results["uncached"]
            assert (workload.messages, workload.tools) == original
        assert not llm._clients
        after = caches()
        for name, info in after.items():
            assert info["misses"] == before[name]["misses"], name
            assert info["currsize"] == before[name]["currsize"], name
    cached, uncached = (summary(samples[name]) for name in ("cached", "uncached"))
    return {
        "protocol": protocol,
        "cached": cached,
        "uncached": uncached,
        "speedup": uncached["median_ms"] / cached["median_ms"],
        "cache_before": before,
        "cache_after": after,
    }


async def run(options):
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_sha256": source(),
        "harness_sha256": harness(),
        "versions": {
            name: version(name)
            for name in ("openai", "anthropic", "google-genai", "ollama", "pydantic")
        },
        "iterations": options.iterations,
        "warmup": options.warmup,
        "definition": (
            "Actual MyCodec.prepare with default complex native workloads; five rule LRUs "
            "enabled vs bypassed through the same compilers, alternating AB/BA pairs. "
            "Message caching disabled in both. Includes input projection/validation, "
            "translation and final request filtering/validation. Compiler switching, "
            "fixtures, instance setup, assertions and close excluded. No SDK clients, "
            "network or inference; default GC. Not a comparison of source versions."
        ),
        "cases": [],
    }
    for protocol in PROTOCOLS:
        case = await benchmark(protocol, options.iterations, options.warmup)
        report["cases"].append(case)
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{protocol}: cached={case['cached']['median_ms']:.3f}ms "
            f"uncached={case['uncached']['median_ms']:.3f}ms "
            f"speedup={case['speedup']:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    options = parser.parse_args()
    if options.iterations < 1 or options.warmup < 1:
        parser.error("iterations and warmup must be positive")
    asyncio.run(run(options))
