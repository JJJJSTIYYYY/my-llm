"""Run with python -m tests.performance; all response bytes stay in memory."""

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from my_llm import __version__
from tests.helpers.protocols import PROTOCOLS
from tests.performance.runner import benchmark
from tests.performance.workloads import Size, build


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("native", "cross", "all"), default="all")
    parser.add_argument("--upstream", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--stdio", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--mode", choices=("collect", "stream", "both"), default="both")
    parser.add_argument("--iterations", type=positive, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--wire-chunk-bytes", type=positive, default=1024)
    for name, default in asdict(Size()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=positive, default=default)
    parser.add_argument("--json", type=Path, help="write metadata, raw samples and summaries")
    parsed = parser.parse_args()
    if parsed.warmup < 0:
        parser.error("--warmup must be non-negative")
    if parsed.parallel_calls > parsed.tools:
        parser.error("--parallel-calls cannot exceed --tools")
    if parsed.suite == "native" and not set(parsed.upstream).intersection(parsed.stdio):
        parser.error("--suite native requires a protocol shared by --upstream and --stdio")
    return parsed


async def run(options):
    size = Size(**{name: getattr(options, name) for name in asdict(Size())})
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "my_llm_version": __version__,
        "git_revision": revision,
        "dirty_worktree": dirty,
        "versions": {
            package: version(package)
            for package in (
                "my-llm-py",
                "openai",
                "anthropic",
                "google-genai",
                "ollama",
                "httpx",
                "pydantic",
            )
        },
        "size": asdict(size),
        "warmup": options.warmup,
        "wire_chunk_bytes": options.wire_chunk_bytes,
        "timing": "generate entry through complete consumption, response close and result copy; MyLLM construction, fixtures, assertions and instance close excluded; no network/server waits; real SDK parsing included; GC enabled",
        "cold_definition": "first generate on a new instance after unmeasured preflight; includes lazy SDK client creation, excludes process/import/schema cold start",
        "cases": [],
    }
    print(
        "suite  upstream          stdio             mode     cold ms   p50 ms   p95 ms   CPU ms   req/s",
        flush=True,
    )
    for native in [True, False] if options.suite == "all" else [options.suite == "native"]:
        for upstream in options.upstream:
            targets = (
                [upstream]
                if native and upstream in options.stdio
                else ([] if native else options.stdio)
            )
            for stdio in targets:
                workload = build(upstream, stdio, size, native=native)
                for stream in (
                    [False, True] if options.mode == "both" else [options.mode == "stream"]
                ):
                    result = await benchmark(
                        workload,
                        stream=stream,
                        iterations=options.iterations,
                        warmup=options.warmup,
                        wire_chunk_bytes=options.wire_chunk_bytes,
                    )
                    report["cases"].append(result)
                    warm = result["warm"]
                    print(
                        f"{result['suite']:6} {upstream:17} {stdio:17} {'stream' if stream else 'collect':7} {result['cold']['total_ms']:9.3f} {warm['median_ms']:8.3f} {warm['p95_ms']:8.3f} {warm['cpu_median_ms']:8.3f} {warm['requests_per_second']:7.2f}",
                        flush=True,
                    )
    if options.json:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"JSON: {options.json}", flush=True)
    return report


if __name__ == "__main__":
    asyncio.run(run(arguments()))
