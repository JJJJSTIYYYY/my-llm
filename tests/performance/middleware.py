"""Matched native vs direct middleware I/O over five real SDKs and local HTTP.

Run: python -m tests.performance.middleware --json /tmp/middleware.json
"""

import argparse
import asyncio
import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from my_llm import MyCodec, MyLLM, ToolResult
from tests.helpers.protocols import PROTOCOLS
from tests.performance.fingerprint import harness, source
from tests.performance.request_chain import probe
from tests.performance.runner import measure, summarize
from tests.performance.transport import LocalSDK, wire_frames
from tests.performance.validation import validate_exchange, validate_workload
from tests.performance.workloads import Size, build, controls, profiles


def direct_workload(workload):
    """Fixtures are constructed before timing; application values are unchanged.

    Remove encoder provenance to exercise the same public path as hand-built
    middleware models. Native signatures and ownership tags remain intact.
    """
    encoded = MyCodec().encode(workload.messages, workload.tools, input_format=workload.protocol)

    def part(value):
        changes = {"origin": None}
        if isinstance(value, ToolResult):
            changes["content"] = [part(child) for child in value.content]
        return replace(value, **changes)

    return replace(
        workload,
        stdio="middleware",
        messages=[
            replace(message, origin=None, content=[part(value) for value in message.content])
            for message in encoded.messages
        ],
        tools=[replace(tool, origin=None) for tool in encoded.tools],
    )


async def paired(workload, *, streaming, iterations, warmup, chunk_bytes):
    validate_workload(workload)
    direct = direct_workload(workload)
    frames = wire_frames(workload.protocol, workload.events, chunk_bytes)
    provider, model = profiles(workload.protocol)
    parameters = controls(workload.protocol)
    samples = {"native": [], "middleware": []}
    observed = {}
    with LocalSDK(workload.protocol, frames) as sdk:
        async with (
            MyLLM(provider=provider, model=model, stdio_protocol=workload.stdio) as native_llm,
            MyLLM(provider=provider, model=model, stdio_protocol="middleware") as middle_llm,
        ):
            variants = {
                "native": (native_llm, workload),
                "middleware": (middle_llm, direct),
            }
            for index in range(warmup + iterations):
                # N/M then M/N forms repeated N M M N blocks, balancing order.
                order = ("native", "middleware") if index % 2 == 0 else ("middleware", "native")
                for name in order:
                    llm, current = variants[name]
                    sdk.release_samples()
                    sample, result = await measure(
                        llm,
                        current,
                        stream=streaming,
                        parameters=parameters,
                    )
                    validate_exchange(current, current.stdio, result, sdk)
                    request = sdk.requests[-1].content
                    observed[name] = hashlib.sha256(request).hexdigest()
                    if index >= warmup:
                        samples[name].append(sample)
            assert observed["native"] == observed["middleware"], "request bytes must match"
            assert sdk.created_clients == 2
        assert all(body.closed for body in sdk.bodies)
        assert all(client.is_closed for client in sdk.http_clients)
    results = {name: summarize(values) for name, values in samples.items()}
    return {
        "upstream": workload.protocol,
        "stream": streaming,
        "request_sha256": observed["native"],
        "request_bytes_identical": True,
        "response_sha256": hashlib.sha256(b"".join(frames)).hexdigest(),
        "workload_sha256": hashlib.sha256(
            json.dumps(asdict(workload), sort_keys=True).encode()
        ).hexdigest(),
        "http_chunks": len(frames),
        **results,
        "median_change_percent": (
            results["middleware"]["median_ms"] / results["native"]["median_ms"] - 1
        )
        * 100,
    }, direct


async def run(options):
    report = {
        "source_sha256": source(),
        "harness_sha256": harness(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in (
                "openai",
                "anthropic",
                "google-genai",
                "ollama",
                "httpx",
                "pydantic",
            )
        },
        "size": asdict(Size()),
        "iterations": options.iterations,
        "warmup": options.warmup,
        "concurrency": options.concurrency,
        "timing_definition": (
            "paired native/middleware I/O, alternating N M M N order; generate through "
            "complete consumption, result retrieval and response close; real SDK parsing "
            "over immediate local HTTP; no tracing/network/server waits; default GC"
        ),
        "input_definition": (
            "identical native supported history, images, schemas, tool calls/results and "
            "signed reasoning; typed fixtures built and encoder provenance removed outside "
            "timing; target SDK request bytes must be identical; returned native vs typed "
            "formats differ, so stream event counts are reported separately"
        ),
        "probe_definition": (
            "separate warm tracemalloc and bounded cache snapshots; excludes fixtures and "
            "client construction, includes result and HTTP observations; not RSS; "
            "four-way response barrier proves overlap; two warmup batches"
        ),
        "cases": [],
    }
    for protocol in PROTOCOLS:
        workload = build(protocol, protocol, Size(), native=True)
        for streaming in (False, True):
            result, direct = await paired(
                workload,
                streaming=streaming,
                iterations=options.iterations,
                warmup=options.warmup,
                chunk_bytes=1024,
            )
            # No memory/concurrency probing overlaps serial timing.
            result["probes"] = {}
            for name, current in (("native", workload), ("middleware", direct)):
                result["probes"][name] = await probe(
                    current,
                    streaming,
                    options.concurrency,
                    options.iterations,
                )
            report["cases"].append(result)
            options.json.parent.mkdir(parents=True, exist_ok=True)
            options.json.write_text(json.dumps(report, indent=2) + "\n")
            print(
                protocol,
                "stream" if streaming else "collect",
                round(result["median_change_percent"], 2),
                "%",
                flush=True,
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    options = parser.parse_args()
    if options.iterations < 1 or options.warmup < 1 or options.concurrency < 2:
        parser.error("iterations/warmup must be positive; concurrency must be >=2")
    asyncio.run(run(options))
