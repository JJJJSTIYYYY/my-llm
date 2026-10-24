"""Compare matched request-chain reports without merging changed workloads.

Run: python -m tests.performance.compare before.json after.json --json comparison.json
"""

import argparse
import json
from math import ceil
from pathlib import Path
from statistics import median


def key(case):
    return tuple(case[name] for name in ("suite", "upstream", "stdio", "stream"))


def percent(before, after):
    return (after / before - 1) * 100 if before else None


def metrics(case):
    return {
        "serial_p50_ms": case["warm"]["median_ms"],
        "serial_p95_ms": case["warm"]["p95_ms"],
        "cpu_p50_ms": case["warm"]["cpu_median_ms"],
        "concurrent_p50_ms": case["concurrent"]["batch_median_ms"],
        "serial_peak_bytes": case["memory"]["peak_bytes"],
        "concurrent_peak_bytes": case["concurrent"]["peak_bytes"],
    }


def compare(before, after):
    for name in (
        "python",
        "platform",
        "versions",
        "size",
        "iterations",
        "concurrency",
        "harness_sha256",
    ):
        if before[name] != after[name]:
            raise ValueError(f"reports differ in {name}")
    left = {key(case): case for case in before["cases"]}
    right = {key(case): case for case in after["cases"]}
    if (
        not left
        or left.keys() != right.keys()
        or len(left) != len(before["cases"])
        or len(right) != len(after["cases"])
    ):
        raise ValueError("reports must contain the same unique cases")
    cases = []
    for identity, old in left.items():
        new = right[identity]
        for name in ("workload_sha256", "response_sha256", "response_bytes", "http_chunks"):
            if old[name] != new[name]:
                raise ValueError(f"{identity}: reports differ in {name}")
        old_metrics, new_metrics = metrics(old), metrics(new)
        cases.append(
            {
                **dict(zip(("suite", "upstream", "stdio", "stream"), identity)),
                "before": old_metrics,
                "after": new_metrics,
                "change_percent": {
                    name: percent(old_metrics[name], new_metrics[name]) for name in old_metrics
                },
                "request_bytes_before": old["request_bytes"],
                "request_bytes_after": new["request_bytes"],
                "request_identical": old["request_sha256"] == new["request_sha256"],
                "output_events_before": old["warm"]["output_events"],
                "output_events_after": new["warm"]["output_events"],
            }
        )
    groups = {}
    for name, selected in (
        ("all", cases),
        ("native", [case for case in cases if case["suite"] == "native"]),
        ("cross", [case for case in cases if case["suite"] == "cross"]),
        ("collect", [case for case in cases if not case["stream"]]),
        ("stream", [case for case in cases if case["stream"]]),
    ):
        if selected:
            groups[name] = {
                "cases": len(selected),
                "median_change_percent": {
                    metric: median(case["change_percent"][metric] for case in selected)
                    for metric in selected[0]["change_percent"]
                },
            }
    return {
        "definition": "per-case percentage changes, then median across cases; negative means lower latency or memory; P95 is nearest-rank of each run, not a confidence bound; identical fixture and response hashes required; request normalization and public output-event changes reported explicitly",
        "before_source_sha256": before["source_sha256"],
        "after_source_sha256": after["source_sha256"],
        "harness_sha256": before["harness_sha256"],
        "groups": groups,
        "request_identical_cases": sum(case["request_identical"] for case in cases),
        "before_cache_growth_cases": sum(
            any(
                value["currsize"] != case["cache_before"][name]["currsize"]
                or value["misses"] != case["cache_before"][name]["misses"]
                for name, value in case["cache_after"].items()
            )
            for case in before["cases"]
        ),
        "after_cache_growth_cases": sum(
            any(
                value["currsize"] != case["cache_before"][name]["currsize"]
                or value["misses"] != case["cache_before"][name]["misses"]
                for name, value in case["cache_after"].items()
            )
            for case in after["cases"]
        ),
        "cases": cases,
    }


def compare_construction(before, after):
    for name in ("harness_sha256", "python", "platform"):
        if before[name] != after[name]:
            raise ValueError(f"construction reports differ in {name}")
    left = {case["protocol"]: case for case in before["cases"]}
    right = {case["protocol"]: case for case in after["cases"]}
    if not left or left.keys() != right.keys():
        raise ValueError("construction reports must cover the same protocols")
    return [
        {
            "protocol": protocol,
            "before_ms": old["median_ms"],
            "after_ms": right[protocol]["median_ms"],
            "change_percent": percent(old["median_ms"], right[protocol]["median_ms"]),
            "before_retained_bytes_per_instance": old["memory"]["current_bytes"]
            / old["memory"]["retained_instances"],
            "after_retained_bytes_per_instance": right[protocol]["memory"]["current_bytes"]
            / right[protocol]["memory"]["retained_instances"],
        }
        for protocol, old in left.items()
    ]


def compare_abba(before_runs, after_runs, *, before, after):
    for label, runs in (("before", before_runs), ("after", after_runs)):
        expected = before if label == "before" else after
        for run in runs:
            for name in ("source_sha256", "harness_sha256", "size", "python", "platform"):
                if run[name] != expected[name]:
                    raise ValueError(f"{label} ABBA report differs in {name}")
            if run["driver_sha256"] != before_runs[0]["driver_sha256"]:
                raise ValueError("ABBA reports use different measuring drivers")
            if any(run["versions"][name] != value for name, value in expected["versions"].items()):
                raise ValueError("ABBA reports use different SDK versions")
    results = []
    reference = {key(case): case for case in before_runs[0]["cases"]}
    for identity, case in reference.items():
        summary = {}
        for label, runs in (("before", before_runs), ("after", after_runs)):
            samples, run_medians = [], []
            for run in runs:
                entries = {key(entry): entry for entry in run["cases"]}
                if entries.keys() != reference.keys():
                    raise ValueError("ABBA reports cover different cases")
                entry = entries[identity]
                for name in ("workload_sha256", "response_sha256", "response_bytes", "http_chunks"):
                    if entry[name] != case[name]:
                        raise ValueError(f"{identity}: ABBA reports differ in {name}")
                samples.extend(sample["total_ms"] for sample in entry["warm"]["samples"])
                run_medians.append(entry["warm"]["median_ms"])
            ordered = sorted(samples)
            summary[label] = {
                "p50_ms": median(ordered),
                "p95_ms": ordered[ceil(len(ordered) * 0.95) - 1],
                "run_medians_ms": run_medians,
                "samples": len(ordered),
            }
        results.append(
            {
                **dict(zip(("suite", "upstream", "stdio", "stream"), identity)),
                **summary,
                "change_percent": percent(summary["before"]["p50_ms"], summary["after"]["p50_ms"]),
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--construction-before", type=Path)
    parser.add_argument("--construction-after", type=Path)
    parser.add_argument("--abba-before", nargs=2, type=Path)
    parser.add_argument("--abba-after", nargs=2, type=Path)
    options = parser.parse_args()
    if bool(options.construction_before) != bool(options.construction_after):
        parser.error("both construction reports are required")
    if bool(options.abba_before) != bool(options.abba_after):
        parser.error("both pairs of ABBA reports are required")
    before = json.loads(options.before.read_text())
    after = json.loads(options.after.read_text())
    report = compare(before, after)
    if options.construction_before:
        old = json.loads(options.construction_before.read_text())
        new = json.loads(options.construction_after.read_text())
        if (
            old["source_sha256"] != before["source_sha256"]
            or new["source_sha256"] != after["source_sha256"]
        ):
            parser.error("construction reports do not match the compared runtimes")
        report["construction"] = compare_construction(old, new)
    if options.abba_before:
        report["native_abba"] = compare_abba(
            [json.loads(path.read_text()) for path in options.abba_before],
            [json.loads(path.read_text()) for path in options.abba_after],
            before=before,
            after=after,
        )
        report["native_abba_median_change_percent"] = median(
            case["change_percent"] for case in report["native_abba"]
        )
    options.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["groups"], indent=2))
