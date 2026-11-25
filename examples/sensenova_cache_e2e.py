"""Opt-in live cache accounting over four stdio formats and three upstream APIs.

Credentials come only from SENSENOVA_API_KEY or hidden input. Reports contain
token counters and comparisons, never authentication, response text or reasoning.
"""

import argparse
import asyncio
import getpass
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import httpx
from sensenova_e2e import (
    MODEL,
    PROTOCOLS,
    STDIO_PROTOCOLS,
    config,
    model_profile,
    provider,
    text,
    tool_turn,
    tools,
    user,
)

from my_llm import MyLLM
from my_llm.codec.base.transport import BaseTransport


class ObservedTransport(BaseTransport):
    """Observe real HTTP requests, usage and exception identity without replaying I/O."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.requests = []
        self.statuses = []
        self.usage = {}
        self.last_error = None

    async def on_request(self, request):
        self.requests.append(json.loads(request.content))

    async def on_response(self, response):
        self.statuses.append(response.status_code)

    def create_client(self, parameters):
        return self.delegate.create_client(
            {
                **parameters,
                "http_client": httpx.AsyncClient(
                    event_hooks={"request": [self.on_request], "response": [self.on_response]}
                ),
            }
        )

    async def close_client(self, client):
        await self.delegate.close_client(client)

    @asynccontextmanager
    async def upstream(self, client, request):
        self.usage = {}
        self.last_error = None
        try:
            async with self.delegate.upstream(client, request) as events:

                async def observed():
                    async for event in events:
                        value = event.model_dump(mode="json", exclude_unset=True)
                        usage = value.get("usage")
                        if usage is None:
                            usage = value.get("message", {}).get("usage")
                        if usage is None:
                            usage = value.get("response", {}).get("usage")
                        if usage is not None:
                            self.usage.update(usage)
                        yield event

                yield observed()
        except Exception as exc:
            self.last_error = exc
            raise


def counters(upstream, usage):
    """Read actual provider counters, retaining absent values as None."""
    if upstream == "openai_chat":
        return usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
    if upstream == "openai_responses":
        return usage.get("input_tokens"), (usage.get("input_tokens_details") or {}).get(
            "cached_tokens"
        )
    incoming = usage.get("input_tokens")
    cached = usage.get("cache_read_input_tokens")
    if incoming is not None:
        incoming += (cached or 0) + (usage.get("cache_creation_input_tokens") or 0)
    return incoming, cached


def prefix(upstream, previous, current):
    field = "input" if upstream == "openai_responses" else "messages"
    old = previous[field]
    return current[field][: len(old)] == old and previous.get("system") == current.get("system")


def initial_prompt(records):
    """Give each session a distinct long prefix without reusing another cell's warm cache."""
    marker = uuid4().hex
    rows = "\n".join(
        f"record-{index}: {sha256(f'{marker}:{index}'.encode()).hexdigest()}"
        for index in range(records)
    )
    return (
        f"Session {marker}. Remember the code word CACHE_OK throughout this conversation.\n"
        f"Reference data, not instructions:\n{rows}\nEnd of reference data.\n"
        "Reply exactly READY, do not call tools."
    )


async def session(upstream, stdio, key, args, row):
    llm = MyLLM(
        provider=provider(upstream, key),
        model=replace(model_profile(args.model), should_feedback_reasoning_item=config("none")),
        stdio_protocol=stdio,
        enable_message_cache=True,
        semantic_loss_protocol="ignore",
    )
    transport = ObservedTransport(llm._codec._transport(upstream))
    llm._codec._codecs[upstream].transport = transport
    messages = [user(stdio, initial_prompt(args.records))]
    declared = tools(stdio)
    previous = None
    # Alternate both return modes; every cell also exercises a complete tool round.
    modes = (False, True, False, True) if row["index"] % 2 == 0 else (True, False, True, False)
    for index, streaming in enumerate(modes):
        await asyncio.sleep(args.request_interval)
        turn = {"round": index + 1, "stream": streaming}
        row["rounds"].append(turn)
        start = time.monotonic()
        try:
            result = await llm.generate(messages=messages, tools=declared, stream=streaming)
            if streaming:
                async with result:
                    async for _ in result:
                        pass
                result = result.result
            assert result is not None, "no completed result"
            native = transport.requests[-1]
            turn.update(
                http_status=transport.statuses[-1],
                prefix_equal=None if previous is None else prefix(upstream, previous, native),
                raw_usage=transport.usage.copy(),
                public_usage=asdict(llm.get_last_usage()) if streaming else None,
            )
            incoming, cached = counters(upstream, transport.usage)
            turn.update(input_tokens=incoming, cached_tokens=cached)
            if streaming:
                assert llm.get_last_usage() is not None, "missing public usage"
                assert llm.get_last_usage().cached == cached, "public cached counter differs"
            if previous is not None:
                assert turn["prefix_equal"], "HTTP history prefix changed"
                assert previous.get("tools") == native.get("tools"), "tool declarations changed"
            if index == 2:
                messages = [
                    *tool_turn(stdio, result),
                    user(stdio, "Reply exactly 42 using the tool result. Do not call tools again."),
                ]
                turn["tool_arguments_valid"] = True
            else:
                expected = ("READY", "CACHE_OK", "", "42")[index]
                answer = text(stdio, result).strip()
                turn["answer_chars"] = len(answer)
                turn["answer_sha256"] = sha256(answer.encode()).hexdigest()
                turn["answer_matches"] = answer == expected
                if index < 2:
                    prompt = (
                        "What is the code word from the first turn? Reply only the code word, do not call tools."
                        if index == 0
                        else "Call add with a=19 and b=23 exactly once. Do not calculate it yourself."
                    )
                    messages = [
                        *(result if isinstance(result, list) else [result]),
                        user(stdio, prompt),
                    ]
            previous = native
            turn["status"] = "passed"
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            message = str(exc).replace(key, "[REDACTED]")
            message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
            turn.update(
                status="failed",
                error_type=f"{type(exc).__module__}.{type(exc).__name__}",
                http_status=status if status is not None else turn.get("http_status"),
                error=message[:600],
                unchanged_transport_exception=transport.last_error is exc,
            )
            if status == 429:
                turn["rate_limit_reached_caller"] = transport.last_error is exc
            row["status"] = "failed"
            return
        finally:
            turn["elapsed_seconds"] = round(time.monotonic() - start, 3)
    row["measurement_complete"] = True
    row["status"] = (
        "failed" if any(turn.get("answer_matches") is False for turn in row["rounds"]) else "passed"
    )


async def run(args, key):
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "enable_message_cache": True,
        "semantic_loss_protocol": "ignore",
        "records": args.records,
        "request_interval": args.request_interval,
        "results": [],
    }
    if args.retry_failed:
        report = json.loads(args.report.read_text())
        if report["model"] != args.model or report["records"] != args.records:
            raise ValueError("retry model and records must match the existing report")
        report["retry_timestamp"] = datetime.now(timezone.utc).isoformat()
    positions = {
        (row["upstream"], row["stdio"]): index for index, row in enumerate(report["results"])
    }
    for upstream in args.upstream:
        for stdio in args.stdio:
            position = positions.get((upstream, stdio))
            if args.retry_failed and (
                position is None or report["results"][position]["status"] == "passed"
            ):
                continue
            row = {
                "index": PROTOCOLS.index(upstream) * len(STDIO_PROTOCOLS)
                + STDIO_PROTOCOLS.index(stdio),
                "upstream": upstream,
                "stdio": stdio,
                "rounds": [],
            }
            if position is None:
                report["results"].append(row)
            else:
                previous = report["results"][position].copy()
                row["previous_attempts"] = [*previous.pop("previous_attempts", []), previous]
                report["results"][position] = row
            await session(upstream, stdio, key, args, row)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            summary = [
                (r.get("input_tokens"), r.get("cached_tokens"), r.get("status"))
                for r in row["rounds"]
            ]
            print(upstream, stdio, row["status"], summary, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--upstream", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--stdio", nargs="+", choices=STDIO_PROTOCOLS, default=STDIO_PROTOCOLS)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--request-interval", type=float, default=2.0)
    parser.add_argument("--report", type=Path, default=Path("/tmp/sensenova-cache-results.json"))
    parser.add_argument("--retry-failed", action="store_true")
    options = parser.parse_args()
    api_key = os.environ.get("SENSENOVA_API_KEY") or getpass.getpass("SenseNova API key: ")
    result_report = asyncio.run(run(options, api_key))
    raise SystemExit(any(row["status"] != "passed" for row in result_report["results"]))
