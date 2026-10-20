"""Live SenseNova 3x3 protocol matrix; opt-in, billable network requests.

Run: uv run python examples/sensenova_e2e.py --report /tmp/sensenova-results.json
Reads SENSENOVA_API_KEY or prompts without echo. Never persists credentials.
"""

import argparse
import asyncio
import getpass
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from my_llm import Config, ModelProfile, MyLLM, Parameter, ProviderProfile

PROTOCOLS = ("openai_chat", "openai_responses", "anthropic")
MODEL = "sensenova-6.8-flash-lite"
SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def config(name, value):
    return Config(name=name, default_value=value)


def provider(protocol, key):
    """Use existing public configuration for the compatible upstream APIs."""
    body = {
        "openai_chat": {"reasoning_effort": "none"},
        "openai_responses": {"reasoning": {"effort": "none"}},
        "anthropic": {"thinking": {"type": "disabled"}},
    }[protocol]
    overrides = {}
    if protocol == "openai_chat":
        overrides["max_tokens"] = Parameter(
            name="max_tokens", path="$.max_tokens", default_value=512, phase="decode"
        )
        # SenseNova's Chat stream uses reasoning rather than reasoning_content.
        overrides["reasoning_content"] = Parameter(
            name="reasoning_content",
            path="$.choices[0].delta.reasoning",
            default_value=None,
            phase="encode",
        )
    return ProviderProfile(
        protocol=protocol,
        base_url=config(
            "base_url", "https://token.sensenova.cn" + ("" if protocol == "anthropic" else "/v1")
        ),
        api_key=config("api_key", key),
        # Anthropic's SDK normally sends x-api-key; this service specifies Bearer.
        extra_headers=config("extra_headers", {"Authorization": f"Bearer {key}"}),
        extra_body=config("extra_body", body),
        timeout=config("timeout", 60.0),
        max_retries=config("max_retries", 0),
        **overrides,
    )


def user(protocol, text):
    if protocol == "openai_responses":
        return {"role": "user", "content": [{"type": "input_text", "text": text}]}
    return {"role": "user", "content": text}


def tools(protocol):
    function = {"name": "add", "description": "Add two integers.", "parameters": SCHEMA}
    if protocol == "openai_chat":
        return [{"type": "function", "function": function}]
    if protocol == "openai_responses":
        return [{"type": "function", **function, "strict": False}]
    return [{"name": "add", "description": function["description"], "input_schema": SCHEMA}]


def text(protocol, result):
    if protocol == "openai_chat":
        assert result["role"] == "assistant"
        return result.get("content") or ""
    if protocol == "openai_responses":
        return "".join(
            part["text"]
            for item in result
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
    assert result["role"] == "assistant"
    return "".join(part["text"] for part in result["content"] if part["type"] == "text")


def tool_turn(protocol, result):
    """Inspect and replay native output without converting the test oracle."""
    if protocol == "openai_chat":
        calls = result.get("tool_calls", [])
        assert len(calls) == 1, "expected exactly one function call"
        call = calls[0]
        name, args = call["function"]["name"], json.loads(call["function"]["arguments"])
        history = [result, {"role": "tool", "tool_call_id": call["id"], "content": "42"}]
    elif protocol == "openai_responses":
        calls = [item for item in result if item["type"] == "function_call"]
        assert len(calls) == 1, "expected exactly one function call"
        call = calls[0]
        name, args = call["name"], json.loads(call["arguments"])
        history = [
            *result,
            {"type": "function_call_output", "call_id": call["call_id"], "output": "42"},
        ]
    else:
        calls = [part for part in result["content"] if part["type"] == "tool_use"]
        assert len(calls) == 1, "expected exactly one function call"
        call = calls[0]
        name, args = call["name"], call["input"]
        history = [
            result,
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "42"}],
            },
        ]
    assert name == "add" and args == {"a": 19, "b": 23}, (name, args)
    assert args["a"] + args["b"] == 42
    return history


async def consume(llm, *, stream, **kwargs):
    result = await llm.generate(stream=stream, max_tokens=512, **kwargs)
    if not stream:
        return result, []
    events = []
    async with result:
        async for event in result:
            events.append(event)
    assert events and result.result is not None, "stream did not complete"
    return result.result, events


def delta_text(protocol, events):
    if protocol == "openai_chat":
        return "".join(
            choice.get("delta", {}).get("content") or ""
            for event in events
            for choice in event.get("choices", [])
        )
    if protocol == "openai_responses":
        return "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    return "".join(
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )


def check_answer(llm, protocol, answer, expected):
    # Foreign thinking blocks intentionally become text (translation-boundaries.md).
    # Require exact output otherwise; do not silently accept arbitrary prefixes.
    foreign_thinking = llm.provider.protocol in ("openai_chat", "anthropic") and (
        protocol != llm.provider.protocol
    )
    if foreign_thinking:
        assert answer.splitlines()[-1] == expected, "unexpected final answer line"
    else:
        assert answer == expected, "unexpected answer"
    return {"text": expected, "rendered_prefix_chars": len(answer) - len(expected)}


async def scenario(llm, protocol, case):
    if case in ("text_collected", "text_stream"):
        streamed = case == "text_stream"
        result, events = await consume(
            llm,
            stream=streamed,
            messages=[user(protocol, "Reply with exactly SENSENOVA_OK and nothing else.")],
        )
        answer = text(protocol, result).strip()
        checked = check_answer(llm, protocol, answer, "SENSENOVA_OK")
        if streamed:
            assert delta_text(protocol, events).strip() == answer, "delta/result mismatch"
        return {**checked, "events": len(events), "requests": 1}
    prompt = user(
        protocol,
        "You must call add with a=19 and b=23. After receiving the tool result, "
        "reply with exactly the result number and nothing else. Do not calculate it yourself.",
    )
    declared = tools(protocol)
    result, events = await consume(llm, stream=True, messages=[prompt], tools=declared)
    history = [prompt, *tool_turn(protocol, result)]
    final, final_events = await consume(llm, stream=True, messages=history, tools=declared)
    answer = text(protocol, final).strip()
    checked = check_answer(llm, protocol, answer, "42")
    assert delta_text(protocol, final_events).strip() == answer, "tool final delta mismatch"
    return {
        "tool": "add",
        "arguments": {"a": 19, "b": 23},
        **checked,
        "events": len(events) + len(final_events),
        "requests": 2,
    }


async def run(args, key):
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": "https://token.sensenova.cn",
        "results": [],
    }
    model = ModelProfile(
        model=args.model, supports_function_calling=config("supports_function_calling", True)
    )
    for upstream in args.upstream or PROTOCOLS:
        for stdio in args.stdio or PROTOCOLS:
            async with MyLLM(
                provider=provider(upstream, key), model=model, stdio_protocol=stdio
            ) as llm:
                for case in ("text_collected", "text_stream", "tool_roundtrip"):
                    row = {"upstream": upstream, "stdio": stdio, "case": case}
                    started = time.monotonic()
                    try:
                        row.update(await asyncio.wait_for(scenario(llm, stdio, case), 150))
                        row["status"] = "passed"
                    except Exception as exc:
                        # Never persist SDK request headers, profiles or tracebacks.
                        error = str(exc).replace(key, "[REDACTED]")
                        row.update(
                            status="failed",
                            error_type=type(exc).__name__,
                            error=re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", error)[:1500],
                        )
                    row["seconds"] = round(time.monotonic() - started, 3)
                    report["results"].append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if args.report:
                        args.report.parent.mkdir(parents=True, exist_ok=True)
                        args.report.write_text(json.dumps(report, indent=2) + "\n")
    failed = sum(row["status"] != "passed" for row in report["results"])
    print(f"{len(report['results']) - failed} passed, {failed} failed", flush=True)
    return bool(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--upstream", nargs="+", choices=PROTOCOLS)
    parser.add_argument("--stdio", nargs="+", choices=PROTOCOLS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    key = os.environ.get("SENSENOVA_API_KEY") or getpass.getpass("SenseNova API key: ")
    if not key.strip():
        parser.error("SENSENOVA_API_KEY is required")
    raise SystemExit(asyncio.run(run(args, key)))


if __name__ == "__main__":
    main()
