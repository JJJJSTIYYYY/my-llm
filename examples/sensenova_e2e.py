"""Live SenseNova 3x4 protocol matrix; opt-in, billable network requests.

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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from my_llm import (
    AssistantMessage,
    Attribute,
    Config,
    FunctionTool,
    ImageContent,
    ImageSource,
    MessageEvent,
    ModelProfile,
    MyLLM,
    Parameter,
    ProviderProfile,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)

PROTOCOLS = ("openai_chat", "openai_responses", "anthropic")
STDIO_PROTOCOLS = (*PROTOCOLS, "middleware")
BASE_CASES = ("text_collected", "text_stream", "tool_roundtrip")
POLICY_CASES = ("capability_losses", "feedback_none", "feedback_when_tool_calls")
MODEL = "sensenova-6.8-flash-lite"
SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def config(value):
    return Config(default_value=value)


def provider(protocol, key):
    """Use existing public configuration for the compatible upstream APIs."""
    body = {
        "openai_chat": {"reasoning_effort": "none"},
        "openai_responses": {"reasoning": {"effort": "none"}},
        "anthropic": {"thinking": {"type": "disabled"}},
    }[protocol]
    overrides = {}
    if protocol == "openai_chat":
        overrides["max_tokens"] = Parameter(path="$.max_tokens", default_value=512)
        overrides["reasoning_effort"] = Parameter(path="$.reasoning_effort")
        # SenseNova's Chat stream uses reasoning rather than reasoning_content.
        overrides["reasoning_content"] = Parameter(
            path="$.choices[0].delta.reasoning",
            default_value=None,
        )
    return ProviderProfile(
        protocol=Config(default_value=protocol),
        base_url=config("https://token.sensenova.cn" + ("" if protocol == "anthropic" else "/v1")),
        # Anthropic's SDK normally sends x-api-key; this service specifies Bearer.
        api_key=Parameter(
            path="$.extra_headers.Authorization" if protocol == "anthropic" else "$.api_key",
            default_value=f"Bearer {key}" if protocol == "anthropic" else key,
        ),
        extra_body=config(body),
        timeout=config(60.0),
        max_retries=config(0),
        **overrides,
    )


def user(protocol, text):
    if protocol == "middleware":
        return UserMessage("user", [TextContent(text)])
    if protocol == "openai_responses":
        return {"role": "user", "content": [{"type": "input_text", "text": text}]}
    return {"role": "user", "content": text}


def tools(protocol):
    function = {"name": "add", "description": "Add two integers.", "parameters": SCHEMA}
    if protocol == "middleware":
        return [FunctionTool(**function, strict=False)]
    if protocol == "openai_chat":
        return [{"type": "function", "function": function}]
    if protocol == "openai_responses":
        return [{"type": "function", **function, "strict": False}]
    return [{"name": "add", "description": function["description"], "input_schema": SCHEMA}]


def text(protocol, result):
    if protocol == "middleware":
        assert isinstance(result, AssistantMessage), "expected typed assistant output"
        return "".join(part.text for part in result.content if isinstance(part, TextContent))
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
    """Inspect and replay the declared I/O format without converting the test oracle."""
    if protocol == "middleware":
        assert isinstance(result, AssistantMessage)
        calls = [part for part in result.content if isinstance(part, ToolCall)]
        assert len(calls) == 1, "expected exactly one typed function call"
        call = calls[0]
        assert isinstance(call.call_id, str) and call.call_id, "missing typed tool-call ID"
        name, args = call.name, call.arguments
        history = [
            result,
            ToolMessage("user", [ToolResult(call.call_id, [TextContent("42")], name=name)]),
        ]
    elif protocol == "openai_chat":
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


async def consume(llm, *, stream, request_interval=0.0, **kwargs):
    if request_interval:
        await asyncio.sleep(request_interval)
    result = await llm.generate(stream=stream, max_tokens=512, **kwargs)
    if not stream:
        return result, []
    events = []
    async with result:
        async for event in result:
            events.append(event)
    assert events and result.result is not None, "stream did not complete"
    if llm.stdio_protocol == "middleware":
        assert all(isinstance(event, MessageEvent) for event in events)
        assert events[-1].type == "finish", "typed stream did not finish"
        assert all(event.field != "arguments" for event in events), "tool arguments fragmented"
    return result.result, events


def delta_text(protocol, events):
    if protocol == "middleware":
        return "".join(
            event.value for event in events if event.type == "delta" and event.field == "text"
        )
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


def check_answer(answer, expected):
    # Reasoning remains separate from visible text for every I/O format.
    assert answer == expected, f"unexpected answer: {answer[:160]!r}"
    return {"text": expected, "rendered_prefix_chars": len(answer) - len(expected)}


async def scenario(llm, protocol, case, *, request_interval=0.0):
    if case in ("text_collected", "text_stream"):
        streamed = case == "text_stream"
        result, events = await consume(
            llm,
            stream=streamed,
            request_interval=request_interval,
            messages=[user(protocol, "Reply with exactly SENSENOVA_OK and nothing else.")],
        )
        answer = text(protocol, result).strip()
        checked = check_answer(answer, "SENSENOVA_OK")
        if streamed:
            assert delta_text(protocol, events).strip() == answer, "delta/result mismatch"
        return {**checked, "events": len(events), "requests": 1}
    prompt = user(
        protocol,
        "You must call add with a=19 and b=23. After receiving the tool result, "
        "reply with exactly the result number and nothing else. Do not calculate it yourself.",
    )
    declared = tools(protocol)
    result, events = await consume(
        llm, stream=True, request_interval=request_interval, messages=[prompt], tools=declared
    )
    history = [prompt, *tool_turn(protocol, result)]
    if protocol == "middleware":
        calls = [event.block for event in events if event.type == "tool"]
        assert len(calls) == 1 and isinstance(calls[0], ToolCall)
        assert calls[0].arguments == {"a": 19, "b": 23}, "incomplete typed tool arguments"
        assert calls[0].call_id == history[-1].call_id, "typed tool/result ID mismatch"
    final, final_events = await consume(
        llm, stream=True, request_interval=request_interval, messages=history, tools=declared
    )
    answer = text(protocol, final).strip()
    checked = check_answer(answer, "42")
    assert delta_text(protocol, final_events).strip() == answer, "tool final delta mismatch"
    return {
        "tool": "add",
        "arguments": {"a": 19, "b": 23},
        **checked,
        "events": len(events) + len(final_events),
        "requests": 2,
    }


async def policy_scenario(upstream, model_id, key, case, *, request_interval=0.0):
    """Exercise decoder policies, then send the complete filtered draft upstream."""
    unsupported = case == "capability_losses"
    thought = "POLICY_THOUGHT_MUST_NOT_BE_SENT"
    picture = "https://example.invalid/POLICY_IMAGE_MUST_NOT_BE_SENT.png"
    prompt = "Reply with exactly SENSENOVA_OK and nothing else."
    messages = [
        UserMessage("user", [TextContent("Remember the test marker.")]),
        AssistantMessage("assistant", [ReasoningContent(thought), TextContent("Ready.")]),
        UserMessage("user", [TextContent(prompt)]),
    ]
    declared = []
    if unsupported:
        settings = {
            "feedback_reasoning_content": config("none"),
            "supports_vision": config(False),
            "supports_function_calling": config(False),
            "supports_message_name": config(False),
            "reasoning_enable": Parameter(
                path={
                    "openai_chat": "$.reasoning_effort",
                    "openai_responses": "$.reasoning",
                    "anthropic": "$.thinking",
                }[upstream],
                supports=False,
            ),
        }
        messages[-1].name = "POLICY_NAME_MUST_NOT_BE_SENT"
        messages[-1].content.append(ImageContent(ImageSource("url", url=picture)))
        declared = tools("middleware")
    else:
        settings = {"feedback_reasoning_content": config(case.removeprefix("feedback_"))}
    before = deepcopy((messages, declared))
    input_losses, output_losses, drafts = [], [], []

    def input_hook(draft, losses, context):
        assert context.origin_source == "input" and context.stage == "request"
        assert context.target_protocol == upstream
        assert draft["model"] == model_id and prompt in json.dumps(draft)
        assert not draft.get("tools"), "unsupported tools reached request draft"
        rendered = json.dumps(draft)
        assert "POLICY_" not in rendered, "unsupported history reached request draft"
        assert all("does not support" in item.reason for item in losses)
        input_losses.extend(losses)
        drafts.append(draft)

    def output_hook(draft, losses, context):
        assert context.origin_source == "output"
        output_losses.extend(losses)

    async with MyLLM(
        provider=provider(upstream, key),
        model=ModelProfile(model=Attribute(default_value=model_id), **settings),
        stdio_protocol="middleware",
        semantic_loss_protocol="hook",
        input_semantic_loss_hook=input_hook,
        output_semantic_loss_hook=output_hook,
    ) as llm:
        result, events = await consume(
            llm,
            stream=True,
            request_interval=request_interval,
            messages=messages,
            tools=declared,
        )
    assert (messages, declared) == before, "translation changed caller input"
    if unsupported:
        assert len(drafts) == 1, "request losses were not collected together"
        for label in ("image input", "function calling", "message names", "reasoning content"):
            assert any(label in item.reason for item in input_losses), label
    else:
        assert not input_losses, "feedback policy exclusion reported a semantic loss"
    answer = text("middleware", result).strip()
    assert delta_text("middleware", events).strip() == answer, "policy delta/result mismatch"
    return {
        **check_answer(answer, "SENSENOVA_OK"),
        "input_losses": len(input_losses),
        "output_losses": len(output_losses),
        "events": len(events),
        "requests": 1,
    }


async def run(args, key):
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": "https://token.sensenova.cn",
        "results": [],
    }
    if args.retry_failed:
        report = json.loads(args.report.read_text())
        if report["model"] != args.model:
            raise ValueError("retry model must match the existing report")
        report["retry_timestamp"] = datetime.now(timezone.utc).isoformat()
    report["semantic_loss_protocol"] = args.semantic_loss_protocol
    positions = {
        (row["upstream"], row["stdio"], row["case"]): index
        for index, row in enumerate(report["results"])
    }
    model = ModelProfile(
        model=Attribute(default_value=args.model), supports_function_calling=config(True)
    )
    for upstream in args.upstream or PROTOCOLS:
        for stdio in args.stdio or STDIO_PROTOCOLS:
            async with MyLLM(
                provider=provider(upstream, key),
                model=model,
                stdio_protocol=stdio,
                semantic_loss_protocol=args.semantic_loss_protocol,
            ) as llm:
                cases = (*BASE_CASES, *POLICY_CASES) if stdio == "middleware" else BASE_CASES
                for case in cases:
                    if args.case and case not in args.case:
                        continue
                    row = {
                        "upstream": upstream,
                        "stdio": stdio,
                        "case": case,
                        "semantic_loss_protocol": (
                            "hook" if case in POLICY_CASES else args.semantic_loss_protocol
                        ),
                    }
                    position = positions.get((upstream, stdio, case))
                    if args.retry_failed:
                        if position is None or report["results"][position]["status"] == "passed":
                            continue
                        previous = report["results"][position]
                        row["previous_attempts"] = [
                            *previous.get("previous_attempts", []),
                            {
                                key: value
                                for key, value in previous.items()
                                if key != "previous_attempts"
                            },
                        ]
                    started = time.monotonic()
                    try:
                        operation = (
                            policy_scenario(
                                upstream,
                                args.model,
                                key,
                                case,
                                request_interval=args.request_interval,
                            )
                            if case in POLICY_CASES
                            else scenario(llm, stdio, case, request_interval=args.request_interval)
                        )
                        row.update(await asyncio.wait_for(operation, 150))
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
                    if position is None:
                        report["results"].append(row)
                    else:
                        report["results"][position] = row
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
    parser.add_argument("--stdio", nargs="+", choices=STDIO_PROTOCOLS)
    parser.add_argument("--case", nargs="+", choices=(*BASE_CASES, *POLICY_CASES))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--request-interval", type=float, default=3.0, metavar="SECONDS")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--semantic-loss-protocol", choices=("raise", "ignore"), default="ignore")
    args = parser.parse_args()
    if args.request_interval < 0:
        parser.error("--request-interval must be nonnegative")
    if args.retry_failed and (args.report is None or not args.report.is_file()):
        parser.error("--retry-failed requires an existing --report file")
    key = os.environ.get("SENSENOVA_API_KEY") or getpass.getpass("SenseNova API key: ")
    if not key.strip():
        parser.error("SENSENOVA_API_KEY is required")
    raise SystemExit(asyncio.run(run(args, key)))


if __name__ == "__main__":
    main()
