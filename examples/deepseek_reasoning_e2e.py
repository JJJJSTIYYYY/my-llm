"""Live DeepSeek reasoning feedback audit; opt-in, billable API requests.

Run: uv run python examples/deepseek_reasoning_e2e.py
Use --validate for an offline check or --isolated for independent-ID controls.
The key is entered without echo and never written to files, environment variables,
command arguments or request logs. Output contains redacted summaries only.
"""

import argparse
import asyncio
import getpass
import json
import logging
import re
import resource
import secrets
import sys
import time
from copy import deepcopy

import httpx
from openai import AsyncOpenAI

from my_llm import (
    Attribute,
    Config,
    FunctionTool,
    InputParameter,
    Message,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolResult,
)
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport

BASE = "https://api.deepseek.com/v1"
MODEL = None
SECRET = None
CASE = "startup"
RECORDS = []
TOOL = FunctionTool(
    "get_audit_value",
    "Return the unknown audit number for the supplied key.",
    {
        "type": "object",
        "properties": {"key": {"type": "string", "enum": ["audit-key"]}},
        "required": ["key"],
        "additionalProperties": False,
    },
)
PROMPT = Message(
    "user",
    [
        TextContent(
            "Call get_audit_value with key audit-key exactly once to obtain an unknown number. "
            "You cannot know the number without calling the tool. After receiving its result, "
            "reply with that number only and do not call the tool again."
        )
    ],
)


def safe(value):
    value = str(value)
    if SECRET:
        value = value.replace(SECRET, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)


def emit(kind, **data):
    print(safe(json.dumps({"event": kind, "case": CASE, **data}, ensure_ascii=False)), flush=True)


async def observe_request(request):
    if request.url.host != "api.deepseek.com":
        raise RuntimeError("Refusing to send credentials to a different host")
    if request.method == "POST":
        payload = json.loads(request.content)
        messages = payload.get("messages", [])
        assistants = [m for m in messages if m.get("role") == "assistant"]
        calls = [c.get("id") for m in assistants for c in m.get("tool_calls", [])]
        results = [m for m in messages if m.get("role") == "tool"]
        summary = {
            "stream": payload.get("stream", False),
            "thinking": payload.get("thinking"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "assistant_feedback": [
                {
                    "present": "reasoning_content" in m,
                    "type": type(m.get("reasoning_content")).__name__,
                    "length": len(m.get("reasoning_content") or ""),
                }
                for m in assistants
            ],
            "tool_calls": len(calls),
            "tool_results": len(results),
            "tool_ids_paired": set(calls) == {m.get("tool_call_id") for m in results},
            "empty_tool_results": sum(not m.get("content") for m in results),
        }
        RECORDS.append({"case": CASE, "summary": summary})
        emit("wire", **summary)


async def observe_response(response):
    emit("http", status=response.status_code)


def http_client():
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        timeout=90,
        event_hooks={"request": [observe_request], "response": [observe_response]},
    )


def provider():
    return ProviderProfile(
        protocol=Config(default_value="openai_chat"),
        base_url=Config(default_value=BASE),
        api_key=InputParameter(path="$.api_key", default_value=SECRET),
        timeout=Config(default_value=90),
        max_retries=Config(default_value=0),
    )


def model():
    return ModelProfile(
        model=Attribute(default_value=MODEL or "validation-only"),
        supports_function_calling=Config(default_value=True),
        should_feedback_reasoning_item=Config(default_value="when_tool_calls"),
        max_tokens=InputParameter(path="$.max_tokens", default_value=1024),
        reasoning_enable=InputParameter(
            path="$.extra_body.thinking", default_value={"type": "enabled"}
        ),
        reasoning_effort=InputParameter(path="$.reasoning_effort"),
    )


def client_factory(self, parameters):
    return AsyncOpenAI(**parameters, http_client=http_client())


def message_summary(message):
    return {
        "reasoning_chars": sum(
            len(p.reasoning or "") for p in message.content if isinstance(p, ReasoningContent)
        ),
        "feedback_chars": sum(
            len(p.feedback_item or "") for p in message.content if isinstance(p, ReasoningContent)
        ),
        "tools": sum(isinstance(p, ToolCall) for p in message.content),
        "answer": "".join(p.text for p in message.content if isinstance(p, TextContent))[:200],
    }


async def generate(messages, *, streaming=False):
    async with MyLLM(provider=provider(), model=model(), stdio_protocol="middleware") as llm:
        result = await llm.generate(
            messages=messages,
            tools=[TOOL],
            stream=streaming,
            reasoning=True,
            reasoning_effort="low",
        )
        deltas = 0
        if streaming:
            async with result:
                async for event in result:
                    deltas += event.type == "delta" and event.field == "reasoning"
                    if event.type == "delta" and event.field not in ("text", "reasoning"):
                        raise AssertionError("Unexpected public private-feedback delta")
            result = result.result
        emit("middleware_result", reasoning_delta_count=deltas, **message_summary(result))
        return result


async def run_case(label, operation):
    global CASE
    CASE = label
    start = time.monotonic()
    emit("case_start")
    try:
        value = await operation()
    except Exception as error:
        emit(
            "case_error",
            error_type=type(error).__name__,
            status=getattr(error, "status_code", None),
            message=safe(error)[:650],
            seconds=round(time.monotonic() - start, 2),
        )
        return None
    emit("case_success", seconds=round(time.monotonic() - start, 2))
    return value


async def validate_only():
    global SECRET
    SECRET = "dummy-local-validation-only"
    async with MyLLM(provider=provider(), model=model(), stdio_protocol="middleware") as llm:
        prepared = await llm.codec.prepare(
            {
                "messages": [PROMPT],
                "tools": [TOOL],
                "stream": True,
                "reasoning": True,
                "reasoning_effort": "low",
            },
            llm.resolved_profile,
            input_format="middleware",
            client=llm.client_parameters,
        )
        request = prepared["request"]
        assert request.get("thinking") == {"type": "enabled"} or request.get("extra_body", {}).get(
            "thinking"
        ) == {"type": "enabled"}
        assert request["max_tokens"] == 1024
        assert request["reasoning_effort"] == "low"
    SECRET = None
    emit("local_validation_passed")


async def main():
    global MODEL, CASE
    async with http_client() as http:
        CASE = "model_discovery"
        response = await http.get(BASE + "/models", headers={"Authorization": "Bearer " + SECRET})
        if response.status_code != 200:
            emit("discovery_failed", status=response.status_code, message=safe(response.text)[:500])
            return
        available = [m["id"] for m in response.json().get("data", [])]
        MODEL = next(
            (
                m
                for m in (
                    "deepseek-flash",
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                )
                if m in available
            ),
            None,
        )
        if MODEL is None:
            emit("no_selected_model", available=available)
            return
        emit("model_selected", model=MODEL, generation_token_limit=1024, automatic_retries=0)

    first = await run_case("myllm_stream_tool_call", lambda: generate([PROMPT], streaming=True))
    if first is None:
        return
    calls = [p for p in first.content if isinstance(p, ToolCall)]
    feedback = [p for p in first.content if isinstance(p, ReasoningContent)]
    if (
        len(calls) != 1
        or not feedback
        or not all(isinstance(p.feedback_item, str) for p in feedback)
    ):
        emit("insufficient_first_turn", tool_count=len(calls), feedback_blocks=len(feedback))
        return
    assert all(p.feedback_item == (p.reasoning or "") for p in feedback)
    tool_result = Message("user", [ToolResult(calls[0].call_id, [TextContent('{"value":719}')])])
    history = [PROMPT, first, tool_result]
    before = deepcopy(history)

    if "--isolated" not in sys.argv:
        await run_case("myllm_original_feedback", lambda: generate(history))
    assert history == before
    empty = deepcopy(history)
    for part in empty[1].content:
        if isinstance(part, ReasoningContent):
            part.feedback_item = None
    if "--isolated" not in sys.argv:
        await run_case("myllm_empty_feedback_known_id", lambda: generate(empty))
    assert all(p.feedback_item is None for p in empty[1].content if isinstance(p, ReasoningContent))

    codec = MyCodec()
    native = codec.decoder("openai_chat").messages(history)
    raw_tools = codec.decoder("openai_chat").tools([TOOL])
    async with AsyncOpenAI(
        api_key=SECRET, base_url=BASE, timeout=90, max_retries=0, http_client=http_client()
    ) as client:

        async def raw_request(messages):
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=raw_tools,
                max_tokens=1024,
                reasoning_effort="low",
                extra_body={"thinking": {"type": "enabled"}},
            )
            choice = response.choices[0]
            emit(
                "raw_result",
                finish_reason=choice.finish_reason,
                answer=(choice.message.content or "")[:200],
                completion_tokens=response.usage.completion_tokens if response.usage else None,
            )
            return True

        omitted = deepcopy(native)
        omitted[1].pop("reasoning_content", None)
        if "--isolated" not in sys.argv:
            await run_case("sdk_omitted_feedback_known_id", lambda: raw_request(omitted))

        used_ids = {calls[0].call_id}

        def unseen_history():
            value = deepcopy(native)
            new_id = "call_audit_" + secrets.token_hex(10)
            assert new_id not in used_ids
            used_ids.add(new_id)
            value[1]["tool_calls"][0]["id"] = new_id
            value[2]["tool_call_id"] = new_id
            return value

        unseen = unseen_history()
        await run_case("sdk_original_feedback_unseen_id", lambda: raw_request(unseen))
        unseen_empty = unseen_history()
        unseen_empty[1]["reasoning_content"] = ""
        await run_case("sdk_empty_feedback_unseen_id", lambda: raw_request(unseen_empty))
        unseen_omit = unseen_history()
        unseen_omit[1].pop("reasoning_content", None)
        await run_case("sdk_omitted_feedback_unseen_id", lambda: raw_request(unseen_omit))

    if "--isolated" not in sys.argv:
        await run_case("myllm_missing_tool_result_padding", lambda: generate([PROMPT, first]))
    emit(
        "audit_finished", actual_generation_requests=len(RECORDS), credential_written_to_disk=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate request preparation without a key or network",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="run only a new tool turn and independent-ID controls",
    )
    options = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    if options.validate:
        asyncio.run(validate_only())
    else:
        if not sys.stdin.isatty():
            raise SystemExit("A non-echoing terminal is required; no credential was read")
        ChatCompletionsTransport.create_client = client_factory
        SECRET = getpass.getpass("DeepSeek API key (memory only; no echo): ")
        try:
            asyncio.run(main())
        except BaseException as error:
            emit("audit_stopped", error_type=type(error).__name__, message=safe(error)[:650])
        finally:
            SECRET = None
