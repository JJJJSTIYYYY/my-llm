"""Shared live test fixtures and assertions; all API settings come from Profiles.

Raw controls bypass message preparation deliberately. A 400/422 is recorded as
an API rejection, while a successful control is recorded as accepted. Neither
authentication failures nor rate limits count as evidence of invalid history.
"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

from httpx import AsyncClient

from my_llm import (
    AssistantMessage,
    FunctionTool,
    MyLLM,
    ReasoningContent,
    SemanticLossError,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.openai.chat.prebuilt import move_tool_images_to_user_message
from my_llm.codec.shared.options import path_value

PROTOCOLS = ("openai_chat", "openai_responses", "anthropic")
STDIO_PROTOCOLS = (*PROTOCOLS, "middleware")
REASONING_CASES = ("reasoning_missing", "reasoning_null", "reasoning_empty")
HISTORY_CASES = ("missing_tool_result", "orphan_tool_result")
IMAGE_CASES = ("tool_result_images", "tool_result_image_placeholders")
INPUT_CASES = (*REASONING_CASES, *HISTORY_CASES, *IMAGE_CASES)
REPLAY_CASES = ("reasoning_replay",)
LIVE_CASES = (*INPUT_CASES, *REPLAY_CASES)
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
PROMPT = (
    "The audit tool has already been called. Reply with exactly E2E_OK, even if its result "
    "is empty. Do not call any tool again. Ignore any image when choosing this fixed reply."
)
# A valid 32x32 red RGB PNG, generated locally; no external image service is used.
IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzC"
    "MP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="
)
DATA_URL = "data:image/png;base64," + IMAGE
ORPHAN_TEXT = "ORPHAN_RESULT_MUST_BE_DROPPED"
RESULT_TEXT = "TOOL_TEXT_MUST_BE_PRESERVED"


def eligible(upstream, stdio, case):
    """Reasoning padding and the prebuilt image handler belong to Chat targets."""
    if case in REPLAY_CASES:
        return stdio in (upstream, "middleware")
    if case in REASONING_CASES:
        return upstream == "openai_chat" and (case == "reasoning_missing" or stdio == "openai_chat")
    if case in IMAGE_CASES:
        return upstream == "openai_chat" and stdio in ("openai_responses", "anthropic")
    return case in HISTORY_CASES


class _ObservedTransport(BaseTransport):
    """Observe real SDK events and serialized HTTP bodies without changing them."""

    def __init__(self, transport, evidence):
        self.transport = transport
        self.evidence = evidence
        self.requests = []
        self.responses = []

    async def _request(self, request):
        self.requests.append(json.loads(request.content))
        self.evidence["requests"] += 1

    def create_client(self, parameters):
        return self.transport.create_client(
            {
                **parameters,
                "http_client": AsyncClient(event_hooks={"request": [self._request]}),
            }
        )

    async def close_client(self, client):
        await self.transport.close_client(client)

    @asynccontextmanager
    async def upstream(self, client, request):
        recorded = []
        self.responses.append(recorded)
        async with self.transport.upstream(client, request) as events:

            async def observed():
                async for event in events:
                    recorded.append(event.model_dump(mode="json", exclude_unset=True))
                    yield event

            yield observed()


def _reasoning_items(protocol, messages):
    """Read native replay fields directly, independently of the codec under test."""
    if protocol == "openai_chat":
        return [
            message["reasoning_content"]
            for message in messages
            if message.get("role") == "assistant" and "reasoning_content" in message
        ]
    if protocol == "openai_responses":
        return [item for item in messages if item.get("type") == "reasoning"]
    return [
        part
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") in ("thinking", "redacted_thinking")
    ]


def _source_reasoning(protocol, events, profile):
    if protocol == "openai_chat":
        binding = profile.parameter("reasoning_content", "output")
        return [
            "".join(
                value
                for event in events
                if isinstance(value := path_value(event, binding.path), str)
            )
        ]
    if protocol == "openai_responses":
        completed = [
            event["response"]["output"]
            for event in events
            if event.get("type") == "response.completed"
        ]
        assert completed, "missing original completed Responses output"
        return _reasoning_items(protocol, completed[-1])
    blocks = {}
    for event in events:
        kind = event.get("type")
        if kind == "content_block_start":
            part = event["content_block"]
            if part["type"] in ("thinking", "redacted_thinking"):
                blocks[event["index"]] = deepcopy(part)
        elif kind == "content_block_delta" and event["index"] in blocks:
            delta = event["delta"]
            field = {"thinking_delta": "thinking", "signature_delta": "signature"}.get(
                delta["type"]
            )
            if field:
                block = blocks[event["index"]]
                block[field] = block.get(field, "") + delta[field]
    return [blocks[index] for index in sorted(blocks)]


def _has_reasoning(items):
    for item in items:
        if isinstance(item, str) and item.strip():
            return True
        if isinstance(item, dict):
            if (
                item.get("encrypted_content")
                or item.get("data")
                or item.get("thinking", "").strip()
            ):
                return True
            if any(
                part.get("text", "").strip()
                for key in ("content", "summary")
                for part in item.get(key) or []
            ):
                return True
    return False


def _fingerprint(items):
    return sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _tools(protocol):
    if protocol == "middleware":
        return [FunctionTool("audit_value", "Read the audit value.", SCHEMA, strict=False)]
    if protocol == "openai_chat":
        return [{"type": "function", "function": {"name": "audit_value", "parameters": SCHEMA}}]
    if protocol == "openai_responses":
        return [{"type": "function", "name": "audit_value", "parameters": SCHEMA, "strict": False}]
    return [{"name": "audit_value", "input_schema": SCHEMA}]


def _replay_turn(stdio, result, value):
    if stdio == "middleware":
        calls = [part for part in result.content if isinstance(part, ToolCall)]
        assert len(calls) == 1 and calls[0].name == "audit_value" and calls[0].arguments == {}
        return [result, ToolMessage("user", [TextContent(value)], call_id=calls[0].call_id)]
    if stdio == "openai_chat":
        calls = result.get("tool_calls", [])
        assert len(calls) == 1 and calls[0]["function"]["name"] == "audit_value"
        assert json.loads(calls[0]["function"]["arguments"]) == {}
        return [result, {"role": "tool", "tool_call_id": calls[0]["id"], "content": value}]
    if stdio == "openai_responses":
        calls = [item for item in result if item["type"] == "function_call"]
        assert len(calls) == 1 and calls[0]["name"] == "audit_value"
        assert json.loads(calls[0]["arguments"]) == {}
        return [
            *result,
            {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": value},
        ]
    calls = [part for part in result["content"] if part["type"] == "tool_use"]
    assert len(calls) == 1 and calls[0]["name"] == "audit_value" and calls[0]["input"] == {}
    return [
        result,
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"], "content": value}],
        },
    ]


async def _collect(llm, messages, tools, stream, request_interval):
    await asyncio.sleep(request_interval)
    result = await llm.generate(messages=messages, tools=tools, stream=stream)
    if stream:
        async with result:
            async for _ in result:
                pass
        result = result.result
    return result


async def replay_scenario(provider, model, stdio, *, request_interval=0.0, evidence=None):
    """Require nonempty real reasoning to survive both turns without padding."""
    evidence = {} if evidence is None else evidence
    evidence.update(requests=0, roundtrips=[])
    prompt_text = (
        "Think briefly about why an external lookup is necessary, then call audit_value "
        "exactly once with no arguments to obtain an unknown number. After receiving its "
        "result, reply with that number only and do not call the tool again."
    )
    for stream in (False, True):
        async with MyLLM(
            provider=provider, model=model, stdio_protocol=stdio, semantic_loss_protocol="raise"
        ) as llm:
            upstream = llm.resolved_profile.protocol
            observed = _ObservedTransport(llm.codec.transport(upstream), evidence)
            llm.codec.codecs[upstream].transport = observed
            prompt = (
                UserMessage("user", [TextContent(prompt_text)])
                if stdio == "middleware"
                else {"role": "user", "content": prompt_text}
            )
            tools = _tools(stdio)
            first = await _collect(llm, [prompt], tools, stream, request_interval)
            source = _source_reasoning(upstream, observed.responses[0], llm.resolved_profile)
            assert _has_reasoning(source), (
                "upstream returned no nonempty reasoning; padding cannot satisfy this test"
            )
            returned = (
                [part.feedback_item for part in first.content if isinstance(part, ReasoningContent)]
                if stdio == "middleware"
                else _reasoning_items(stdio, first if stdio == "openai_responses" else [first])
            )
            record = {
                "stream": stream,
                "source_items": len(source),
                "source_sha256": _fingerprint(source),
                "returned_sha256": _fingerprint(returned),
            }
            evidence["roundtrips"].append(record)
            if stdio == "middleware" and upstream == "openai_chat":
                assert [
                    part.reasoning for part in first.content if isinstance(part, ReasoningContent)
                ] == source, "visible Chat reasoning changed"
            record["returned_exact"] = returned == source
            assert record["returned_exact"], "MyLLM's first result changed the original reasoning"
            value = str(int(uuid4().hex[:7], 16))
            messages = [prompt, *_replay_turn(stdio, first, value)]
            before = deepcopy(messages)
            prepared = await prepare(llm, messages, tools)
            prepared_items = _reasoning_items(
                upstream, prepared["input" if upstream == "openai_responses" else "messages"]
            )
            record["prepared_exact"] = prepared_items == source
            record["prepared_sha256"] = _fingerprint(prepared_items)
            assert record["prepared_exact"], (
                "next request preparation changed the original reasoning"
            )
            final = await _collect(llm, messages, tools, stream, request_interval)
            assert len(observed.requests) == 2, "unexpected number of real HTTP requests"
            body = observed.requests[1]
            replayed = _reasoning_items(
                upstream, body["input" if upstream == "openai_responses" else "messages"]
            )
            record["wire_exact"] = replayed == source
            record["wire_sha256"] = _fingerprint(replayed)
            assert record["wire_exact"], "actual HTTP request did not replay the original reasoning"
            assert _has_reasoning(replayed), "empty reasoning padding reached the wire"
            paired = wire_summary(upstream, body)
            assert paired["calls"] == paired["results"] == 1 and paired["ids_paired"]
            assert paired["empty_results"] == 0
            assert answer(final, stdio).strip() == value, "tool roundtrip did not complete"
            assert messages == before, "replay changed caller history"
            record["nonempty_replay"] = True
    return evidence


def history(protocol, case, call_id):
    """Construct each input format independently, without using MyCodec as an oracle."""
    orphan_id = "call_orphan_" + uuid4().hex
    missing = case == "missing_tool_result"
    orphan = case == "orphan_tool_result"
    image = case in IMAGE_CASES
    if protocol == "middleware":
        result = ToolMessage("user", [TextContent(RESULT_TEXT)], call_id=call_id)
        messages = [
            UserMessage("user", [TextContent(PROMPT)]),
            AssistantMessage("assistant", [ToolCall(call_id, "audit_value", {})]),
        ]
        if not missing:
            messages.append(result)
        if orphan:
            messages.append(ToolMessage("user", [TextContent(ORPHAN_TEXT)], call_id=orphan_id))
    elif protocol == "openai_chat":
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "audit_value", "arguments": "{}"},
                }
            ],
        }
        if case == "reasoning_null":
            assistant["reasoning_content"] = None
        elif case == "reasoning_empty":
            assistant["reasoning_content"] = ""
        messages = [{"role": "user", "content": PROMPT}, assistant]
        if not missing:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": RESULT_TEXT})
        if orphan:
            messages.append({"role": "tool", "tool_call_id": orphan_id, "content": ORPHAN_TEXT})
    elif protocol == "openai_responses":
        parts = [{"type": "input_text", "text": RESULT_TEXT}]
        if image:
            parts.append({"type": "input_image", "image_url": DATA_URL, "detail": "auto"})
        messages = [
            {"role": "user", "content": PROMPT},
            {"type": "function_call", "call_id": call_id, "name": "audit_value", "arguments": "{}"},
        ]
        if not missing:
            messages.append({"type": "function_call_output", "call_id": call_id, "output": parts})
        if orphan:
            messages.append(
                {"type": "function_call_output", "call_id": orphan_id, "output": ORPHAN_TEXT}
            )
    else:
        assert protocol == "anthropic"
        parts = [{"type": "text", "text": RESULT_TEXT}]
        if image:
            parts.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": IMAGE},
                }
            )
        messages = [
            {"role": "user", "content": PROMPT},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": call_id, "name": "audit_value", "input": {}}
                ],
            },
        ]
        if not missing:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": call_id, "content": parts}],
                }
            )
        if orphan:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": orphan_id, "content": ORPHAN_TEXT}
                    ],
                }
            )
    return messages, _tools(protocol)


def wire_summary(protocol, request):
    """Summarize the prepared native request without credentials or reasoning text."""
    messages = request["input" if protocol == "openai_responses" else "messages"]
    if protocol == "openai_chat":
        assistants = [m for m in messages if m.get("role") == "assistant"]
        calls = [c["id"] for m in assistants for c in m.get("tool_calls", [])]
        results = [(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"]
        reasoning = [
            {"present": "reasoning_content" in m, "empty": m.get("reasoning_content") == ""}
            for m in assistants
            if m.get("tool_calls")
        ]
    elif protocol == "openai_responses":
        calls = [m["call_id"] for m in messages if m.get("type") == "function_call"]
        results = [
            (m["call_id"], m["output"]) for m in messages if m.get("type") == "function_call_output"
        ]
        reasoning = []
    else:
        blocks = [part for m in messages for part in m["content"] if isinstance(part, dict)]
        calls = [part["id"] for part in blocks if part["type"] == "tool_use"]
        results = [
            (part["tool_use_id"], part.get("content"))
            for part in blocks
            if part["type"] == "tool_result"
        ]
        reasoning = []
    return {
        "calls": len(calls),
        "results": len(results),
        "ids_paired": len(results) == len(calls) and set(calls) == {key for key, _ in results},
        "empty_results": sum(value in (None, "", []) for _, value in results),
        "reasoning": reasoning,
        "orphan_marker": ORPHAN_TEXT in json.dumps(request),
    }


async def prepare(llm, messages, tools):
    return (
        await llm.codec.prepare(
            {"messages": messages, "tools": tools, "stream": True},
            llm.resolved_profile,
            input_format=llm.stdio_protocol,
            client=llm.client_parameters,
        )
    )["request"]


async def raw_control(llm, case, *, request_interval):
    """Use Profile-generated options but send the invalid fixture without repair."""
    profile = llm.resolved_profile
    protocol = profile.protocol
    messages, tools = history(protocol, case, "call_raw_" + uuid4().hex)
    decoder = llm.codec.decoder(protocol)
    request = decoder.assemble_request(
        messages,
        tools,
        profile,
        request=decoder.request_parameters(profile, decoder.validate_parameters(profile)),
    )
    summary = wire_summary(protocol, request)
    transport = llm.codec.codecs[protocol].transport
    client = transport.create_client(llm.client_parameters)
    try:
        await asyncio.sleep(request_interval)
        events = 0
        async with transport.upstream(client, request) as stream:
            async for event in stream:
                kind = getattr(event, "type", "")
                if kind in ("error", "response.failed", "response.incomplete"):
                    raise AssertionError(f"raw control returned {kind}")
                events += 1
        assert events, "raw API response had no events"
        return {"outcome": "accepted", "http_status": 200, "events": events, "request": summary}
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status not in (400, 422):
            raise
        key = llm.provider.api_key.default_value
        message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(error).replace(key, "[REDACTED]"))
        expected_fields = (
            ("reasoning", "thinking") if case in REASONING_CASES else ("tool", "call", "function")
        )
        if not any(field in message.lower() for field in expected_fields):
            raise
        return {
            "outcome": "rejected",
            "http_status": status,
            "error": message[:1200],
            "request": summary,
        }
    finally:
        await transport.close_client(client)


def answer(result, stdio):
    if stdio == "middleware":
        assert isinstance(result, AssistantMessage)
        assert not any(isinstance(part, ToolCall) for part in result.content)
        return "".join(part.text for part in result.content if isinstance(part, TextContent))
    if stdio == "openai_chat":
        assert not result.get("tool_calls"), "model called a tool instead of completing"
        return result.get("content") or ""
    if stdio == "openai_responses":
        assert not any(item["type"] == "function_call" for item in result)
        return "".join(
            part["text"]
            for item in result
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
    assert not any(part["type"] == "tool_use" for part in result["content"])
    return "".join(part["text"] for part in result["content"] if part["type"] == "text")


async def complete(llm, messages, tools, *, stream, request_interval):
    await asyncio.sleep(request_interval)
    result = await llm.generate(messages=messages, tools=tools, stream=stream)
    events = 0
    if stream:
        async with result:
            async for _ in result:
                events += 1
        assert events and result.result is not None, "MyLLM stream did not finish"
        result = result.result
    value = answer(result, llm.stdio_protocol).strip()
    assert value == "E2E_OK", f"unexpected answer: {value[:160]!r}"
    return {"stream": stream, "text": value, "events": events}


async def input_scenario(
    provider,
    model,
    stdio,
    case,
    *,
    request_interval=0.0,
    offline=False,
    semantic_loss_protocol="ignore",
    evidence=None,
):
    """Compare the raw API with MyLLM, asserting repairs before any live retry."""
    image = case in IMAGE_CASES
    evidence = {} if evidence is None else evidence
    evidence["requests"] = 0
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol=stdio,
        semantic_loss_protocol="hook" if image else semantic_loss_protocol,
        input_semantic_loss_hook=move_tool_images_to_user_message if image else None,
    ) as llm:
        messages, tools = history(stdio, case, "call_myllm_" + uuid4().hex)
        before = deepcopy((messages, tools))
        if image:
            llm.set_semantic_loss_policy(semantic_loss_protocol="raise")
            try:
                await prepare(llm, messages, tools)
            except SemanticLossError as error:
                assert any("image" in loss.reason for loss in error.losses)
                losses = len(error.losses.items)
            else:
                raise AssertionError("mixed tool-result image did not trigger semantic loss")
            llm.set_semantic_loss_policy(
                semantic_loss_protocol="hook",
                input_semantic_loss_hook=move_tool_images_to_user_message,
            )
            control = {"outcome": "local_semantic_loss", "losses": losses}
        else:
            evidence["requests"] += int(not offline)
            control = (
                {"outcome": "not_sent_offline"}
                if offline
                else await raw_control(llm, case, request_interval=request_interval)
            )
        evidence["raw_api"] = control
        request = await prepare(llm, messages, tools)
        summary = wire_summary(llm.resolved_profile.protocol, request)
        evidence["prepared_request"] = summary
        assert summary["calls"] == summary["results"] == 1 and summary["ids_paired"], summary
        assert not summary["orphan_marker"], "orphan tool-result reached the request"
        if case == "missing_tool_result":
            assert summary["empty_results"] == 1, "MyLLM did not pad the missing result"
        if case in REASONING_CASES:
            assert summary["reasoning"] == [{"present": True, "empty": True}], summary
        if image:
            native = request["messages"]
            tool = next(m for m in native if m["role"] == "tool")
            assert tool["content"] == RESULT_TEXT
            assert native[-1]["role"] == "user"
            if case == "tool_result_images":
                assert native[-1]["content"] == [
                    {"type": "image_url", "image_url": {"url": DATA_URL}}
                ]
            else:
                assert native[-1]["content"] == "[Unsupported Image]"
            assert (
                native.index(tool)
                == next(i for i, m in enumerate(native) if m.get("tool_calls")) + 1
            )
        completions = evidence["completions"] = []
        for stream in () if offline else (False, True):
            evidence["requests"] += 1
            completions.append(
                await complete(
                    llm, messages, tools, stream=stream, request_interval=request_interval
                )
            )
        assert (messages, tools) == before, "example or translation changed caller input"
        return evidence
