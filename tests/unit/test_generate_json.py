"""Structured output through the normal codec, cache and transport pipeline."""

import json
from copy import deepcopy
from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from my_llm import (
    AssistantMessage,
    FunctionTool,
    MyCodec,
    SemanticLossError,
    TextContent,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from my_llm.basic.exceptions import ConfigurationError, JSONGenerationError, MessageValidationError
from my_llm.utils.json_schema import object_json_schema, parse_json_object
from tests.helpers.json_output import (
    assistant_content,
    install_sdk,
    make_llm,
    response_events,
    structured_object,
)
from tests.helpers.protocols import chat_event, response_completed
from tests.helpers.protocols import history as protocol_history
from tests.helpers.protocols import tools as protocol_tools
from tests.helpers.transport import FakeClient


class Answer(BaseModel):
    city: str


@dataclass
class DataAnswer:
    city: str


class TypedAnswer(TypedDict):
    city: str


SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def inputs(stdio):
    if stdio == "middleware":
        return [UserMessage("user", [TextContent("Which city?")])]
    if stdio == "openai_responses":
        return [{"role": "user", "content": [{"type": "input_text", "text": "Which city?"}]}]
    return [{"role": "user", "content": "Which city?"}]


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic", "middleware"])
@pytest.mark.parametrize("use_tools", [True, False])
async def test_structured_output_matrix(monkeypatch, protocol, stdio, use_tools):
    clients = install_sdk(monkeypatch, protocol, tools=use_tools)
    llm = make_llm(protocol, stdio, tools=use_tools)
    messages = inputs(stdio)
    before = deepcopy(messages)
    assert structured_object(
        await llm.generate_json(messages=messages, schema=Answer, max_tokens=512),
        llm.stdio_protocol,
    ) == {"city": "上海"}
    assert messages == before and llm.stdio_protocol == stdio
    client = clients[0]
    assert client.closed and client.upstream.closed
    request = client.requests[0]
    assert bool(request.get("tools")) is use_tools
    assert "Which city?" in str(request)
    if protocol == "anthropic":
        instruction = request["system"]
        assert all(message["role"] != "system" for message in request["messages"])
    else:
        history = request["input" if protocol == "openai_responses" else "messages"]
        assert history[-1]["role"] == "system"
        instruction = history[-1]
    assert ("exactly once" if use_tools else "JSON Schema") in str(instruction)
    if use_tools:
        assert "structured_output" in str(request["tools"])
        assert "city" in str(request["tools"])
    else:
        assert "JSON Schema" in str(request)


def business_tools(stdio):
    if stdio == "middleware":
        return [FunctionTool("weather", "Weather", SCHEMA, strict=False)]
    return protocol_tools(stdio)


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic", "middleware"])
@pytest.mark.parametrize("fake_tool_name", ["structured_output", "final_answer"])
async def test_business_tools_are_preserved_with_named_output_tool(
    monkeypatch, protocol, stdio, fake_tool_name
):
    clients = install_sdk(monkeypatch, protocol, tools=True, tool_name=fake_tool_name)
    llm = make_llm(protocol, stdio)
    tools = business_tools(stdio)
    before = deepcopy(tools)
    assert structured_object(
        await llm.generate_json(
            messages=inputs(stdio),
            schema=SCHEMA,
            tools=tools,
            fake_tool_name=fake_tool_name,
            max_tokens=512,
        ),
        llm.stdio_protocol,
    ) == {"city": "上海"}
    assert tools == before
    request = clients[0].requests[0]
    declarations = [tool.get("function", tool) for tool in request["tools"]]
    assert [tool["name"] for tool in declarations] == ["weather", fake_tool_name]
    assert all(
        tool.get("parameters", tool.get("input_schema"))["properties"]["city"]["type"] == "string"
        for tool in declarations
    )
    instruction = (
        request["system"]
        if protocol == "anthropic"
        else request["input" if protocol == "openai_responses" else "messages"][-1]
    )
    assert fake_tool_name in str(instruction)
    assert clients[0].closed and clients[0].upstream.closed


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("policy", ["raise", "ignore", "hook"])
async def test_unsupported_business_tools_follow_existing_loss_policy(
    monkeypatch, protocol, policy
):
    clients = install_sdk(monkeypatch, protocol, tools=False)
    llm = make_llm(protocol, "openai_chat", tools=False)
    observed = []

    def hook(draft, losses, context):
        observed.append((draft, losses, context))

    llm.set_semantic_loss_policy(semantic_loss_protocol=policy, input_semantic_loss_hook=hook)
    tools = business_tools("openai_chat")
    before = deepcopy(tools)
    options = dict(
        messages=inputs("openai_chat"), schema=SCHEMA, tools=tools, max_tokens=512, max_retries=2
    )
    if policy == "raise":
        with pytest.raises(
            SemanticLossError, match="model does not support function calling"
        ) as caught:
            await llm.generate_json(**options)
        assert len(caught.value.losses) == 1
        assert "weather" in str(caught.value.losses.items[0].origin_value)
        assert not clients
    else:
        assert structured_object(await llm.generate_json(**options), llm.stdio_protocol) == {
            "city": "上海"
        }
        assert len(clients) == 1
        assert not clients[0].requests[0].get("tools")
        assert "JSON Schema" in str(clients[0].requests[0])
        assert clients[0].closed and clients[0].upstream.closed
        if policy == "hook":
            assert len(observed) == 1
            draft, losses, context = observed[0]
            assert not draft.get("tools")
            assert len(losses) == 1
            assert losses.items[0].reason == "model does not support function calling"
            assert context.target_protocol == protocol
        else:
            assert not observed
    assert tools == before


@pytest.mark.parametrize("schema", [Answer, DataAnswer, TypedAnswer, SCHEMA])
def test_schema_types(schema):
    parameters = object_json_schema(schema)
    assert parameters["type"] == "object"
    assert parameters["properties"]["city"]["type"] == "string"
    parameters["properties"]["city"]["type"] = "integer"
    assert object_json_schema(schema)["properties"]["city"]["type"] == "string"


@pytest.mark.parametrize("schema", [str, list, {"type": "array"}])
def test_non_object_schema(schema):
    with pytest.raises(ValueError, match="object"):
        object_json_schema(schema)


@pytest.mark.parametrize(
    "text",
    [
        '{"nested":{"items":[1,false,null]},"brace":"} and {","quote":"\\""}',
        'prefix ```json\n{"nested":{"items":[1,false,null]},"brace":"} and {","quote":"\\""}\n``` suffix',
    ],
)
def test_extract_nested_json(text):
    assert parse_json_object(text) == {
        "nested": {"items": [1, False, None]},
        "brace": "} and {",
        "quote": '"',
    }


@pytest.mark.parametrize(
    "text", ["no JSON", "[]", '[{"city":"上海"}]', '{"city":', "{'city': '上海'}"]
)
def test_invalid_output(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("use_tools", [True, False])
async def test_request_instructions_do_not_enter_history_cache(monkeypatch, protocol, use_tools):
    clients = install_sdk(monkeypatch, protocol, tools=use_tools)
    llm = make_llm(protocol, "openai_chat", tools=use_tools, cache=True)
    messages = inputs("openai_chat")
    for _ in range(2):
        assert structured_object(
            await llm.generate_json(messages=messages, schema=SCHEMA, max_tokens=512),
            llm.stdio_protocol,
        ) == {"city": "上海"}
    assert clients[0].requests == clients[1].requests
    await llm.generate(messages=[{"role": "user", "content": "Next question"}], max_tokens=512)
    history = clients[-1].requests[0]["input" if protocol == "openai_responses" else "messages"]
    assert len(history) == 2
    assert "system" not in clients[-1].requests[0]
    assert all(
        "JSON Schema" not in str(message) and "exactly once" not in str(message)
        for message in history
    )


async def test_missing_tool_is_error_without_retry(monkeypatch):
    clients = install_sdk(monkeypatch, "openai_chat", tools=False)
    llm = make_llm("openai_chat", "openai_chat")
    with pytest.raises(ValueError, match="exactly one"):
        await llm.generate_json(messages=inputs("openai_chat"), schema=SCHEMA)
    assert len(clients) == 1 and clients[0].closed


def install_attempts(monkeypatch, llm, clients):
    parameters = []

    def create_client(self, values):
        index = len(parameters)
        if index:
            assert clients[index - 1].closed and clients[index - 1].upstream.closed
        parameters.append(deepcopy(values))
        return clients[index]

    transport = llm._codec._transport(llm.resolved_profile.protocol)
    monkeypatch.setattr(type(transport), "create_client", create_client)
    return parameters


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
async def test_retry_preserves_business_tools_and_custom_output_name(monkeypatch, protocol):
    llm = make_llm(protocol, "openai_chat", cache=True)
    clients = [
        FakeClient(response_events(protocol, tools=False, text="not JSON")),
        FakeClient(response_events(protocol, tools=True, tool_name="final_answer")),
    ]
    install_attempts(monkeypatch, llm, clients)
    tools = business_tools("openai_chat")
    before = deepcopy(tools)
    assert structured_object(
        await llm.generate_json(
            messages=inputs("openai_chat"),
            schema=SCHEMA,
            tools=tools,
            fake_tool_name="final_answer",
            max_retries=1,
            max_tokens=512,
        ),
        llm.stdio_protocol,
    ) == {"city": "上海"}
    assert tools == before
    assert llm.get_last_json_failures()[0]["stage"] == "tool_call"
    assert "final_answer" in llm.get_last_json_failures()[0]["errors"][0]["message"]
    assert clients[0].requests[0]["tools"] == clients[1].requests[0]["tools"]
    assert len(clients[1].requests[0]["tools"]) == 2
    assert all(client.closed and client.upstream.closed for client in clients)


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic", "middleware"])
async def test_business_tool_turn_returns_complete_middleware_message(monkeypatch, protocol, stdio):
    clients = install_sdk(monkeypatch, protocol, tools=True, tool_name="weather", text="Checking")
    llm = make_llm(protocol, stdio)
    result = await llm.generate_json(
        messages=inputs(stdio),
        schema=SCHEMA,
        tools=business_tools(stdio),
        fake_tool_name="final_answer",
        max_retries=2,
        max_tokens=512,
    )
    content = assistant_content(result, stdio)
    calls = [part for part in content if isinstance(part, ToolCall)]
    assert len(calls) == 1
    assert (calls[0].call_id, calls[0].name, calls[0].arguments) == (
        "call_1",
        "weather",
        {"city": "上海"},
    )
    assert any(isinstance(part, TextContent) and part.text == "Checking" for part in content)
    assert llm.get_last_json_failures() == []
    usage = llm.get_last_usage()
    assert usage.input == 1 and usage.output == 2
    assert len(clients) == 1 and clients[0].closed and clients[0].upstream.closed


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("stdio", ["middleware", "openai_chat", "openai_responses", "anthropic"])
async def test_business_tool_round_continues_to_structured_result(
    monkeypatch, protocol, cache, stdio
):
    llm = make_llm(protocol, stdio, cache=cache)
    clients = [
        FakeClient(response_events(protocol, tools=True, tool_name="weather")),
        FakeClient(response_events(protocol, tools=True, tool_name="final_answer")),
    ]
    install_attempts(monkeypatch, llm, clients)
    history = inputs(stdio)
    options = dict(
        schema=SCHEMA,
        tools=business_tools(stdio),
        fake_tool_name="final_answer",
        max_tokens=512,
    )
    result = await llm.generate_json(messages=history, **options)
    if stdio == "middleware":
        assert isinstance(result, AssistantMessage)
        batch = [result, ToolMessage("tool", [TextContent("Sunny")], call_id="call_1")]
    else:
        tool_result = protocol_history(stdio)[-1]
        batch = [*result, tool_result] if stdio == "openai_responses" else [result, tool_result]
    continuation = batch if cache else [*history, *batch]
    assert structured_object(
        await llm.generate_json(messages=continuation, **options), llm.stdio_protocol
    ) == {"city": "上海"}
    request = clients[1].requests[0]
    assert "Sunny" in str(request) and "call_1" in str(request)
    assert request["tools"] == clients[0].requests[0]["tools"]
    assert all(client.closed and client.upstream.closed for client in clients)


@pytest.mark.parametrize("business_first", [False, True])
@pytest.mark.parametrize("stdio", ["middleware", "openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("valid_output", [False, True])
async def test_mixed_calls_validate_output_and_preserve_business_calls(
    monkeypatch, business_first, stdio, valid_output
):
    llm = make_llm("openai_chat", stdio)
    business = {
        "index": int(not business_first),
        "id": "business_call",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"上海"}'},
    }
    synthetic = {
        "index": int(business_first),
        "id": "output_call",
        "type": "function",
        "function": {
            "name": "final_answer",
            "arguments": '{"city":"上海"}' if valid_output else '{"city":42}',
        },
    }
    calls = [business, synthetic] if business_first else [synthetic, business]
    client = FakeClient([chat_event(calls=calls), chat_event(finish="tool_calls")])
    install_attempts(monkeypatch, llm, [client])
    options = dict(
        messages=inputs(stdio),
        schema=SCHEMA,
        tools=business_tools(stdio),
        fake_tool_name="final_answer",
    )
    if valid_output:
        result = await llm.generate_json(**options)
        content = assistant_content(result, stdio)
        returned = [part for part in content if isinstance(part, ToolCall)]
        assert [(call.call_id, call.name) for call in returned] == [("business_call", "weather")]
        assert structured_object(result, stdio) == {"city": "上海"}
        if stdio == "middleware":
            assert isinstance(content[0 if business_first else 1], ToolCall)
        assert llm.get_last_json_failures() == []
    else:
        with pytest.raises(JSONGenerationError) as caught:
            await llm.generate_json(**options)
        assert caught.value.failures[0]["stage"] == "schema"
        assert caught.value.failures[0]["errors"][0]["instance_path"] == "/city"
    assert client.closed and client.upstream.closed


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("stdio", ["middleware", "openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("business_first", [False, True])
async def test_failed_mixed_turns_keep_calls_and_use_normal_missing_results(
    monkeypatch, protocol, stdio, cache, business_first
):
    llm = make_llm(protocol, stdio, cache=cache)
    clients = []
    for attempt in range(2):
        calls = [
            (f"output_{attempt}", "final_answer", {"city": 42}),
            (f"business_{attempt}", "weather", {"city": "上海"}),
        ]
        if business_first:
            calls.reverse()
        if protocol == "openai_chat":
            events = [
                chat_event(
                    calls=[
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        for index, (call_id, name, arguments) in enumerate(calls)
                    ]
                ),
                chat_event(finish="tool_calls"),
            ]
        elif protocol == "openai_responses":
            events = [
                response_completed(
                    [
                        {
                            "type": "function_call",
                            "id": f"fc_{call_id}",
                            "call_id": call_id,
                            "name": name,
                            "arguments": json.dumps(arguments),
                            "status": "completed",
                        }
                        for call_id, name, arguments in calls
                    ]
                )
            ]
        else:
            events = response_events(protocol, tools=False)[:1]
            for index, (call_id, name, arguments) in enumerate(calls):
                events.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": name,
                                "input": arguments,
                            },
                        },
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            events.extend(response_events(protocol, tools=True)[-2:])
        clients.append(FakeClient(events))
    clients.extend(
        FakeClient(response_events(protocol, tools=True, tool_name="final_answer"))
        for _ in range(2)
    )
    install_attempts(monkeypatch, llm, clients)
    history = inputs(stdio)
    before = deepcopy(history)
    options = dict(
        messages=history,
        schema=SCHEMA,
        tools=business_tools(stdio),
        fake_tool_name="final_answer",
        max_retries=2,
        max_tokens=512,
    )
    assert structured_object(await llm.generate_json(**options), stdio) == {"city": "上海"}
    failures = llm.get_last_json_failures()
    for attempt in (1, 2):
        request = clients[attempt].requests[0]
        native = request["input" if protocol == "openai_responses" else "messages"]
        messages = MyCodec().encode(native, input_format=protocol).messages
        tool_results = {m.call_id: m for m in messages if isinstance(m, ToolMessage)}
        calls = [p for m in messages for p in m.content if isinstance(p, ToolCall)]
        assert len(calls) == 2 * attempt
        assert len(tool_results) == 2 * attempt
        for previous in range(attempt):
            assert tool_results[f"business_{previous}"].content == [] or all(
                isinstance(p, TextContent) and p.text == ""
                for p in tool_results[f"business_{previous}"].content
            )
            feedback = "".join(p.text for p in tool_results[f"output_{previous}"].content)
            assert json.loads(feedback.split("\n", 1)[1]) == failures[: previous + 1]
    assert history == before
    await llm.generate_json(**options)
    assert "business_0" not in str(clients[-1].requests[0])
    assert "output_0" not in str(clients[-1].requests[0])
    assert all(client.closed and client.upstream.closed for client in clients)


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.parametrize("cache", [False, True])
async def test_retry_succeeds_and_sums_usage(monkeypatch, protocol, use_tools, cache):
    llm = make_llm(protocol, "openai_chat", tools=use_tools, cache=cache)
    clients = [
        FakeClient(response_events(protocol, tools=False, text="not JSON")),
        FakeClient(response_events(protocol, tools=use_tools)),
    ]
    parameters = install_attempts(monkeypatch, llm, clients)
    assert structured_object(
        await llm.generate_json(
            messages=inputs("openai_chat"), schema=SCHEMA, max_retries=3, max_tokens=512
        ),
        llm.stdio_protocol,
    ) == {"city": "上海"}
    assert len(parameters) == 2
    assert all(p["max_retries"] == 7 for p in parameters)
    assert "Previous structured output attempts failed" in str(clients[1].requests)
    assert "Which city?" in str(clients[1].requests)
    assert len(llm.get_last_json_failures()) == 1
    assert all(client.closed and client.upstream.closed for client in clients)
    usage = llm.get_last_usage()
    assert (usage.input, usage.output) == (2, 4)
    assert usage.total == (None if protocol == "anthropic" else 6)
    usage.input = 999
    assert llm.get_last_usage().input == 2


@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.parametrize("retries", [0, 2])
async def test_retry_exhaustion_keeps_all_usage(monkeypatch, use_tools, retries):
    llm = make_llm("openai_chat", "openai_chat", tools=use_tools)
    clients = [
        FakeClient(response_events("openai_chat", tools=False, text="not JSON"))
        for _ in range(retries + 1)
    ]
    parameters = install_attempts(monkeypatch, llm, clients)
    with pytest.raises(ValueError):
        await llm.generate_json(messages=inputs("openai_chat"), schema=SCHEMA, max_retries=retries)
    assert len(parameters) == retries + 1
    assert llm.get_last_usage() == Usage(
        input=retries + 1, output=2 * (retries + 1), total=3 * (retries + 1)
    )
    assert all(client.closed and client.upstream.closed for client in clients)


@pytest.mark.parametrize("succeeds", [True, False])
async def test_malformed_generated_tool_arguments_retry_with_usage(monkeypatch, succeeds):
    llm = make_llm("openai_chat", "openai_chat")
    bad_events = response_events("openai_chat", tools=True)
    for event in bad_events:
        for choice in event["choices"]:
            for call in choice["delta"].get("tool_calls", []):
                function = call["function"]
                if function.get("arguments", "").startswith("{"):
                    function["arguments"] = "invalid" + function["arguments"]
    clients = [
        FakeClient(bad_events),
        FakeClient(response_events("openai_chat", tools=True) if succeeds else bad_events),
    ]
    install_attempts(monkeypatch, llm, clients)
    if succeeds:
        assert structured_object(
            await llm.generate_json(messages=inputs("openai_chat"), schema=SCHEMA, max_retries=1),
            llm.stdio_protocol,
        ) == {"city": "上海"}
    else:
        with pytest.raises(JSONGenerationError, match="function arguments"):
            await llm.generate_json(messages=inputs("openai_chat"), schema=SCHEMA, max_retries=1)
    assert llm.get_last_usage() == Usage(input=2, output=4, total=6)


@pytest.mark.parametrize("max_retries", [-1, True, 1.5, None, "2"])
async def test_invalid_retry_limit_is_not_retried(monkeypatch, max_retries):
    clients = install_sdk(monkeypatch, "openai_chat", tools=False)
    llm = make_llm("openai_chat", "openai_chat", tools=False)
    with pytest.raises(ConfigurationError, match="max_retries"):
        await llm.generate_json(
            messages=inputs("openai_chat"), schema=SCHEMA, max_retries=max_retries
        )
    assert not clients


async def test_invalid_input_is_not_retried(monkeypatch):
    clients = install_sdk(monkeypatch, "openai_chat", tools=False)
    llm = make_llm("openai_chat", "openai_chat", tools=False)
    with pytest.raises(MessageValidationError):
        await llm.generate_json(messages=[], schema=SCHEMA, max_retries=2)
    assert not clients


async def test_sdk_value_error_is_not_a_json_retry(monkeypatch):
    llm = make_llm("openai_chat", "openai_chat", tools=False)
    failure = ValueError("SDK failure")
    clients = [FakeClient([], failure=failure)]
    parameters = install_attempts(monkeypatch, llm, clients)
    with pytest.raises(ValueError) as caught:
        await llm.generate_json(messages=inputs("openai_chat"), schema=SCHEMA, max_retries=2)
    assert caught.value is failure and len(parameters) == 1
    assert clients[0].closed and clients[0].upstream.closed
    assert llm.get_last_usage() is None


async def test_retry_usage_preserves_missing_counters_and_does_not_include_previous_calls(
    monkeypatch,
):
    llm = make_llm("openai_chat", "openai_chat", tools=False)
    events = [
        [
            chat_event(content="not JSON"),
            chat_event(finish="stop"),
            {**chat_event(usage={"prompt_tokens": 4}), "choices": []},
        ],
        [
            chat_event(content='{"ok":true}'),
            chat_event(finish="stop"),
            {
                **chat_event(
                    usage={
                        "completion_tokens": 2,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    }
                ),
                "choices": [],
            },
        ],
        [chat_event(content='{"ok":true}'), chat_event(finish="stop")],
    ]
    clients = [FakeClient(batch) for batch in events]
    install_attempts(monkeypatch, llm, clients)
    assert structured_object(
        await llm.generate_json(
            messages=inputs("openai_chat"),
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            max_retries=1,
        ),
        llm.stdio_protocol,
    ) == {"ok": True}
    assert llm.get_last_usage() == Usage(input=4, output=2, reasoning=0)
    assert structured_object(
        await llm.generate_json(
            messages=inputs("openai_chat"),
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        ),
        llm.stdio_protocol,
    ) == {"ok": True}
    assert llm.get_last_usage() is None


async def test_retry_usage_sums_all_reported_counters(monkeypatch):
    llm = make_llm("openai_chat", "openai_chat", tools=False)
    clients = []
    for text, cached, reasoning in [("invalid", 2, 3), ('{"ok":true}', 5, 7)]:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        }
        clients.append(
            FakeClient(
                [
                    chat_event(content=text),
                    chat_event(finish="stop"),
                    {**chat_event(usage=usage), "choices": []},
                ]
            )
        )
    install_attempts(monkeypatch, llm, clients)
    assert structured_object(
        await llm.generate_json(
            messages=inputs("openai_chat"),
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            max_retries=1,
        ),
        llm.stdio_protocol,
    ) == {"ok": True}
    assert llm.get_last_usage() == Usage(input=20, output=40, total=60, cached=7, reasoning=10)


@pytest.mark.parametrize(
    "invalid_text",
    [
        '{"outer":{"city":"上海"}',
        'Result: [{"city":"上海"}]',
        '{"city":"上海"}{"city":"北京"}',
        '{"city":"上海","city":"北京"}',
        '{"outer":{"city":"上海"},"value":NaN}',
    ],
)
async def test_ambiguous_or_partial_json_triggers_retry(monkeypatch, invalid_text):
    llm = make_llm("openai_chat", "openai_chat", tools=False)
    clients = [
        FakeClient(response_events("openai_chat", tools=False, text=invalid_text)),
        FakeClient(response_events("openai_chat", tools=False)),
    ]
    parameters = install_attempts(monkeypatch, llm, clients)
    assert structured_object(
        await llm.generate_json(messages=inputs("openai_chat"), schema=SCHEMA, max_retries=1),
        llm.stdio_protocol,
    ) == {"city": "上海"}
    assert len(parameters) == 2
    assert llm.get_last_usage() == Usage(input=2, output=4, total=6)


class NestedAnswer(BaseModel):
    details: Answer


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic"])
@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic", "middleware"])
@pytest.mark.parametrize("use_tools", [False, True])
async def test_nested_schema_is_inlined_before_protocol_translation(
    monkeypatch, protocol, stdio, use_tools
):
    def nested_response(value):
        if isinstance(value, dict):
            return {key: nested_response(child) for key, child in value.items()}
        if isinstance(value, list):
            return [nested_response(child) for child in value]
        if isinstance(value, str):
            return value.replace('{"city":', '{"details":{"city":').replace('"上海"}', '"上海"}}')
        return value

    llm = make_llm(protocol, stdio, tools=use_tools)
    clients = [FakeClient(nested_response(response_events(protocol, tools=use_tools)))]
    install_attempts(monkeypatch, llm, clients)
    assert structured_object(
        await llm.generate_json(messages=inputs(stdio), schema=NestedAnswer, max_tokens=512),
        llm.stdio_protocol,
    ) == {"details": {"city": "上海"}}
    request = clients[0].requests[0]
    assert "$ref" not in json.dumps(request)
    assert "$defs" not in json.dumps(request)
    if use_tools:
        tool = request["tools"][0]
        function = tool.get("function", tool)
        schema = function.get("parameters", function.get("input_schema"))
        assert schema["properties"]["details"]["type"] == "object"
        assert schema["properties"]["details"]["properties"]["city"]["type"] == "string"
