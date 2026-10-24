"""Protocol and semantic checks performed outside benchmark measurements."""

import base64
import json

from jsonschema import Draft202012Validator

from my_llm import ReasoningContent, TextContent, ToolCall
from tests.helpers.schema import validate_request
from tests.helpers.sdk import EVENT_TYPES, MESSAGE_TYPES, TOOL_TYPES, assert_native
from tests.performance.workloads import PNG, Workload, arguments, parameters


def validate_workload(workload: Workload):
    for message in workload.messages:
        assert_native(message, MESSAGE_TYPES[workload.stdio])
    for tool in workload.tools:
        assert_native(tool, TOOL_TYPES[workload.stdio])
    for event in workload.events:
        schema = EVENT_TYPES[workload.protocol]
        assert_native(event, schema[event["type"]] if isinstance(schema, dict) else schema)
    argument_schema = parameters(
        workload.size, extended=workload.native and workload.protocol != "ollama"
    )
    for call in workload.calls:
        Draft202012Validator(argument_schema).validate(call["input"])


def result_parts(protocol: str, result):
    """Extract public fields directly, independently of MyCodec conversion."""
    if protocol == "middleware":
        return (
            "".join(part.text for part in result.content if isinstance(part, TextContent)),
            "".join(
                part.reasoning for part in result.content if isinstance(part, ReasoningContent)
            ),
            [(part.name, part.arguments) for part in result.content if isinstance(part, ToolCall)],
        )
    messages = result if isinstance(result, list) else [result]
    texts, thoughts, calls = [], [], []
    for message in messages:
        if protocol == "openai_responses":
            kind = message["type"]
            if kind == "message":
                texts.extend(
                    part["text"] for part in message["content"] if part["type"] == "output_text"
                )
            elif kind == "reasoning":
                thoughts.extend(part["text"] for part in message["summary"])
            elif kind == "function_call":
                calls.append((message["name"], json.loads(message["arguments"])))
        elif protocol == "anthropic":
            for block in message["content"]:
                if block["type"] == "text":
                    texts.append(block["text"])
                elif block["type"] == "thinking":
                    thoughts.append(block["thinking"])
                elif block["type"] == "tool_use":
                    calls.append((block["name"], block["input"]))
        elif protocol == "gemini":
            for part in message["parts"]:
                if "text" in part:
                    (thoughts if part.get("thought") else texts).append(part["text"])
                if "function_call" in part:
                    call = part["function_call"]
                    calls.append((call["name"], call["args"]))
        else:
            texts.append(message.get("content") or "")
            thoughts.append(
                message.get("reasoning_content" if protocol == "openai_chat" else "thinking", "")
            )
            for call in message.get("tool_calls", []):
                function = call["function"]
                calls.append(
                    (
                        function["name"],
                        json.loads(function["arguments"])
                        if protocol == "openai_chat"
                        else function["arguments"],
                    )
                )
    return "".join(texts), "".join(thoughts), calls


def validate_result(workload: Workload, stdio: str, result):
    actual_text, actual_reason, calls = result_parts(stdio, result)
    assert actual_text == workload.answer
    assert actual_reason == workload.reasoning
    assert calls == [(call["name"], call["input"]) for call in workload.calls]


def result_value(value):
    """Compare tool-result text independently of the target's native envelope."""
    if isinstance(value, list):
        assert all(part.get("type") in ("text", "input_text") for part in value)
        value = "".join(part["text"] for part in value)
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict) and set(value) == {"result"}:
        return result_value(value["result"])
    return value


def validate_exchange(workload: Workload, stdio: str, result, sdk):
    validate_result(workload, stdio, result)
    assert len(sdk.requests) == len(sdk.bodies)
    assert sdk.bodies[-1].closed
    assert sdk.bodies[-1].read == len(sdk.frames)
    body = json.loads(sdk.requests[-1].content)
    validate_request(workload.protocol, body)
    assert sdk.requests[-1].headers["x-benchmark"] == "request"
    if workload.protocol == "gemini":
        assert "/models/benchmark:" in sdk.requests[-1].url.path
    else:
        assert body["model"] == "benchmark"
    assert all("__my_llm_codec_" not in key for key in body)
    protocol = workload.protocol
    if protocol == "openai_responses":
        functions = body["tools"]
        calls = [
            (item["name"], json.loads(item["arguments"]))
            for item in body["input"]
            if item.get("type") == "function_call"
        ]
        results = [item for item in body["input"] if item.get("type") == "function_call_output"]
        result_values = [item["output"] for item in results]
        schemas = [function["parameters"] for function in functions]
        images = [
            part["image_url"].partition(",")[2]
            for item in body["input"]
            if isinstance(item.get("content"), list)
            for part in item["content"]
            if part["type"] == "input_image"
        ]
    elif protocol == "anthropic":
        functions = body["tools"]
        blocks = [block for message in body["messages"] for block in message["content"]]
        calls = [(block["name"], block["input"]) for block in blocks if block["type"] == "tool_use"]
        results = [block for block in blocks if block["type"] == "tool_result"]
        result_values = [block["content"] for block in results]
        schemas = [function["input_schema"] for function in functions]
        images = [block["source"]["data"] for block in blocks if block["type"] == "image"]
    elif protocol == "gemini":
        functions = [
            function for tool in body["tools"] for function in tool["functionDeclarations"]
        ]
        parts = [part for message in body["contents"] for part in message["parts"]]
        calls = [
            (part["functionCall"]["name"], part["functionCall"]["args"])
            for part in parts
            if "functionCall" in part
        ]
        results = [part for part in parts if "functionResponse" in part]
        result_values = [part["functionResponse"]["response"] for part in results]
        schemas = [function["parameters_json_schema"] for function in functions]
        images = [
            part["inlineData"]["data"]
            for part in parts
            if part.get("inlineData", {}).get("mimeType") == "image/png"
        ]
    else:
        functions = [tool["function"] for tool in body["tools"]]
        calls = [
            (
                call["function"]["name"],
                json.loads(call["function"]["arguments"])
                if protocol == "openai_chat"
                else call["function"]["arguments"],
            )
            for message in body["messages"]
            for call in message.get("tool_calls", [])
        ]
        results = [message for message in body["messages"] if message["role"] == "tool"]
        result_values = [message["content"] for message in results]
        schemas = [function["parameters"] for function in functions]
        images = (
            [
                part["image_url"]["url"].partition(",")[2]
                for message in body["messages"]
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part["type"] == "image_url"
            ]
            if protocol == "openai_chat"
            else [image for message in body["messages"] for image in message.get("images", [])]
        )
    assert len(images) == (workload.size.rounds - 1) // 4 + 2
    assert all(
        base64.b64decode(image, altchars=b"-_", validate=True) == base64.b64decode(PNG)
        for image in images
    )
    assert [function["name"] for function in functions] == [
        f"analyze_{index}" for index in range(workload.size.tools)
    ]
    expected_schema = parameters(workload.size, extended=workload.native and protocol != "ollama")
    assert all(schema == expected_schema for schema in schemas)
    expected_calls = [
        (f"analyze_{index}", arguments(workload.size, index))
        for _ in range(workload.size.rounds)
        for index in range(workload.size.parallel_calls)
    ]
    assert calls == expected_calls
    assert len(results) == len(expected_calls)
    expected_output = {
        "status": "ok",
        "records": [
            arguments(workload.size, index) for index in range(workload.size.parallel_calls)
        ],
    }
    assert all(result_value(value) == expected_output for value in result_values)
