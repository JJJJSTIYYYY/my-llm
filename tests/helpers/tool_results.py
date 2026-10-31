"""Independent native tool-result fixtures and expected text extraction."""

import json


def application_result_history(protocol, payload):
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    if protocol == "openai_chat":
        return [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c", "content": payload},
        ]
    if protocol == "openai_responses":
        return [
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": payload},
        ]
    if protocol == "anthropic":
        return [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c", "content": payload}],
            },
        ]
    if protocol == "gemini":
        return [
            {"role": "model", "parts": [{"function_call": {"id": "c", "name": "f", "args": {}}}]},
            {
                "role": "user",
                "parts": [{"function_response": {"id": "c", "name": "f", "response": payload}}],
            },
        ]
    return [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "c", "tool_name": "f", "content": payload},
    ]


def native_result_text(protocol, request):
    if protocol in ("openai_chat", "ollama"):
        return request["messages"][-1]["content"]
    if protocol == "openai_responses":
        content = request["input"][-1]["output"]
    elif protocol == "anthropic":
        content = request["messages"][-1]["content"][0]["content"]
    else:
        content = request["contents"][-1]["parts"][0]["function_response"]["response"]
        return content["result"] if set(content) == {"result"} else json.dumps(content)
    return content if isinstance(content, str) else "".join(part["text"] for part in content)
