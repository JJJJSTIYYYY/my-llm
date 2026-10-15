"""Offline demo: both API types, streaming, and caller-executed function tools.

Run with `uv run python examples/local_demo.py`. The actual OpenAI SDK uses a
local httpx.MockTransport, so this example needs no key or network service.
"""

import asyncio
import json

import httpx
from openai import AsyncOpenAI

from my_llm import MessageChunk, ModelProfile, ProviderProfile, create


def chat(content=None, finish=None, calls=None):
    return {
        "id": "chat_demo",
        "created": 1,
        "model": "demo",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "tool_calls": calls},
                "finish_reason": finish,
            }
        ],
    }


def handler(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/responses"):
        events = [
            {
                "type": "response.output_text.delta",
                "delta": "Hello from Responses",
                "sequence_number": 1,
            },
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {"id": "resp_demo", "status": "completed", "output": []},
            },
        ]
    elif body.get("tools") and not any(message["role"] == "tool" for message in body["messages"]):
        events = [
            chat(
                calls=[
                    {
                        "index": 0,
                        "id": "call_add",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                    }
                ]
            ),
            chat(finish="tool_calls"),
        ]
    else:
        events = [
            chat(content="5" if body.get("tools") else "Hello from Chat Completions"),
            chat(finish="stop"),
        ]
    sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})


async def main():
    async with AsyncOpenAI(
        api_key="local-demo",
        base_url="https://local.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as sdk:
        for api_type in ("chat_completions", "responses"):
            stream = await create(
                provider=ProviderProfile(api_type=api_type),
                model="demo",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                client=sdk,
            )
            combined = MessageChunk()
            async with stream:
                async for chunk in stream:
                    combined += chunk
            print(api_type, combined.to_message(require_finished=True)["content"])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        model = ModelProfile(model="demo", supports_function_calling=True)
        history = [{"role": "user", "content": "What is 2 + 3?"}]
        response = await create(
            provider=ProviderProfile(), model=model, messages=history, tools=tools, client=sdk
        )
        history.append(response)
        for call in response["tool_calls"]:
            if call["function"]["name"] != "add":
                raise ValueError("Unknown local tool")
            arguments = json.loads(call["function"]["arguments"])
            output = arguments["a"] + arguments["b"]
            history.append({"role": "tool", "tool_call_id": call["id"], "content": str(output)})
        final = await create(
            provider=ProviderProfile(), model=model, messages=history, tools=tools, client=sdk
        )
        print("Local tool result:", final["content"])


if __name__ == "__main__":
    asyncio.run(main())
