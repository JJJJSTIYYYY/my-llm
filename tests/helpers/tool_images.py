"""Independent native image fixtures and opt-in preset request helpers."""

from dataclasses import replace

from my_llm import (
    AssistantMessage,
    ImageContent,
    ImageSource,
    MyLLM,
    TextContent,
    ToolCall,
    ToolResult,
    UserMessage,
    omit,
)
from my_llm.codec.openai.chat.prebuilt import move_tool_images_to_user_message as chat_hook
from tests.performance.workloads import profiles

HOOKS = {"openai_chat": chat_hook}
URL = "https://example.com/a.png"
DATA = "aW1hZ2U="


def picture_history():
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_a", "name": "look", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_a",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aW1hZ2U=",
                            },
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        },
        {"role": "user", "content": "complete tail"},
    ]


def responses_history():
    return [
        {"role": "assistant", "content": "reading images"},
        {"type": "function_call", "call_id": "c", "name": "look", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c",
            "output": [
                {"type": "input_text", "text": "before"},
                {"type": "input_image", "image_url": URL},
                {"type": "input_image", "image_url": f"data:image/png;base64,{DATA}"},
                {"type": "input_text", "text": "after"},
            ],
        },
        {"role": "user", "content": "tail"},
    ]


def middleware_history():
    return [
        AssistantMessage("assistant", [ToolCall("c", "look", {})]),
        UserMessage(
            "user",
            [
                ToolResult(
                    "c",
                    [
                        TextContent("before"),
                        ImageContent(ImageSource(type="url", url=URL)),
                        ImageContent(ImageSource(type="base64", data=DATA, media_type="image/png")),
                        TextContent("after"),
                    ],
                )
            ],
        ),
        UserMessage("user", [TextContent("tail")]),
    ]


def history(source):
    return {
        "anthropic": picture_history,
        "openai_responses": responses_history,
        "middleware": middleware_history,
    }[source]()


def expected_image_message(target):
    return {
        "openai_chat": {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": URL}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{DATA}"}},
            ],
        }
    }[target]


def configured_llm(target, source="anthropic", cached=False, **options):
    provider, model = profiles(target)
    llm = MyLLM(
        provider=provider,
        model=replace(model, **options),
        stdio_protocol=source,
        semantic_loss_protocol="hook",
        input_semantic_loss_hook=HOOKS[target],
        enable_message_cache=cached,
    )
    return llm


async def prepare(llm, messages, **options):
    return (
        await llm.codec.prepare(
            {"messages": messages, "tools": omit, "stream": False, **options},
            llm.resolved_profile,
            input_format=llm.stdio_protocol,
            client=llm.client_parameters,
            enable_message_cache=llm.enable_message_cache,
            message_cache_key=llm._message_cache_key,
        )
    )["request"]


async def consume(stream):
    async with stream:
        async for _ in stream:
            pass
    return stream.result


def request_history(request):
    return request["messages"]
