"""Independent native image fixtures and opt-in preset request helpers."""

import re
from dataclasses import replace

from my_llm import (
    AssistantMessage,
    ImageContent,
    ImageSource,
    MyLLM,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
    omit,
)
from my_llm.codec.openai.chat.prebuilt import chat_semantic_loss_hook as chat_hook
from tests.performance.workloads import profiles

HOOKS = {"openai_chat": chat_hook}
URL = "https://example.com/a.png"
DATA = "aW1hZ2U="
IMAGE_REFERENCE = re.compile(r"\[Image: image_[0-9a-f]{64}\]")
REFERENCES = [
    "[Image: image_c4ceecc2a6a51c495dd6f21eda45173e30be2115db57ef7ab67c75329200f1d1]",
    "[Image: image_a9c7e8d9437708942fac41a6a556f51261afc865fa12f8a6bf4fe30eb1f3e69a]",
]


def image_references(message):
    return IMAGE_REFERENCE.findall(str(message["content"]))


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
        ToolMessage(
            "tool",
            [
                TextContent("before"),
                ImageContent(ImageSource(type="url", url=URL)),
                ImageContent(ImageSource(type="base64", data=DATA, media_type="image/png")),
                TextContent("after"),
            ],
            call_id="c",
        ),
        UserMessage("user", [TextContent("tail")]),
    ]


def history(source):
    return {
        "anthropic": picture_history,
        "openai_responses": responses_history,
        "middleware": middleware_history,
    }[source]()


def expected_image_message(target, references):
    return {
        "openai_chat": {
            "role": "user",
            "content": [
                {"type": "text", "text": references[0] + "\n"},
                {"type": "image_url", "image_url": {"url": URL}},
                {"type": "text", "text": references[1] + "\n"},
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
        await llm._codec._prepare(
            {"messages": messages, "tools": omit, "stream": False, **options},
            llm.resolved_profile,
            input_format=llm.stdio_protocol,
            client=llm._client_parameters,
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
