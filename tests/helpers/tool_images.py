"""Independent native image fixtures and opt-in preset request helpers."""

from dataclasses import replace

import httpx
import pytest

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
from my_llm.codec.gemini.prebuilt import move_tool_images_to_user_message as gemini_hook
from my_llm.codec.ollama.prebuilt import move_tool_images_to_user_message as ollama_hook
from my_llm.codec.ollama.prebuilt import tool_images as ollama_images
from my_llm.codec.openai.chat.prebuilt import move_tool_images_to_user_message as chat_hook
from tests.performance.workloads import profiles

HOOKS = {"openai_chat": chat_hook, "gemini": gemini_hook, "ollama": ollama_hook}
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


class Unreadable:
    def __deepcopy__(self, memo):
        raise AssertionError("an unsupported native field was traversed")


def gemini_image_history(media, sdk):
    image = (
        {"inline_data": {"mime_type": "image/png", "data": b"png"}}
        if media == "inline"
        else {"file_data": {"mime_type": "image/png", "file_uri": "https://example.test/p.png"}}
    )
    parts = [
        {"inline_data": {"mime_type": "audio/wav", "data": Unreadable()}},
        image,
        {"file_data": {"mime_type": "application/pdf", "file_uri": Unreadable()}},
    ]
    messages = [
        {"role": "model", "parts": [{"function_call": {"id": "c", "name": "f", "args": {}}}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "c",
                        "name": "f",
                        "response": {"result": "ok"},
                        "parts": parts,
                    }
                }
            ],
        },
    ]
    if sdk:
        from google.genai import types

        native_parts = []
        for part in parts:
            field = next(iter(part))
            model = (
                types.FunctionResponseBlob
                if field == "inline_data"
                else types.FunctionResponseFileData
            )
            native_parts.append(
                types.FunctionResponsePart.model_construct(
                    **{field: model.model_construct(**part[field])}
                )
            )
        messages = [
            types.Content(**messages[0]),
            types.Content.model_construct(
                role="user",
                parts=[
                    types.Part.model_construct(
                        function_response=types.FunctionResponse.model_construct(
                            id="c", name="f", response={"result": "ok"}, parts=native_parts
                        )
                    )
                ],
            ),
        ]
        image = native_parts[1]
    return messages, image


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
        },
        "gemini": {
            "role": "user",
            "parts": [
                {"file_data": {"file_uri": URL}},
                {"inline_data": {"mime_type": "image/png", "data": DATA}},
            ],
        },
        "ollama": {"role": "user", "content": "[Unsupported Image]", "images": [DATA]},
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


@pytest.fixture
def unavailable_tool_images(monkeypatch):
    """Keep legacy relocation tests offline while covering download failure fallback."""

    def client(**options):
        return httpx.AsyncClient(
            **options,
            transport=httpx.MockTransport(lambda request: httpx.Response(404)),
            trust_env=False,
        )

    monkeypatch.setattr(ollama_images, "AsyncClient", client)


async def consume(stream):
    async with stream:
        async for _ in stream:
            pass
    return stream.result


def request_history(protocol, request):
    return request["contents" if protocol == "gemini" else "messages"]
