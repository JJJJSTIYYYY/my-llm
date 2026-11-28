"""Assert structured-generation response shapes and read their object payloads."""

import json

from my_llm import (
    AssistantMessage,
    Attribute,
    Config,
    ModelProfile,
    MyCodec,
    MyLLM,
    ProviderProfile,
    TextContent,
)
from tests.helpers.protocols import stream_events
from tests.helpers.transport import FakeClient


def assistant_content(result, protocol):
    """Read completed assistant content through the public protocol boundary."""
    if protocol == "middleware":
        assert isinstance(result, AssistantMessage)
        return result.content
    messages = result if protocol == "openai_responses" else [result]
    encoded = MyCodec().encode(messages, input_format=protocol)
    assert all(isinstance(message, AssistantMessage) for message in encoded.messages)
    return [part for message in encoded.messages for part in message.content]


def structured_object(result, protocol):
    """Read a JSON object from assistant text in any stdio protocol."""
    content = assistant_content(result, protocol)
    return json.loads("".join(part.text for part in content if isinstance(part, TextContent)))


def make_llm(protocol, stdio, *, tools=True, cache=False, sdk_retries=7):
    return MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value=protocol), max_retries=Config(default_value=sdk_retries)
        ),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=tools),
        ),
        stdio_protocol=stdio,
        enable_message_cache=cache,
    )


def response_events(
    protocol,
    *,
    tools,
    text='Here is JSON: ```json\n{"city":"上海"}\n```',
    tool_name="structured_output",
):
    # Keep native stream identity and fragmented arguments from the shared fixtures.
    return json.loads(
        json.dumps(stream_events(protocol, tool=tools))
        .replace("weather", tool_name)
        .replace("Hello", text.replace('"', '\\"').replace("\n", "\\n"))
    )


def install_sdk(
    monkeypatch,
    protocol,
    *,
    tools,
    text='Here is JSON: ```json\n{"city":"上海"}\n```',
    tool_name="structured_output",
):
    events = response_events(protocol, tools=tools, text=text, tool_name=tool_name)
    clients = []

    def factory(**kwargs):
        client = FakeClient(events)
        clients.append(client)
        return client

    module = {
        "openai_chat": "openai.chat.transport.AsyncOpenAI",
        "openai_responses": "openai.responses.transport.AsyncOpenAI",
        "anthropic": "anthropic.transport.AsyncAnthropic",
    }[protocol]
    monkeypatch.setattr(f"my_llm.codec.{module}", factory)
    return clients
