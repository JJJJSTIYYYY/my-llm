"""Compile documented snippets and execute examples against the official SDK."""

import ast
import json
import re
from dataclasses import fields
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from my_llm import ModelProfile, MyLLM, ProviderProfile
from tests.helpers.profiles import parameter
from tests.helpers.protocols import chat_event
from tests.helpers.transport import sse

ROOT = Path(__file__).resolve().parents[2]


def python_blocks(path):
    return re.findall(r"^```python\n(.*?)^```", path.read_text(), re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "docs/reference/api.md",
        "docs/reference/configuration.md",
        "docs/guides/adding-protocol.md",
    ],
)
def test_documented_python_is_valid(relative_path):
    blocks = python_blocks(ROOT / relative_path)
    assert blocks
    for index, source in enumerate(blocks):
        compile(source, f"{relative_path}:block{index}", "exec")


def test_configuration_guide_covers_profile_fields_and_local_links():
    config = (ROOT / "docs/reference/configuration.md").read_text()
    for profile in (ProviderProfile, ModelProfile):
        for field in fields(profile):
            assert f"`{field.name}`" in config, field.name
    for path in (ROOT / "README.md", *(ROOT / "docs").rglob("*.md")):
        prose = re.sub(r"^```.*?^```", "", path.read_text(), flags=re.MULTILINE | re.DOTALL)
        for target in re.findall(r"\]\(([^)]+)\)", prose):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).is_file(), target


def test_offline_request_pipeline_example():
    import runpy

    namespace = runpy.run_path(str(ROOT / "examples/request_pipeline.py"))
    prepared = namespace["main"]()
    assert prepared["request"]["model"] == "example-model"


def api_example(name):
    for source in python_blocks(ROOT / "docs/reference/api.md"):
        tree = ast.parse(source)
        if any(isinstance(node, ast.AsyncFunctionDef) and node.name == name for node in tree.body):
            namespace = {"__name__": "documentation_example"}
            exec(compile(tree, "api.md", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"Missing API example: {name}")


async def test_guide_text_stream_and_tool_round_trip(monkeypatch):
    clients, requests = [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        history = body["messages"]
        has_result = any(m["role"] == "tool" for m in history)
        if body.get("tools") and not has_result:
            events = [
                chat_event(
                    calls=[
                        {
                            "index": 0,
                            "id": "call_add",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                        }
                    ]
                ),
                chat_event(finish="tool_calls"),
            ]
        else:
            events = [chat_event(content="5" if has_result else "Hello"), chat_event(finish="stop")]
        return httpx.Response(200, text=sse(events), headers={"Content-Type": "text/event-stream"})

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    monkeypatch.setattr("my_llm.codec.openai.responses.transport.AsyncOpenAI", factory)
    provider = ProviderProfile(
        protocol="openai_chat", api_key=parameter("api_key", "local-doc-test")
    )
    llm = MyLLM(stdio_protocol="openai_chat", provider=provider, model=ModelProfile(model="demo"))
    for name in ("ask", "print_stream"):
        assert (await api_example(name)(llm, "Hello"))["content"] == "Hello"
    assert len(clients) == 1 and not clients[0].is_closed()
    await llm.aclose()
    assert (await api_example("add_round_trip")(provider, "demo"))["content"] == "5"
    assert requests[-1]["messages"][-1]["role"] == "tool"
    assert len(clients) == 2
    assert all(c.is_closed() for c in clients)
