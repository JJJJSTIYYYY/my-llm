"""Compile documented snippets and execute examples against the official SDK."""

import ast
import json
import re
from dataclasses import fields
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile
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


@pytest.mark.parametrize("stdio", ["openai_chat", "openai_responses", "anthropic", "middleware"])
@pytest.mark.parametrize("case", ["text_collected", "text_stream", "tool_roundtrip"])
async def test_sensenova_example_scenarios_with_local_sdk(monkeypatch, stdio, case):
    import runpy

    example = runpy.run_path(str(ROOT / "examples/sensenova_e2e.py"))
    clients, requests = [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        has_result = any(message["role"] == "tool" for message in body["messages"])
        if body.get("tools") and not has_result:
            events = [
                chat_event(
                    calls=[
                        {
                            "index": 0,
                            "id": "call_add",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":19,"b":23}'},
                        }
                    ]
                ),
                chat_event(finish="tool_calls"),
            ]
        else:
            events = [
                chat_event(content="42" if has_result else "SENSENOVA_OK"),
                chat_event(finish="stop"),
            ]
        return httpx.Response(200, text=sse(events), headers={"Content-Type": "text/event-stream"})

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    async with MyLLM(
        provider=example["provider"]("openai_chat", "local-example-key"),
        model=ModelProfile(
            model=Attribute(default_value="local-example"),
            supports_function_calling=Config(default_value=True),
        ),
        stdio_protocol=stdio,
    ) as llm:
        result = await example["scenario"](llm, stdio, case)
    assert result["requests"] == len(requests) == (2 if case == "tool_roundtrip" else 1)
    assert result["text"] == ("42" if case == "tool_roundtrip" else "SENSENOVA_OK")
    assert len(clients) == 1 and clients[0].is_closed()


@pytest.mark.parametrize("succeeds", [False, True])
async def test_sensenova_retry_preserves_passed_rows_and_failure_history(
    monkeypatch, tmp_path, succeeds
):
    import runpy
    from types import SimpleNamespace

    example = runpy.run_path(str(ROOT / "examples/sensenova_e2e.py"))
    completed = {
        "upstream": "openai_chat",
        "stdio": "middleware",
        "case": "text_collected",
        "status": "passed",
    }
    failed = {
        **completed,
        "case": "text_stream",
        "status": "failed",
        "error_type": "RateLimitError",
        "error": "rpm exhausted",
    }
    report = tmp_path / "results.json"
    report.write_text(json.dumps({"model": "local-example", "results": [completed, failed]}))
    calls = []

    async def scenario(llm, stdio, case, *, request_interval):
        calls.append(case)
        assert stdio == llm.stdio_protocol == "middleware"
        assert request_interval == 0
        assert llm.codec.semantic_loss_adaptor.semantic_loss_protocol == "ignore"
        if not succeeds:
            raise RuntimeError("failure with local-example-key")
        return {"text": "SENSENOVA_OK", "events": 2, "requests": 1}

    monkeypatch.setitem(example["run"].__globals__, "scenario", scenario)
    has_failure = await example["run"](
        SimpleNamespace(
            model="local-example",
            upstream=["openai_chat"],
            stdio=["middleware"],
            report=report,
            retry_failed=True,
            request_interval=0,
            semantic_loss_protocol="ignore",
        ),
        "local-example-key",
    )
    assert has_failure is not succeeds
    results = json.loads(report.read_text())["results"]
    assert calls == ["text_stream"]
    assert results[0] == completed
    assert results[1]["status"] == ("passed" if succeeds else "failed")
    assert results[1]["previous_attempts"] == [failed]
    assert "local-example-key" not in report.read_text()


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
        protocol=Config(default_value="openai_chat"), api_key=parameter("api_key", "local-doc-test")
    )
    llm = MyLLM(
        stdio_protocol="openai_chat",
        provider=provider,
        model=ModelProfile(model=Attribute(default_value="demo")),
    )
    for name in ("ask", "print_stream"):
        assert (await api_example(name)(llm, "Hello"))["content"] == "Hello"
    assert len(clients) == 1 and not clients[0].is_closed()
    await llm.aclose()
    assert (await api_example("add_round_trip")(provider, "demo"))["content"] == "5"
    assert requests[-1]["messages"][-1]["role"] == "tool"
    assert len(clients) == 2
    assert all(c.is_closed() for c in clients)
