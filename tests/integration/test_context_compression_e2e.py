"""External-input/privacy boundaries and real SDK serialization without live billing."""

import json
import runpy
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from tests.helpers.protocols import chat_event
from tests.helpers.transport import sse

SCRIPT = Path(__file__).resolve().parents[2] / "examples/context_compression_e2e.py"


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return runpy.run_path(str(SCRIPT))


def test_external_context_is_redacted_without_changing_source(tmp_path, example):
    path = tmp_path / "context.json"
    secret = "sk-" + "x" * 30
    source = json.dumps({"messages": [{"role": "user", "content": secret}]})
    path.write_text(source)
    assert example["load_messages"](path, secret)[0]["content"] == "[REDACTED_API_KEY]"
    assert path.read_text() == source


def test_checkout_context_and_report_paths_are_rejected(example):
    with pytest.raises(ValueError, match="outside"):
        example["external_path"](SCRIPT.parent / "private-context.json")


def test_private_report_rejects_credentials_and_uses_private_permissions(tmp_path, example):
    path = tmp_path / "report.json"
    secret = "sk-" + "y" * 30
    with pytest.raises(ValueError, match="credential"):
        example["save_private"](path, {"summary": secret}, secret)
    assert not path.exists()
    example["save_private"](path, {"ok": True}, secret)
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("scenario", ["compression", "compact", "handoff", "indexed_handoff"])
async def test_conversation_and_new_user_request_reach_official_sdk(monkeypatch, example, scenario):
    messages = [{"role": "user", "content": "Synthetic context for an offline transport test."}]
    payload = {
        "summary": "Synthetic summary",
        **{
            name: []
            for name in (
                "user_goals",
                "decisions",
                "implemented",
                "pending",
                "constraints",
                "caveats",
            )
        },
    }
    args = json.dumps(payload)
    events = [
        chat_event(
            calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "structured_output", "arguments": args},
                }
            ]
        ),
        chat_event(finish="tool_calls"),
        {
            **chat_event(usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            "choices": [],
        },
    ]
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, text=sse(events), headers={"Content-Type": "text/event-stream"})

    def factory(**parameters):
        http_client = parameters.pop("http_client")
        http_client._transport = httpx.MockTransport(handler)
        http_client._mounts.clear()
        return AsyncOpenAI(**parameters, http_client=http_client)

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    row = await example["run_case"](
        messages, "local-test-key", "openai_chat", scenario, "demo", 1024
    )
    assert captured[0]["messages"][0] == messages[0]
    request = captured[0]["messages"][-2]
    assert request["role"] == "user"
    assert request["content"].startswith(example["SCENARIOS"][scenario])
    if scenario == "indexed_handoff":
        assert "1 / user / Synthetic context" in request["content"]
    else:
        assert request["content"] == example["SCENARIOS"][scenario]
    assert captured[0]["messages"][-1]["role"] == "system"
    assert row["snapshot"] == payload and row["clients_closed"]
    assert row["evidence"]["requests"] == 1
    assert row["schema_valid"]
    assert row["usage"]["input"] == 3 and row["usage"]["output"] == 2 and row["usage"]["total"] == 5
    assert messages == [
        {"role": "user", "content": "Synthetic context for an offline transport test."}
    ]
