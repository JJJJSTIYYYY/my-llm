"""Complete JSON validation diagnostics, feedback and retry accounting."""

import json
from dataclasses import asdict

import pytest
from jsonschema import Draft202012Validator

from my_llm import Attribute, Config, JSONGenerationError, ModelProfile, MyLLM, ProviderProfile
from my_llm.basic.exceptions import ConfigurationError
from my_llm.utils.json_schema import json_schema_errors
from tests.helpers.protocols import chat_event
from tests.helpers.transport import FakeClient

SCHEMA = {
    "type": "object",
    "required": ["name", "age", "tags", "profile"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "profile": {
            "type": "object",
            "required": ["enabled"],
            "additionalProperties": False,
            "properties": {"enabled": {"type": "boolean"}},
        },
    },
}
BAD = {"name": 2, "age": -1, "tags": ["bad", -2], "profile": {"extra": 1}, "extra": "x"}
GOOD = {"name": "test", "age": 1, "tags": [1, 2], "profile": {"enabled": False}}
EXPECTED = {
    ("/name", "type"),
    ("/age", "minimum"),
    ("/tags/0", "type"),
    ("/tags/1", "minimum"),
    ("/profile", "required"),
    ("/profile", "additionalProperties"),
    ("", "additionalProperties"),
}


def setup(monkeypatch, outputs, *, tools=False, cache=False):
    clients = []
    for output in outputs:
        text = json.dumps(output)
        delta = (
            chat_event(
                calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "structured_output", "arguments": text},
                    }
                ]
            )
            if tools
            else chat_event(content=text)
        )
        clients.append(
            FakeClient(
                [
                    delta,
                    chat_event(finish="tool_calls" if tools else "stop"),
                    {
                        **chat_event(
                            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
                        ),
                        "choices": [],
                    },
                ]
            )
        )
    calls = []

    def factory(**parameters):
        index = len(calls)
        if index:
            assert clients[index - 1].closed and clients[index - 1].upstream.closed
        calls.append(parameters)
        return clients[index]

    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", factory)
    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_chat")),
        model=ModelProfile(
            model=Attribute(default_value="demo"),
            supports_function_calling=Config(default_value=tools),
        ),
        stdio_protocol="openai_chat",
        enable_message_cache=cache,
    )
    return llm, clients, calls


@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("cache", [False, True])
async def test_all_schema_errors_are_recorded_and_sent_back(monkeypatch, tools, cache):
    llm, clients, _ = setup(monkeypatch, [BAD, BAD, GOOD], tools=tools, cache=cache)
    messages = [{"role": "user", "content": "Return a profile"}]
    assert await llm.generate_json(messages=messages, schema=SCHEMA, max_retries=2) == GOOD
    failures = llm.get_last_json_failures()
    assert [f["attempt"] for f in failures] == [1, 2]
    for failure in failures:
        assert failure["stage"] == "schema"
        assert {(e["instance_path"], e["validator"]) for e in failure["errors"]} == EXPECTED
        assert all(e["schema_path"] and e["message"] for e in failure["errors"])
    for index in (1, 2):
        history = clients[index].requests[0]["messages"]
        assert history[0] == messages[0]
        feedback = history[-1]
        assert feedback["role"] == "system"
        diagnostics = json.loads(feedback["content"].split("\n", 1)[1])
        assert diagnostics == failures[:index]
        assert sum(m["role"] == "user" for m in history) == 1
    assert asdict(llm.get_last_usage()) == {
        "input": 9,
        "output": 6,
        "total": 15,
        "cached": None,
        "reasoning": None,
    }
    failures[0]["errors"].clear()
    assert len(llm.get_last_json_failures()[0]["errors"]) == 7


async def test_exhaustion_exception_contains_all_attempts_and_errors(monkeypatch):
    llm, _, calls = setup(monkeypatch, [BAD, BAD])
    with pytest.raises(JSONGenerationError) as caught:
        await llm.generate_json(
            messages=[{"role": "user", "content": "hi"}], schema=SCHEMA, max_retries=1
        )
    assert len(calls) == 2
    assert caught.value.failures == llm.get_last_json_failures()
    assert [len(f["errors"]) for f in caught.value.failures] == [7, 7]
    caught.value.failures.clear()
    assert len(llm.get_last_json_failures()) == 2
    assert llm.get_last_usage().total == 10


def test_combinator_branches_and_escaped_paths_are_complete():
    schema = {
        "type": "object",
        "properties": {
            "a/b~c": {
                "anyOf": [{"type": "integer", "minimum": 3}, {"type": "string", "minLength": 4}]
            },
            "other": {"type": "boolean"},
        },
    }
    errors = json_schema_errors(Draft202012Validator(schema), {"a/b~c": [], "other": 1})
    assert len(errors) == 4
    assert {e["validator"] for e in errors} == {"anyOf", "type"}
    assert sum(e["instance_path"] == "/a~1b~0c" for e in errors) == 3
    assert any(e["schema_path"].endswith("/anyOf/0/type") for e in errors)
    assert any(e["schema_path"].endswith("/anyOf/1/type") for e in errors)


async def test_invalid_schema_does_not_consume_requests(monkeypatch):
    llm, _, calls = setup(monkeypatch, [])
    invalid = {
        "type": "object",
        "properties": {"a": {"type": 42}, "b": {"minLength": -1}, "c": {"pattern": "["}},
    }
    with pytest.raises(ConfigurationError) as caught:
        await llm.generate_json(
            messages=[{"role": "user", "content": "hi"}], schema=invalid, max_retries=2
        )
    assert all(
        path in str(caught.value) for path in ("/properties/a", "/properties/b", "/properties/c")
    )
    assert not calls


@pytest.mark.parametrize("output", [{"age": "1"}, {"age": True}, {"age": None}, {}])
async def test_types_are_not_coerced_and_success_clears_history(monkeypatch, output):
    llm, _, _ = setup(monkeypatch, [output, {"age": 1}])
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
    with pytest.raises(JSONGenerationError):
        await llm.generate_json(messages=[{"role": "user", "content": "hi"}], schema=schema)
    assert len(llm.get_last_json_failures()) == 1
    assert await llm.generate_json(messages=[{"role": "user", "content": "hi"}], schema=schema) == {
        "age": 1
    }
    assert llm.get_last_json_failures() == []


async def test_multiple_tool_calls_report_all_candidate_schema_errors(monkeypatch):
    from copy import deepcopy

    llm, clients, _ = setup(monkeypatch, [BAD], tools=True)
    calls = clients[0].events[0]["choices"][0]["delta"]["tool_calls"]
    extra = deepcopy(calls[0])
    extra.update(index=1, id="call_2")
    calls.append(extra)
    with pytest.raises(JSONGenerationError) as caught:
        await llm.generate_json(messages=[{"role": "user", "content": "hi"}], schema=SCHEMA)
    failure = caught.value.failures[0]
    assert failure["stage"] == "tool_call" and len(failure["errors"]) == 15
    for index in (0, 1):
        assert {
            (e["instance_path"], e["validator"])
            for e in failure["errors"]
            if e.get("tool_call_index") == index
        } == EXPECTED
