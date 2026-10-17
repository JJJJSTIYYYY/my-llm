"""Validate the reference artifacts offline, independently of my_llm codecs.

Run with: uv run --no-project --with jsonschema==4.26.0 python <this file>
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def references(value):
    if isinstance(value, dict):
        if "$ref" in value:
            yield value["$ref"]
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def validate(schema: dict, value) -> None:
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)


def frames(path: Path) -> tuple[list[dict], bool]:
    if path.suffix == ".ndjson":
        return [json.loads(line) for line in path.read_text().splitlines() if line], False
    events, done = [], False
    for frame in path.read_text().strip().split("\n\n"):
        data = "\n".join(line[6:] for line in frame.splitlines() if line.startswith("data: "))
        tags = [line[7:] for line in frame.splitlines() if line.startswith("event: ")]
        if data == "[DONE]":
            assert not done
            done = True
        else:
            assert not done, "content after [DONE]"
            value = json.loads(data)
            if tags:
                assert tags == [value["type"]], "SSE event name and JSON type differ"
            events.append(value)
    return events, done


def lifecycle(protocol: str, events: list[dict], done: bool, error: bool) -> None:
    if error:
        assert not done
        return
    if protocol == "openai_chat":
        assert done
        assert len({event["id"] for event in events}) == 1
        assert events[-1]["choices"] == [] and events[-1]["usage"] is not None
        assert events[-2]["choices"][0]["finish_reason"] is not None
    elif protocol == "openai_responses":
        assert not done
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] in (
            "response.completed",
            "response.incomplete",
            "response.failed",
        )
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        opened, closed = {}, {}
        for event in events:
            index = event.get("output_index")
            if event["type"] == "response.output_item.added":
                assert index not in opened
                opened[index] = event["item"]["id"]
            elif index is not None:
                assert index in opened and index not in closed
                if "item_id" in event:
                    assert event["item_id"] == opened[index]
                if event["type"] == "response.output_item.done":
                    assert event["item"]["id"] == opened[index]
                    closed[index] = event["item"]
        assert opened.keys() == closed.keys()
        assert [closed[index] for index in sorted(closed)] == events[-1]["response"]["output"]
    elif protocol == "anthropic":
        assert not done
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        opened, closed = set(), set()
        for event in events:
            if event["type"] == "content_block_start":
                assert event["index"] not in opened
                opened.add(event["index"])
            elif event["type"] in ("content_block_delta", "content_block_stop"):
                assert event["index"] in opened - closed
                if event["type"] == "content_block_stop":
                    closed.add(event["index"])
        assert opened == closed
    elif protocol == "gemini":
        assert not done
        assert events[-1]["candidates"][0]["finishReason"] != "FINISH_REASON_UNSPECIFIED"
    elif protocol == "ollama_chat":
        assert not done
        assert events[-1]["done"] is True
        assert all(event["done"] is False for event in events[:-1])


def main() -> None:
    counts = {"schemas": 0, "templates": 0, "stream_files": 0, "stream_frames": 0}
    for path in ROOT.rglob("*.json"):
        read(path)
    for path in ROOT.rglob("*.schema.json"):
        schema = read(path)
        Draft202012Validator.check_schema(schema)
        for ref in references(schema):
            assert ref.startswith("#/"), (path, ref)
            value = schema
            for key in ref[2:].split("/"):
                value = value[key.replace("~1", "/").replace("~0", "~")]
        counts["schemas"] += 1
    for protocol in read(ROOT / "catalog.json")["protocols"]:
        folder = ROOT / protocol
        for name in ("request", "response", "error"):
            validate(read(folder / f"{name}.schema.json"), read(folder / f"{name}.json"))
        for case in read(folder / "templates.json")["cases"]:
            validate(read(folder / case["schema"]), case["value"])
            counts["templates"] += 1
        for path in sorted([*folder.glob("*.sse"), *folder.glob("*.ndjson")]):
            values, done = frames(path)
            error = ".error." in path.name
            schema_name = (
                "error.schema.json"
                if error and protocol in ("openai_chat", "ollama_chat")
                else "stream-event.schema.json"
            )
            for value in values:
                validate(read(folder / schema_name), value)
            lifecycle(protocol, values, done, error)
            counts["stream_files"] += 1
            counts["stream_frames"] += len(values)
    for path in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).is_file(), (path, target)
    recorded = read(ROOT / "verification.json")
    assert counts == recorded["counts"], (counts, recorded["counts"])
    for name, expected in recorded["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print(json.dumps({"status": "passed", **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
