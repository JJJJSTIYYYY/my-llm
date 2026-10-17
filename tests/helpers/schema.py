"""Validate native wire values against the pinned latest official source snapshot."""

import json
from functools import cache
from pathlib import Path

from google.genai import types
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2] / "docs/reference/schema"


@cache
def schema(protocol: str, kind: str) -> dict:
    folder = "ollama_chat" if protocol == "ollama" else protocol
    return json.loads((ROOT / folder / f"{kind}.schema.json").read_text())


@cache
def validator(protocol: str, kind: str):
    return Draft202012Validator(schema(protocol, kind))


def validate_request(protocol: str, value: dict) -> None:
    validator(protocol, "request").validate(value)


def validate_event(protocol: str, value: dict) -> None:
    if protocol == "gemini":
        value = types.GenerateContentResponse.model_validate(value).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
    validator(protocol, "stream-event").validate(value)


@cache
def response_fields() -> set[str]:
    root = schema("openai_responses", "response")

    def properties(value):
        if "$ref" in value:
            return properties(root["$defs"][value["$ref"].split("/")[-1]])
        result = set(value.get("properties", {}))
        for child in value.get("allOf", []):
            result.update(properties(child))
        return result

    return properties(root)
