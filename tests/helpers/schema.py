"""Validate native wire values against the pinned latest official source snapshot."""

import json
from copy import deepcopy
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


def output_event_for_schema(protocol: str, value: dict) -> dict:
    """Add SDK-required usage placeholders only to an output validation copy.

    MyLLM keeps real usage in its private call record. The pinned native schemas
    still require counters on some event envelopes; upstream validation stays
    strict and never uses this output-only adapter.
    """

    def check(item):
        if isinstance(item, dict):
            assert not (
                {"usage", "usage_metadata", "usageMetadata", "prompt_eval_count", "eval_count"}
                & item.keys()
            )
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)

    check(value)
    projected = deepcopy(value)
    if protocol == "anthropic":
        if projected.get("type") == "message_start":
            projected["message"]["usage"] = {"input_tokens": 0, "output_tokens": 0}
        elif projected.get("type") == "message_delta":
            projected["usage"] = {"output_tokens": 0}
    elif protocol == "openai_responses" and isinstance(projected.get("response"), dict):
        projected["response"]["usage"] = None
    return projected


def validate_output_event(protocol: str, value: dict) -> None:
    validate_event(protocol, output_event_for_schema(protocol, value))


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
