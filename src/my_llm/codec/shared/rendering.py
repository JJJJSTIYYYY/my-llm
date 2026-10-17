"""Projection mechanics for rendering known semantics without foreign extensions."""

import json
from copy import deepcopy

from my_llm.codec.shared.conversion import blocks
from my_llm.codec.shared.keys import SOURCE
from my_llm.exceptions import UnsupportedCapabilityError


def select(value: dict, *fields: str) -> dict:
    return {key: deepcopy(value[key]) for key in fields if key in value}


def select_present(value: dict, *fields: str) -> dict:
    return {key: child for key, child in select(value, *fields).items() if child is not None}


def project_message(message: dict, project, *, fields: tuple[str, ...] = ("role",)) -> dict:
    content = []
    for block in blocks(message.get("content")):
        rendered = project(block)
        if rendered is not None:
            content.append(rendered)
    return {**select(message, *fields, SOURCE), "content": content}


def unsupported(block: dict):
    raise UnsupportedCapabilityError(
        f"target protocol cannot represent content type {block.get('type')!r}"
    )


def project_block(
    block: dict, fields: dict, *, text_fields: dict | None = None, omitted: tuple = ()
):
    """Apply the target's declared field projection and visible-text fallbacks."""
    kind = block.get("type")
    if kind in omitted:
        return None
    if kind in (text_fields or {}):
        return {"type": "text", "text": block.get(text_fields[kind], "")}
    if kind not in fields:
        return unsupported(block)
    return select(block, *fields[kind])


def text_result(value) -> str:
    """Represent canonical text blocks and structured results as a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(block, dict) and block.get("type") == "text" for block in value
    ):
        return "".join(block.get("text", "") for block in value)
    return json.dumps(value, ensure_ascii=False)
