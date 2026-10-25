"""Request-owned accumulation of explicit, provider-independent stream operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from my_llm.basic.data import capture
from my_llm.basic.exceptions import IncompleteStreamError
from my_llm.basic.semantic_loss import Originated
from my_llm.codec.shared.conversion import arguments, source_protocol
from my_llm.codec.shared.stream import operations


def _located(value: dict, source: dict) -> dict:
    if isinstance(source, Originated):
        return Originated(value, source.origin_path, source.origin_value)
    return value


@dataclass(slots=True)
class BlockState:
    """Recognized content and lifecycle for one block."""

    index: int
    content: dict
    source_ref: tuple[str, int, int] | None = None
    fragments: dict[str, list[str]] = field(default_factory=dict)
    argument_parts: list[str] | None = None
    emitted: bool = False
    closed: bool = False
    source_closed: bool = False
    text_sent: int = 0
    _materialized: dict | None = field(default=None, init=False, repr=False)

    @property
    def text_field(self) -> str | None:
        kind = self.content.get("type")
        if kind == "thinking" and (
            self.content.get("redacted")
            or self.content.get("mark") == "encrypted"
            and not self.content.get("thinking")
        ):
            return None
        return kind if kind in ("text", "thinking") else None

    def materialize(self) -> dict:
        """Borrow this version's materialized content within the request.

        Nested content is already owned by this state. Mutations replace nested
        containers and invalidate the cache, so earlier materialized versions
        remain stable without copying their entire trees.
        """
        if self._materialized is None:
            block = _located(self.content.copy(), self.content)
            for name, parts in self.fragments.items():
                block[name] = block.get(name, "") + "".join(parts)
            if block.get("type") == "tool_use":
                if self.argument_parts is not None:
                    block["input"] = arguments("".join(self.argument_parts) or "{}")
                block.setdefault("input", {})
            self._materialized = block
        return self._materialized

    def invalidate(self) -> None:
        self._materialized = None

    def replace(self, content: dict) -> None:
        name = self.text_field
        if name and self.text_sent:
            previous = (self.content.get(name, "") + "".join(self.fragments.get(name, ())))[
                : self.text_sent
            ]
            if not content.get(name, "").startswith(previous):
                raise IncompleteStreamError("final block contradicts already emitted text")
        self.content = capture(content)
        self.fragments.clear()
        self.argument_parts = None
        self.invalidate()


class StreamState:
    """Accumulate one response; tools and following blocks wait for successful EOF.

    Source references address updates, while list position fixes output order.
    Independent blocks are appended explicitly, without synthetic source keys.
    """

    def __init__(self, model: str) -> None:
        self.metadata: dict[str, Any] = {"response_id": f"generated_{uuid4().hex}", "model": model}
        self._blocks: list[BlockState] = []
        self._references: dict[tuple[str, int, int], BlockState] = {}
        self._deferred_from: int | None = None
        self.reason: str | None = None
        self.usage: dict | None = None
        self.source_protocol: str | None = None
        self.terminal = False
        self._terminal_tail: set[str] = set()
        self.started = False
        self.finished = False
        self.sequence_number = 0
        self.result: Any = None
        self._final_messages: list[dict] | None = None

    def _new_block(self, content: dict) -> BlockState:
        block = BlockState(len(self._blocks), capture(content))
        self._blocks.append(block)
        if content.get("type") == "tool_use" and self._deferred_from is None:
            self._deferred_from = block.index
        return block

    def _block(self, ref: tuple[str, int, int] | list, kind: str) -> BlockState:
        # JSON round trips turn the named tuple into a list.
        ref = tuple(ref)
        if ref not in self._references:
            content = {"type": kind}
            if kind in ("text", "thinking"):
                content[kind] = ""
            if kind == "thinking":
                content["mark"] = "raw"
            self._references[ref] = self._new_block(content)
            self._references[ref].source_ref = ref
        return self._references[ref]

    def _capture_origin(self, block: BlockState, event: dict) -> None:
        if isinstance(event, Originated) and not isinstance(block.content, Originated):
            block.content = Originated(block.content, event.origin_path, event.origin_value)
            block.invalidate()

    def _start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [{"type": "start"}]

    def _deferred(self, block: BlockState) -> bool:
        return self._deferred_from is not None and block.index >= self._deferred_from

    def feed(self, chunks: list[dict]) -> list[dict]:
        if self.finished:
            raise IncompleteStreamError("stream has already been finalized")
        result = []
        for chunk in chunks:
            origin = source_protocol(chunk)
            if origin is not None:
                if self.source_protocol not in (None, origin):
                    raise IncompleteStreamError("stream mixes source protocols")
                self.source_protocol = origin
                self.metadata["source_protocol"] = origin
            for event in operations(chunk):
                if self.terminal and event["type"] not in self._terminal_tail:
                    raise IncompleteStreamError("event received after protocol terminal event")
                result.extend(_HANDLERS[event["type"]](self, event))
        return result

    def _metadata(self, event: dict) -> list[dict]:
        target = self.metadata
        for key in event["path"]:
            child = dict(target.get(key, {}))
            target[key] = child
            target = child
        target.update(capture(event["values"]))
        return []

    def _usage(self, event: dict) -> list[dict]:
        if event["usage"] is None:
            return []
        update = event["usage"]
        usage = dict(self.usage or {})
        for name in ("total", "cached", "cache_write", "reasoning"):
            if name in update:
                usage[name] = update[name]
        for name in ("input", "output"):
            if name in update:
                usage[name] = update[name].copy()
        self.usage = usage
        return []

    def _begin(self, event: dict) -> list[dict]:
        return self._start()

    def _reason(self, event: dict) -> list[dict]:
        self.reason = event.get("reason") or self.reason
        if event["type"] == "finish":
            self.terminal = True
            self._terminal_tail = set(event.get("allow_tail", ()))
        return []

    def _snapshot(self, event: dict) -> list[dict]:
        references = {tuple(entry["ref"]) for entry in event["blocks"]}
        if self._references.keys() - references:
            raise IncompleteStreamError("final snapshot omits streamed blocks")
        # Equal text can belong to distinct blocks. Identity comes from encoders.
        for entry in event["blocks"]:
            block = self._block(entry["ref"], entry["block"]["type"])
            block.replace(entry["block"])
        return []

    def _set_block(self, event: dict) -> list[dict]:
        content = event["block"]
        block = self._block(event["ref"], content["type"])
        block.replace(content)
        if content.get("type") == "tool_use" and not block.content.get("id"):
            block.content["id"] = f"call_{uuid4().hex}"
        return self._publish(block)

    def _append_block(self, event: dict) -> list[dict]:
        block = self._new_block(event["block"])
        if block.content.get("type") == "tool_use" and not block.content.get("id"):
            block.content["id"] = f"call_{uuid4().hex}"
        block.source_closed = True
        return self._publish(block)

    def _update_block(self, event: dict) -> list[dict]:
        patch = {
            key: event["block"][key] for key in ("type", "id", "name") if key in event["block"]
        }
        block = self._block(event["ref"], patch["type"])
        self._capture_origin(block, event)
        block.content.update(patch)
        block.invalidate()
        return []

    def _delta(self, event: dict) -> list[dict]:
        name, value = event["field"], event["value"]
        block = self._block(event["ref"], event["kind"])
        self._capture_origin(block, event)
        if "mark" in event:
            block.content["mark"] = event["mark"]
        block.invalidate()
        if name == "arguments":
            if block.argument_parts is None:
                block.argument_parts = []
            block.argument_parts.append(value)
            return []
        block.fragments.setdefault(name, []).append(value)
        if name != block.text_field or self._deferred(block):
            return []
        return self._emit_text(block, value)

    def _stop_block(self, event: dict) -> list[dict]:
        block = self._references.get(tuple(event["ref"]))
        if block is None:
            return []
        block.source_closed = True
        return [] if self._deferred(block) else self._close(block)

    def _emit_text(self, block: BlockState, value: str) -> list[dict]:
        result = self._start()
        if not block.emitted:
            block.emitted = True
            result.append(
                _located(
                    {
                        "type": "block_start",
                        "index": block.index,
                        "source_ref": block.source_ref,
                        "block": block.content,
                    },
                    block.content,
                )
            )
        if value:
            block.text_sent += len(value)
            result.append(
                _located(
                    {
                        "type": "delta",
                        "index": block.index,
                        "source_ref": block.source_ref,
                        "field": block.text_field,
                        "value": value,
                    },
                    block.content,
                )
            )
        return result

    def _close(self, block: BlockState) -> list[dict]:
        if not block.emitted or block.closed or block.text_field is None:
            return []
        block.closed = True
        return [
            _located(
                {
                    "type": "block_stop",
                    "index": block.index,
                    "source_ref": block.source_ref,
                    "block": block.materialize(),
                },
                block.content,
            )
        ]

    def _publish(
        self, block: BlockState, *, final: bool = False, tool_index: int = 0
    ) -> list[dict]:
        if not final and self._deferred(block):
            return []
        if block.text_field is not None:
            content = block.materialize()
            result = self._emit_text(block, content.get(block.text_field, "")[block.text_sent :])
            if final or block.source_closed:
                result.extend(self._close(block))
            return result
        if block.emitted:
            return []
        block.emitted = True
        # These events stay inside the pipeline; public decoder outputs own
        # their values. Reuse the already validated nested content here.
        content = block.materialize()
        event = {
            "type": "block",
            "index": block.index,
            "source_ref": block.source_ref,
            "block": content,
        }
        if content.get("type") == "tool_use":
            event.update(type="tool", tool_index=tool_index)
        return [*self._start(), _located(event, block.content)]

    def messages(self, *, partial: bool = False) -> list[dict]:
        if self._final_messages is not None and not partial:
            return capture(self._final_messages)
        # Never parse unfinished tool JSON while reporting the original failure.
        content = [
            block.materialize()
            for block in self._blocks
            if not partial or block.content.get("type") != "tool_use"
        ]
        return capture(
            [{"role": "assistant", "content": content, "protocol": self.source_protocol}]
        )

    def finish(self) -> list[dict]:
        if self.finished:
            raise IncompleteStreamError("stream has already been finalized")
        if not self.terminal:
            raise IncompleteStreamError("stream ended without its protocol's terminal event")
        # Validate every call before publishing any of them or marking success.
        content = [block.materialize() for block in self._blocks]
        messages = [{"role": "assistant", "content": content, "protocol": self.source_protocol}]
        output_blocks = [
            {"index": block.index, "source_ref": block.source_ref, "block": materialized}
            for block, materialized in zip(self._blocks, content, strict=True)
        ]
        result = self._start()
        tool_index = 0
        for block in self._blocks:
            result.extend(self._publish(block, final=True, tool_index=tool_index))
            tool_index += block.content.get("type") == "tool_use"
        reason = self.reason or "stop"
        if tool_index and reason == "stop":
            reason = "tool_calls"
        result.append(
            {
                "type": "finish",
                "reason": reason,
                "messages": messages,
                "output_blocks": output_blocks,
            }
        )
        self._final_messages = messages
        self.finished = True
        return result


_HANDLERS = {
    "start": StreamState._begin,
    "metadata": StreamState._metadata,
    "usage": StreamState._usage,
    "snapshot": StreamState._snapshot,
    "reason": StreamState._reason,
    "finish": StreamState._reason,
    "block_set": StreamState._set_block,
    "block_append": StreamState._append_block,
    "block_update": StreamState._update_block,
    "block_delta": StreamState._delta,
    "block_stop": StreamState._stop_block,
}
