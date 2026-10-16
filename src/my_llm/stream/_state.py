"""Request-owned accumulation of explicit, provider-independent stream operations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from my_llm.codec.utils.chunks import BlockRef, operations
from my_llm.codec.utils.conversion import arguments, dictionaries, extras, source_protocol
from my_llm.exceptions import IncompleteStreamError


@dataclass(slots=True)
class BlockState:
    """Content and lifecycle for one block; extensions never hold control data."""

    index: int
    content: dict
    fragments: dict[str, list[str]] = field(default_factory=dict)
    argument_parts: list[str] | None = None
    emitted: bool = False
    closed: bool = False
    source_closed: bool = False
    text_sent: int = 0

    @property
    def text_field(self) -> str | None:
        kind = self.content.get("type")
        return kind if kind in ("text", "thinking") else None

    def materialize(self) -> dict:
        block = deepcopy(self.content)
        for name, parts in self.fragments.items():
            block[name] = block.get(name, "") + "".join(parts)
        if block.get("type") == "tool_use":
            if self.argument_parts is not None:
                block["input"] = arguments("".join(self.argument_parts) or "{}")
            block.setdefault("input", {})
        return block

    def replace(self, content: dict) -> None:
        name = self.text_field
        if name and self.text_sent:
            previous = self.materialize().get(name, "")[: self.text_sent]
            if not content.get(name, "").startswith(previous):
                raise IncompleteStreamError("final block contradicts already emitted text")
        self.content = deepcopy(content)
        self.fragments.clear()
        self.argument_parts = None


class StreamState:
    """Accumulate one response; tools and following blocks wait for successful EOF.

    Source references address updates, while list position fixes output order.
    Independent blocks are appended explicitly, without synthetic source keys.
    """

    def __init__(self, model: str) -> None:
        self.metadata: dict[str, Any] = {"response_id": f"generated_{uuid4().hex}", "model": model}
        self._blocks: list[BlockState] = []
        self._references: dict[BlockRef, BlockState] = {}
        self._deferred_from: int | None = None
        self.snapshot: list[dict] | None = None
        self.reason: str | None = None
        self.usage: dict | None = None
        self.usage_protocol: str | None = None
        self.terminal = False
        self.started = False
        self.finished = False
        self.sequence_number = 0
        self.result: Any = None

    def _new_block(self, content: dict) -> BlockState:
        block = BlockState(len(self._blocks), deepcopy(content))
        self._blocks.append(block)
        if content.get("type") == "tool_use" and self._deferred_from is None:
            self._deferred_from = block.index
        return block

    def _block(self, ref: BlockRef | list, kind: str = "text") -> BlockState:
        # JSON round trips turn the named tuple into a list.
        ref = BlockRef(*ref)
        if ref not in self._references:
            self._references[ref] = self._new_block({"type": kind})
        return self._references[ref]

    def _start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [
            {"type": "start", "usage": deepcopy(self.usage), "usage_protocol": self.usage_protocol}
        ]

    def _deferred(self, block: BlockState) -> bool:
        return self._deferred_from is not None and block.index >= self._deferred_from

    def feed(self, chunks: list[dict]) -> list[dict]:
        if self.finished:
            raise IncompleteStreamError("stream has already been finalized")
        result = []
        for chunk in chunks:
            for event in operations(chunk):
                if event["type"] == "usage":
                    self.usage_protocol = source_protocol(chunk) or self.usage_protocol
                result.extend(_HANDLERS[event["type"]](self, event))
        return result

    def _metadata(self, event: dict) -> list[dict]:
        self.metadata.update(
            {
                ("response_id" if k == "id" else k): v
                for k, v in event.items()
                if k != "type" and v is not None
            }
        )
        return []

    def _usage(self, event: dict) -> list[dict]:
        self.usage = {**(self.usage or {}), **event["usage"]}
        return []

    def _reason(self, event: dict) -> list[dict]:
        self.reason = event.get("reason") or self.reason
        self.terminal |= event["type"] == "finish"
        return []

    def _snapshot(self, event: dict) -> list[dict]:
        references = {BlockRef(*entry["ref"]) for entry in event["blocks"]}
        if self._references.keys() - references:
            raise IncompleteStreamError("final snapshot omits streamed blocks")
        # Equal text can carry distinct citations. Identity comes from encoders.
        for entry in event["blocks"]:
            block = self._block(entry["ref"], entry["block"].get("type", "unknown"))
            block.replace(entry["block"])
        if event["messages"]:
            self.snapshot = dictionaries(event["messages"])
        return []

    def _set_block(self, event: dict) -> list[dict]:
        content = event["block"]
        block = self._block(event["ref"], content.get("type", "unknown"))
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
        patch = deepcopy(event["block"])
        block = self._block(event["ref"], patch.get("type", "unknown"))
        if "function" in patch:
            block.content.setdefault("function", {}).update(patch.pop("function"))
        block.content.update(patch)
        return []

    def _delta(self, event: dict) -> list[dict]:
        name, value = event["field"], event["value"]
        block = self._block(event["ref"], event["kind"])
        if name == "arguments":
            if block.argument_parts is None:
                block.argument_parts = []
            block.argument_parts.append(value)
            return []
        block.fragments.setdefault(name, []).append(value)
        block.content.update(extras(event.get("extras", {}), "type", name))
        if name == "signature" or self._deferred(block):
            return []
        return self._emit_text(block, value)

    def _stop_block(self, event: dict) -> list[dict]:
        block = self._references.get(BlockRef(*event["ref"]))
        if block is None:
            return []
        block.source_closed = True
        return [] if self._deferred(block) else self._close(block)

    def _opaque(self, event: dict) -> list[dict]:
        return [deepcopy(event)]

    def _emit_text(self, block: BlockState, value: str) -> list[dict]:
        result = self._start()
        if not block.emitted:
            block.emitted = True
            result.append(
                {"type": "block_start", "index": block.index, "block": deepcopy(block.content)}
            )
        if value:
            block.text_sent += len(value)
            result.append(
                {"type": "delta", "index": block.index, "field": block.text_field, "value": value}
            )
        return result

    def _close(self, block: BlockState) -> list[dict]:
        if not block.emitted or block.closed or block.text_field is None:
            return []
        block.closed = True
        return [{"type": "block_stop", "index": block.index, "block": block.materialize()}]

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
        content = extras(block.materialize())
        event = {"type": "block", "index": block.index, "block": content}
        if content.get("type") == "tool_use":
            event.update(type="tool", tool_index=tool_index)
        return [*self._start(), event]

    def messages(self, *, partial: bool = False) -> list[dict]:
        if self.snapshot is not None:
            return dictionaries(self.snapshot)
        # Never parse unfinished tool JSON while reporting the original failure.
        content = [
            block.materialize()
            for block in self._blocks
            if not partial or block.content.get("type") != "tool_use"
        ]
        return [{"role": "assistant", "content": content}]

    def finish(self) -> list[dict]:
        if self.finished:
            raise IncompleteStreamError("stream has already been finalized")
        if not self.terminal:
            raise IncompleteStreamError("stream ended without its protocol's terminal event")
        # Validate every call before publishing any of them or marking success.
        messages = self.messages()
        output_blocks = [
            {"index": block.index, "block": block.materialize()} for block in self._blocks
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
                "usage": deepcopy(self.usage),
                "usage_protocol": self.usage_protocol,
            }
        )
        self.finished = True
        return result


_HANDLERS = {
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
    "opaque": StreamState._opaque,
}
