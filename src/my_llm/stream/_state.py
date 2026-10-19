"""Request-owned accumulation of explicit, provider-independent stream operations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from my_llm.codec.shared.conversion import arguments, source_protocol
from my_llm.codec.shared.stream import BlockRef, operations
from my_llm.exceptions import IncompleteStreamError


@dataclass(slots=True)
class BlockState:
    """Recognized content and lifecycle for one block."""

    index: int
    content: dict
    source_ref: BlockRef | None = None
    fragments: dict[str, list[str]] = field(default_factory=dict)
    argument_parts: list[str] | None = None
    emitted: bool = False
    closed: bool = False
    source_closed: bool = False
    text_sent: int = 0
    pending_annotations: list[dict] = field(default_factory=list)
    _materialized: dict | None = field(default=None, init=False, repr=False)

    @property
    def text_field(self) -> str | None:
        kind = self.content.get("type")
        return kind if kind in ("text", "thinking", "refusal") else None

    def materialize(self) -> dict:
        """Borrow this version's materialized content within the request.

        Nested content is already owned by this state. Mutations replace nested
        containers and invalidate the cache, so earlier materialized versions
        remain stable without copying their entire trees.
        """
        if self._materialized is None:
            block = self.content.copy()
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
        self.content = deepcopy(content)
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
        self._references: dict[BlockRef, BlockState] = {}
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
            self._references[ref].source_ref = ref
        return self._references[ref]

    def _start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [{"type": "start", "usage": deepcopy(self.usage)}]

    def _deferred(self, block: BlockState) -> bool:
        return self._deferred_from is not None and block.index >= self._deferred_from

    def feed(self, chunks: list[dict]) -> list[dict]:
        if self.finished:
            raise IncompleteStreamError("stream has already been finalized")
        result = []
        for chunk in chunks:
            origin = source_protocol(chunk)
            if origin is not None:
                if not isinstance(origin, str):
                    raise IncompleteStreamError("stream source protocol must be a string")
                if self.source_protocol not in (None, origin):
                    raise IncompleteStreamError("stream mixes source protocols")
                self.source_protocol = origin
                self.metadata["source_protocol"] = origin
            for event in operations(chunk):
                if self.terminal and event["type"] not in self._terminal_tail:
                    raise IncompleteStreamError("event received after protocol terminal event")
                handler = _HANDLERS.get(event["type"])
                if handler is None:
                    raise IncompleteStreamError(f"unsupported stream operation: {event['type']}")
                result.extend(handler(self, event))
        return result

    def _metadata(self, event: dict) -> list[dict]:
        for key in (
            "id",
            "response_id",
            "model",
            "created",
            "created_at",
            "native_stop_reason",
            "native_stop_sequence",
        ):
            if isinstance(event.get(key), (str, int, float)):
                self.metadata["response_id" if key == "id" else key] = event[key]
        if "response_items" in event:
            headers = dict(self.metadata.get("response_items", {}))
            for index, item in event["response_items"].items():
                header = {
                    key: item[key]
                    for key in ("id", "type", "status", "name", "call_id")
                    if key in item and isinstance(item[key], str)
                }
                headers[str(index)] = {**headers.get(str(index), {}), **header}
            self.metadata["response_items"] = headers
        if "response_parameters" in event:
            # Protocol encoders project these standard settings; decoders
            # project them again at the public output boundary.
            self.metadata["response_parameters"] = {
                **self.metadata.get("response_parameters", {}),
                **deepcopy(event["response_parameters"]),
            }
        return []

    def _usage(self, event: dict) -> list[dict]:
        if event["usage"] is None:
            return []
        usage = dict(self.usage or {})
        for name in ("total", "cached", "cache_write", "reasoning"):
            if isinstance(event["usage"].get(name), int):
                usage[name] = event["usage"][name]
        for name, additions in (
            ("input", ("cache_read", "cache_write", "tools")),
            ("output", ("thinking",)),
        ):
            if name not in event["usage"]:
                continue
            group = dict(usage.get(name, {}))
            update = event["usage"][name]
            if isinstance(update.get("base"), int):
                group["base"] = update["base"]
            if "add" in update:
                group["add"] = {
                    **group.get("add", {}),
                    **{
                        key: update["add"][key]
                        for key in additions
                        if isinstance(update["add"].get(key), int)
                    },
                }
            usage[name] = group
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

    def _annotation(self, event: dict) -> list[dict]:
        block = self._block(event["ref"], event.get("kind", "text"))
        if block.source_closed:
            raise IncompleteStreamError("annotation received after source block stopped")
        name = event["field"]
        if name != "citations":
            raise IncompleteStreamError(f"unsupported stream annotation: {name}")
        block.content[name] = [*block.content.get(name, []), deepcopy(event["value"])]
        block.invalidate()
        block.pending_annotations.append(
            {
                "type": "block_annotation",
                "index": block.index,
                "field": event["field"],
                "value": deepcopy(event["value"]),
            }
        )
        return self._annotations(block) if block.emitted and not self._deferred(block) else []

    def _annotations(self, block: BlockState) -> list[dict]:
        pending, block.pending_annotations = block.pending_annotations, []
        return pending

    def _snapshot(self, event: dict) -> list[dict]:
        references = {BlockRef(*entry["ref"]) for entry in event["blocks"]}
        if self._references.keys() - references:
            raise IncompleteStreamError("final snapshot omits streamed blocks")
        # Equal text can carry distinct citations. Identity comes from encoders.
        for entry in event["blocks"]:
            block = self._block(entry["ref"], entry["block"].get("type", "unknown"))
            block.replace(entry["block"])
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
        patch = {
            key: event["block"][key]
            for key in ("type", "id", "name")
            if isinstance(event["block"].get(key), str)
        }
        block = self._block(event["ref"], patch.get("type", "unknown"))
        block.content.update(patch)
        block.invalidate()
        return []

    def _delta(self, event: dict) -> list[dict]:
        name, value = event["field"], event["value"]
        if name not in ("text", "thinking", "refusal", "arguments", "signature"):
            raise IncompleteStreamError(f"unsupported stream delta: {name}")
        block = self._block(event["ref"], event["kind"])
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
        block = self._references.get(BlockRef(*event["ref"]))
        if block is None:
            return []
        block.source_closed = True
        return [] if self._deferred(block) else self._close(block)

    def _emit_text(self, block: BlockState, value: str) -> list[dict]:
        result = self._start()
        if not block.emitted:
            block.emitted = True
            result.append(
                {
                    "type": "block_start",
                    "index": block.index,
                    "source_ref": block.source_ref,
                    "block": deepcopy(block.content),
                }
            )
        if value:
            block.text_sent += len(value)
            result.append(
                {
                    "type": "delta",
                    "index": block.index,
                    "source_ref": block.source_ref,
                    "field": block.text_field,
                    "value": value,
                }
            )
        result.extend(self._annotations(block))
        return result

    def _close(self, block: BlockState) -> list[dict]:
        if not block.emitted or block.closed or block.text_field is None:
            return []
        block.closed = True
        return [
            {
                "type": "block_stop",
                "index": block.index,
                "source_ref": block.source_ref,
                "block": block.materialize(),
            }
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
        return [*self._start(), event]

    def messages(self, *, partial: bool = False) -> list[dict]:
        if self._final_messages is not None and not partial:
            return deepcopy(self._final_messages)
        # Never parse unfinished tool JSON while reporting the original failure.
        content = [
            block.materialize()
            for block in self._blocks
            if not partial or block.content.get("type") != "tool_use"
        ]
        return deepcopy(
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
                "usage": deepcopy(self.usage),
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
    "block_annotation": StreamState._annotation,
    "block_stop": StreamState._stop_block,
}
