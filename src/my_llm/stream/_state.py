"""Request-owned stream accumulation. Protocol codecs themselves hold no state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from my_llm.codec.utils.chunks import operations
from my_llm.codec.utils.conversion import arguments, dictionaries, extras, source_protocol
from my_llm.exceptions import IncompleteStreamError


class StreamState:
    """Accumulate one response and release only complete function calls."""

    def __init__(self, model: str) -> None:
        self.metadata: dict[str, Any] = {"response_id": f"generated_{uuid4().hex}", "model": model}
        self.blocks: dict[Any, dict] = {}
        # Buffer fragments outside native blocks so extensions cannot alter them.
        self.arguments: dict[Any, str] = {}
        self.order: list[Any] = []
        self.emitted: dict[Any, int] = {}
        self.closed_blocks: set[Any] = set()
        self.snapshot: list[dict] | None = None
        self.reason: str | None = None
        self.usage: dict | None = None
        self.usage_protocol: str | None = None
        self.terminal = False
        self.started = False
        self.finished = False
        self.sequence_number = 0
        self.result: Any = None

    def _block(self, key: Any, kind: str = "text") -> dict:
        if key not in self.blocks:
            self.blocks[key] = {"type": kind}
            self.order.append(key)
        return self.blocks[key]

    def _start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [
            {"type": "start", "usage": deepcopy(self.usage), "usage_protocol": self.usage_protocol}
        ]

    def _start_block(self, key: Any, block: dict) -> list[dict]:
        if key in self.emitted:
            return []
        self.emitted[key] = len(self.emitted)
        return [{"type": "block_start", "index": self.emitted[key], "block": deepcopy(block)}]

    def feed(self, chunks: list[dict]) -> list[dict]:
        result = []
        for chunk in chunks:
            for event in operations(chunk):
                kind = event.get("type")
                if kind == "metadata":
                    self.metadata.update(
                        {
                            ("response_id" if k == "id" else k): v
                            for k, v in event.items()
                            if k != "type" and v is not None
                        }
                    )
                elif kind == "usage":
                    self.usage = {**(self.usage or {}), **event["usage"]}
                    self.usage_protocol = source_protocol(chunk) or self.usage_protocol
                elif kind == "snapshot":
                    if event["messages"]:
                        self.snapshot = event["messages"]
                elif kind in ("reason", "finish"):
                    self.reason = event.get("reason") or self.reason
                    if kind == "finish":
                        self.terminal = True
                elif kind == "source_block":
                    key = event["index"]
                    block = deepcopy(event["block"])
                    self._block(key).update(block)
                    if block.get("type") in ("text", "thinking"):
                        field = "text" if block["type"] == "text" else "thinking"
                        value = block.get(field, "")
                        # Route initial text through _delta once, just like later
                        # fragments, without appending it to itself.
                        self.blocks[key][field] = ""
                        result.extend(self._delta(key, field, value, block["type"]))
                elif kind == "delta":
                    key, field, value = event["index"], event["field"], event["value"]
                    # Gemini parts have no stable stream index; keep each part
                    # distinct so signatures stay attached to their own text.
                    if key is None:
                        key = f"part_{len(self.order)}"
                    if field == "arguments":
                        self._block(key, "tool_use")
                        self.arguments[key] = self.arguments.get(key, "") + value
                    elif field == "signature":
                        block = self._block(key, "thinking")
                        block[field] = block.get(field, "") + value
                    else:
                        result.extend(self._delta(key, field, value, event.get("kind", "text")))
                        # Native extensions must not overwrite translated type/text.
                        self.blocks[key].update(extras(event.get("extras", {}), "type", field))
                elif kind == "tool_delta":
                    block = self._block(event["index"], "tool_use")
                    if event.get("id"):
                        block["id"] = event["id"]
                    block.update(extras(event, "type", "index", "id", "function", "name", "input"))
                    fn = event.get("function", {})
                    extensions = extras(fn, "name", "arguments")
                    if extensions:
                        block.setdefault("function", {}).update(extensions)
                    if fn.get("name"):
                        block["name"] = fn["name"]
                    key = event["index"]
                    self.arguments[key] = self.arguments.get(key, "") + fn.get("arguments", "")
                elif kind in ("tool_complete", "complete_block"):
                    key = event.get("index")
                    if key is None:
                        key = f"complete_{len(self.order)}"
                    block = self._block(key, event["block"].get("type", "unknown"))
                    block.clear()
                    # A complete block supersedes any argument fragments for it.
                    self.arguments.pop(key, None)
                    block.update(deepcopy(event["block"]))
                    if block.get("type") == "tool_use" and not block.get("id"):
                        block["id"] = f"call_{uuid4().hex}"
                elif kind == "source_stop":
                    result.extend(self._stop_block(event["index"]))
                elif kind == "opaque":
                    result.append(deepcopy(event))
        return result

    def _delta(self, key: Any, field: str, value: str, kind: str) -> list[dict]:
        block = self._block(key, kind)
        block[field] = block.get(field, "") + value
        if not value:
            return []
        result = self._start()
        result.extend(self._start_block(key, block))
        result.append({"type": "delta", "index": self.emitted[key], "field": field, "value": value})
        return result

    def _stop_block(self, key: Any) -> list[dict]:
        if key not in self.emitted or key in self.closed_blocks:
            return []
        self.closed_blocks.add(key)
        return [
            {"type": "block_stop", "index": self.emitted[key], "block": deepcopy(self.blocks[key])}
        ]

    def messages(self, *, partial: bool = False) -> list[dict]:
        if self.snapshot is not None:
            return dictionaries(self.snapshot)
        content = []
        for key in self.order:
            block = deepcopy(self.blocks[key])
            if block.get("type") == "tool_use":
                # Error reporting must not parse unfinished JSON and mask the
                # provider failure, or expose a call that cannot yet be executed.
                if partial:
                    continue
                if key in self.arguments:
                    block["input"] = arguments(self.arguments[key] or "{}")
                block.setdefault("input", {})
            content.append(block)
        return [{"role": "assistant", "content": content}]

    def finish(self) -> list[dict]:
        # EOF alone is not successful completion; require a protocol terminal event.
        if self.finished or not self.terminal:
            raise IncompleteStreamError("stream ended without its protocol's terminal event")
        self.finished = True
        messages = self.messages()
        content = [block for message in messages for block in message.get("content", [])]
        result = self._start()
        # A final authoritative snapshot can contain text without delta events.
        for kind, field in (("text", "text"), ("thinking", "thinking")):
            if not any(block.get("type") == kind for block in self.blocks.values()):
                for block in content:
                    if block.get("type") == kind and block.get(field):
                        key = f"final_{len(self.order)}"
                        result.extend(self._delta(key, field, block[field], kind))
                        self.blocks[key].update(extras(block))
        output_blocks = []
        for key, index in self.emitted.items():
            block = self.blocks[key]
            # Prefer authoritative annotations/signatures when the snapshot and
            # streamed text describe the same block.
            authoritative = next(
                (
                    b
                    for b in content
                    if b.get("type") == block.get("type")
                    and b.get("text", b.get("thinking")) == block.get("text", block.get("thinking"))
                ),
                block,
            )
            self.blocks[key] = {**block, **authoritative}
            result.extend(self._stop_block(key))
            output_blocks.append({"index": index, "block": deepcopy(self.blocks[key])})
        tool_index = 0
        for block in content:
            kind = block.get("type")
            if kind in ("text", "thinking"):
                continue
            index = len(output_blocks)
            output_blocks.append({"index": index, "block": extras(block)})
            # Publish calls only after arguments are complete and validated.
            if kind == "tool_use":
                result.append(
                    {
                        "type": "tool",
                        "index": index,
                        "tool_index": tool_index,
                        "block": extras(block),
                    }
                )
                tool_index += 1
            else:
                result.append({"type": "block", "index": index, "block": extras(block)})
        reason = self.reason or "stop"
        # Infer tool_calls only from normal completion; preserve length/special exits.
        if tool_index and reason == "stop":
            reason = "tool_calls"
        result.append(
            {
                "type": "finish",
                "reason": reason,
                "messages": messages,
                "output_blocks": output_blocks,
                "usage": self.usage,
                "usage_protocol": self.usage_protocol,
            }
        )
        return result
