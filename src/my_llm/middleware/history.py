"""Tool-history repair fused with structural validation and native iteration."""

from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass
from uuid import uuid4

from my_llm.basic.exceptions import MessageValidationError

MESSAGE_ROLES = ("system", "developer", "user", "assistant")


@dataclass(slots=True)
class _Message:
    value: dict
    had_content: bool = False
    active: bool = False
    discarded: bool = False


@dataclass(slots=True)
class _Tool:
    call: dict | None = None
    result: dict | None = None
    padding: _Message | None = None
    opaque: bool = False


@dataclass(slots=True)
class _Saved:
    call: dict | None
    result: dict | None
    result_size: int | None
    padding_active: bool
    opaque: bool


class HistoryNormalizer:
    """Repair tool associations in the existing structural-validation visit.

    Results without an explicit ID or earlier call discard their entire message.
    Every accepted call reserves an empty result while its block is visited. A
    later valid result suppresses that reservation; discarded results leave it
    empty. Result parts append in order, including repeated content, as the nested
    structural validator visits them. No final tool-table scan or message-list
    rebuild is needed. Calls register by ID; later results resolve through that
    map without looking ahead or rewinding the source-message iterator.

    Native translation consumes this request-local iterable in its existing
    message loop. Inactive discarded/duplicate messages and superseded padding
    never enter that loop's body. Caller messages and result parts stay borrowed
    until normal SDK ownership capture; only repair envelopes are mutated here.
    """

    def __init__(self, *, allow_external_calls: bool = False) -> None:
        """Initialize association state and output slots for one request batch."""
        self.allow_external_calls = allow_external_calls
        self.messages: list[_Message] = []
        self.tools: dict[str, _Tool] = {}
        self.current: _Message | None = None
        self.current_result: _Tool | None = None
        self.error: str | None = None
        self.count = 0
        self.saved: dict[str, _Saved | None] = {}

    def __len__(self) -> int:
        """Return retained message count without consuming the output iterable."""
        return self.count

    def __iter__(self) -> Iterator[dict]:
        """Drive the existing native-message iteration without a materialization pass."""
        for message in self.messages:
            if message.active:
                yield message.value

    def _activate(self, message: _Message, active: bool) -> None:
        """Update output visibility and its count without rescanning history."""
        self.count += int(active) - int(message.active)
        message.active = active

    def observe_message(self, message: dict) -> None:
        """Start an owned envelope; defer publication until its validation ends."""
        if message.get("role") not in MESSAGE_ROLES and self.error is None:
            self.error = "invalid message role"
        value = copy(message)
        value["content"] = []
        self.current = _Message(value, bool(message.get("content")))
        self.messages.append(self.current)
        self.saved = {}
        self.current_result = None

    def end_message(self) -> None:
        """Publish this message in the same outer structural-validation loop."""
        message = self.current
        self._activate(
            message,
            not message.discarded and (bool(message.value["content"]) or not message.had_content),
        )
        self.current_result = None

    def _remember(self, call_id: str) -> None:
        """Journal only the associations touched by this message, for whole-message discard."""
        if call_id in self.saved:
            return
        tool = self.tools.get(call_id)
        if tool is None:
            self.saved[call_id] = None
            return
        result = tool.result
        self.saved[call_id] = _Saved(
            copy(tool.call) if tool.call is not None else None,
            copy(result) if result is not None else None,
            len(result["content"])
            if result is not None and isinstance(result["content"], list) and not tool.opaque
            else None,
            tool.padding.active if tool.padding is not None else False,
            tool.opaque,
        )

    def _discard_message(self) -> None:
        """Undo only this failed message's writes; never scan the complete history."""
        self.current.discarded = True
        self.current.value["content"].clear()
        self.current_result = None
        for call_id, saved in self.saved.items():
            tool = self.tools[call_id]
            if saved is None:
                if tool.padding is not None:
                    self._activate(tool.padding, False)
                del self.tools[call_id]
                continue
            if saved.call is not None:
                tool.call.clear()
                tool.call.update(saved.call)
            if saved.result is None:
                tool.result = None
            else:
                if saved.result_size is not None:
                    del saved.result["content"][saved.result_size :]
                tool.result.clear()
                tool.result.update(saved.result)
            if tool.padding is not None:
                self._activate(tool.padding, saved.padding_active)
            tool.opaque = saved.opaque

    def observe_content(self, block: dict) -> None:
        """Filter tools and reserve missing-result padding during the block visit."""
        self.current_result = None
        owner = self.current
        if owner.discarded:
            return
        kind = block["type"]
        if kind not in ("tool_use", "tool_result"):
            owner.value["content"].append(block)
            return
        key = "id" if kind == "tool_use" else "tool_use_id"
        call_id = block[key]
        if kind == "tool_result" and (
            not isinstance(call_id, str)
            or not call_id
            or call_id not in self.tools
            and not self.allow_external_calls
        ):
            self._discard_message()
            return
        value = copy(block)
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_repaired_{uuid4().hex}"
            value[key] = call_id
        self._remember(call_id)
        tool = self.tools.setdefault(call_id, _Tool())
        if kind == "tool_use":
            if tool.call is not None:
                if (
                    tool.call["name"] == value["name"]
                    and not tool.call.get("input")
                    and value.get("input")
                ):
                    tool.call["input"] = value["input"]
                return
            if tool.result is not None:
                return  # This ID already denotes a completed external call.
            tool.call = value
            owner.value["content"].append(value)
            tool.padding = _Message(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "name": value["name"],
                            "content": [],
                        }
                    ],
                }
            )
            self.messages.append(tool.padding)
            self._activate(tool.padding, True)
            return
        if tool.result is None:
            value["content"] = []
            if tool.call is not None:
                value["name"] = tool.call["name"]
            tool.result = value
            owner.value["content"].append(value)
            if tool.padding is not None:
                self._activate(tool.padding, False)
        self.current_result = tool

    def observe_result_part(self, part: dict) -> None:
        """Append a part without deduplication during its existing validation visit."""
        if self.current_result is None:
            return
        content = self.current_result.result["content"]
        if not self.current_result.opaque and isinstance(content, list):
            content.append(part)

    def observe_result_value(self, value: object) -> None:
        """Keep opaque application data without guessing or serializing new values."""
        if self.current_result is not None and not self.current_result.result["content"]:
            self.current_result.result["content"] = value
            self.current_result.opaque = True

    def finish(self, *, has_cached_history: bool = False) -> "HistoryNormalizer":
        """Perform constant-time final checks; never invent an empty user prompt."""
        if self.error is not None:
            raise MessageValidationError(self.error)
        if not self.count and not has_cached_history:
            raise MessageValidationError("messages contain no usable history")
        return self
