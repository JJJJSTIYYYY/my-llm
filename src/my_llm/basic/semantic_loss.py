"""Request-local loss collection and instance-owned, directional policies."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from my_llm.basic.exceptions import ConfigurationError, SemanticLossError

type LossDirection = Literal["input", "output"]
type LossPolicy = Literal["raise", "ignore", "hook"]


@dataclass(slots=True)
class SemanticLossItem:
    origin_value: object
    origin_path: str
    reason: str = "target cannot express a supported value"


@dataclass(slots=True)
class SemanticLossGroup:
    items: list[SemanticLossItem] = field(default_factory=list)
    origin_source: LossDirection = "input"

    def __iter__(self):
        """Iterate collected losses in their reporting order."""
        return iter(self.items)

    def __len__(self):
        """Count collected semantic losses."""
        return len(self.items)


@dataclass(frozen=True, slots=True)
class SemanticLossContext:
    source_protocol: str | None
    target_protocol: str
    origin_source: LossDirection = "input"
    stage: str = "decoder"


type SemanticLossHook = Callable[[object, SemanticLossGroup, SemanticLossContext], object]


class Originated(dict):
    """Private parsing projection with provenance outside its payload keys."""

    __slots__ = ("origin_path", "origin_value")

    def __init__(self, value: dict, path: str, origin_value: object):
        """Attach a source path and original value outside the projected dictionary keys."""
        super().__init__(value)
        self.origin_path = path
        self.origin_value = getattr(origin_value, "origin_value", origin_value)

    def __deepcopy__(self, memo):
        """Capture projected payloads while retaining original provenance references."""
        from my_llm.basic.data import capture

        result = Originated({}, self.origin_path, self.origin_value)
        memo[id(self)] = result
        for key, value in self.items():
            result[key] = capture(value, memo)
        return result


def originated(value: dict, path: str, origin_value: object = None) -> dict:
    """Wrap a projection with source provenance, using the projection itself when unspecified."""
    return Originated(value, path, value if origin_value is None else origin_value)


@dataclass(slots=True)
class _LossCollection:
    context: SemanticLossContext
    items: dict[tuple[str, str], SemanticLossItem] = field(default_factory=dict)

    def complete(self, draft: object) -> object:
        """Return the finished draft or raise all deduplicated losses with that draft."""
        if self.items:
            raise SemanticLossError(
                draft,
                SemanticLossGroup(list(self.items.values()), self.context.origin_source),
                self.context,
            )
        return draft


_collection: ContextVar[_LossCollection | None] = ContextVar("my_llm_semantic_loss", default=None)


@contextmanager
def collect_losses(context: SemanticLossContext):
    """Nested decoder calls join the same collection; tasks remain independent."""
    existing = _collection.get()
    if existing is not None:
        yield existing, False
        return
    collection = _LossCollection(context)
    token = _collection.set(collection)
    try:
        yield collection, True
    finally:
        _collection.reset(token)


def report_loss(value: object, reason: str, *, path: str | None = None) -> None:
    """Record one loss and let native decoding continue to its complete draft."""
    item = SemanticLossItem(
        getattr(value, "origin_value", value),
        path or getattr(value, "origin_path", "$"),
        reason,
    )
    collection = _collection.get()
    collection.items.setdefault((item.origin_path, item.reason), item)


def _raise(draft, losses, context):
    """Implement the raising policy with the complete draft and loss context."""
    raise SemanticLossError(draft, losses, context)


def _ignore(draft, losses, context):
    """Leave the translated draft unchanged when semantic losses are ignored."""
    return None


class BaseSemanticLossAdaptor:
    """Policy selected once per converter, with separate synchronous hooks."""

    def __init__(
        self,
        semantic_loss_protocol: LossPolicy = "raise",
        input_semantic_loss_hook: SemanticLossHook | None = None,
        output_semantic_loss_hook: SemanticLossHook | None = None,
    ):
        """Validate the policy and select separate synchronous input/output handlers."""
        if semantic_loss_protocol not in ("raise", "ignore", "hook"):
            raise ConfigurationError("semantic_loss_protocol must be raise, ignore or hook")
        for hook in (input_semantic_loss_hook, output_semantic_loss_hook):
            if hook is not None and not callable(hook):
                raise ConfigurationError("semantic loss hooks must be callable")
        self.semantic_loss_protocol = semantic_loss_protocol
        if semantic_loss_protocol == "hook":
            self.input_semantic_loss_hook = (
                _ignore if input_semantic_loss_hook is None else input_semantic_loss_hook
            )
            self.output_semantic_loss_hook = (
                _ignore if output_semantic_loss_hook is None else output_semantic_loss_hook
            )
        else:
            self.input_semantic_loss_hook = self.output_semantic_loss_hook = (
                _raise if semantic_loss_protocol == "raise" else _ignore
            )

    def handle(self, error: SemanticLossError) -> object:
        """Apply the directional loss policy and return the same translated draft.

        Select the input or output hook from the loss group, passing the complete
        draft, collected losses and translation context. Hooks may mutate the draft
        in place; their return values are ignored. The raising policy propagates
        SemanticLossError, and application hook exceptions retain their own types.
        """
        hook = (
            self.input_semantic_loss_hook
            if error.losses.origin_source == "input"
            else self.output_semantic_loss_hook
        )
        hook(error.draft, error.losses, error.context)
        return error.draft
