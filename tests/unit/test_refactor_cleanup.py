"""Ownership and loss aggregation after removal of internal copy fallbacks."""

import pytest

from my_llm import SemanticLossError
from my_llm.basic.semantic_loss import SemanticLossContext, collect_losses, report_loss


def test_loss_deduplication_preserves_order_and_distinct_reasons():
    first, second = object(), object()
    with pytest.raises(SemanticLossError) as caught:
        with collect_losses(SemanticLossContext("source", "target")) as (losses, _):
            report_loss(first, "image unavailable", path="$.a")
            report_loss(second, "signature unavailable", path="$.b")
            report_loss(first, "image unavailable", path="$.a")
            report_loss(first, "another constraint", path="$.a")
            losses.complete({"complete": True})
    assert caught.value.draft == {"complete": True}
    assert [(item.origin_path, item.reason) for item in caught.value.losses] == [
        ("$.a", "image unavailable"),
        ("$.b", "signature unavailable"),
        ("$.a", "another constraint"),
    ]
    assert caught.value.losses.items[0].origin_value is first
    assert caught.value.losses.items[1].origin_value is second
