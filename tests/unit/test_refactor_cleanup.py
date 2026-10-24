"""Ownership and loss aggregation after removal of internal copy fallbacks."""

import json

import pytest

from my_llm import MyCodec, SemanticLossError
from my_llm.basic.semantic_loss import SemanticLossContext, collect_losses, report_loss


def test_serialized_tool_result_borrows_its_source_without_copying_it():
    class BorrowedResult(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("serialized source must not be copied")

    payload = BorrowedResult(rows=[{"value": [1, 2, 3]}])
    messages = [
        {
            "role": "user",
            "parts": [{"function_response": {"name": "f", "id": "c", "response": payload}}],
        }
    ]
    encoded = MyCodec().encode(messages, input_format="gemini")
    text = encoded.messages[0].content[0].content[0]
    assert text.origin.value is payload
    assert json.loads(text.text) == payload
    payload["rows"][0]["value"].append(4)
    assert json.loads(text.text) == {"rows": [{"value": [1, 2, 3]}]}


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
