"""Whole-list matching mechanics; native shape rules belong to encoders."""

from my_llm.codec.shared.conversion import dictionaries
from my_llm.exceptions import MessageValidationError


def mapping_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def match_all(values, predicate) -> bool:
    try:
        items = dictionaries(values)
        return bool(items) and all(predicate(item) for item in items)
    except (MessageValidationError, TypeError, AttributeError, KeyError):
        return False
