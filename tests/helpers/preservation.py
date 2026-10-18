"""Inspect translated protocol fields separately from retained foreign extensions."""

from copy import deepcopy

from my_llm.codec.shared.conversion import REPLAY


def _project(actual, shape):
    if isinstance(actual, dict) and isinstance(shape, dict):
        return {key: _project(actual[key], child) for key, child in shape.items() if key in actual}
    if isinstance(actual, list) and isinstance(shape, list):
        assert len(actual) == len(shape)
        return [_project(value, child) for value, child in zip(actual, shape)]
    return deepcopy(actual)


def translated(values):
    """Keep actual translated field values; discard only retained source paths."""
    result = []
    index = 0
    while index < len(values):
        record = values[index].get(REPLAY)
        if record:
            shape = record["canonical"]
            result.extend(_project(values[index : index + len(shape)], shape))
            index += len(shape)
        else:
            result.append(deepcopy(values[index]))
            index += 1
    return result
