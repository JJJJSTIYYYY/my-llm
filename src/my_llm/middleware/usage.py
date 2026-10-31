"""Tolerant normalization of optional token accounting."""


def normalize_usage(value: object) -> dict | None:
    """Return known nonnegative integer counters, or None for malformed usage.

    A malformed known field invalidates this sample. Unknown fields are ignored;
    absent counters stay absent and booleans are never accepted as token counts.
    """
    if not isinstance(value, dict):
        return None
    result = {}
    for name in ("input", "output", "total", "cached", "cache_write", "reasoning"):
        item = value.get(name)
        if item is None:
            continue
        if name in ("input", "output"):
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                result[name] = {"base": item}
                continue
            if not isinstance(item, dict):
                return None
            base, additions = item.get("base"), item.get("add", {})
            if (
                not isinstance(base, int)
                or isinstance(base, bool)
                or base < 0
                or not isinstance(additions, dict)
                or any(
                    not isinstance(n, int) or isinstance(n, bool) or n < 0
                    for n in additions.values()
                )
            ):
                return None
            result[name] = {"base": base, **({"add": dict(additions)} if additions else {})}
        elif not isinstance(item, int) or isinstance(item, bool) or item < 0:
            return None
        else:
            result[name] = item
    return result or None
