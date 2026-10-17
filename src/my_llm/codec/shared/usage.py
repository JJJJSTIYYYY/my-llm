"""Provider-independent token groups, cumulative merging and accounting."""

from my_llm.codec.shared.conversion import remember


def compact(value):
    if isinstance(value, dict):
        return {key: compact(child) for key, child in value.items() if child is not None}
    return value


def record(
    value,
    protocol,
    *,
    input_base=None,
    input_add=None,
    output_base=None,
    output_add=None,
    total=None,
    cached=None,
    cache_write=None,
    reasoning=None,
):
    return remember(
        compact(
            {
                "input": {"base": input_base, "add": input_add},
                "output": {"base": output_base, "add": output_add},
                "total": total,
                "cached": cached,
                "cache_write": cache_write,
                "reasoning": reasoning,
            }
        ),
        value,
        protocol,
    )


def counts(value):
    def group(name):
        data = value.get(name, {})
        base = data.get("base")
        return base + sum(data.get("add", {}).values()) if base is not None else None

    incoming, outgoing = group("input"), group("output")
    total = value.get("total")
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return (
        incoming,
        outgoing,
        total,
        value.get("cached"),
        value.get("cache_write"),
        value.get("reasoning"),
    )
