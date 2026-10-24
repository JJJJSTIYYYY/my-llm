"""Render complete native streams through request-local accumulation."""

from my_llm import MyCodec
from my_llm.runtime.state import StreamState


def render(source, target, incoming):
    codec = MyCodec()
    state = StreamState("demo")
    output = []
    for index, event in enumerate(incoming):
        chunks = codec.encode(chunks=[event], input_format=source, chunk_offset=index)["chunks"]
        converted = codec.decode(
            chunks=state.feed(chunks),
            target_format=target,
            input_format=source,
            origin_source="output",
            sequence_number=state.sequence_number,
            **state.metadata,
        )["chunks"]
        output.extend(converted)
        state.sequence_number += len(converted)
    output.extend(
        codec.decode(
            chunks=state.finish(),
            target_format=target,
            input_format=source,
            origin_source="output",
            sequence_number=state.sequence_number,
            **state.metadata,
        )["chunks"]
    )
    return output, state
