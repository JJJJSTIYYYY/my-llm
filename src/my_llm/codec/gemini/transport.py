"""Reusable Gemini SDK client with request-local response-body cleanup."""

from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from google import genai

from my_llm.codec.base.transport import BaseTransport

_responses: ContextVar[AsyncExitStack | None] = ContextVar("gemini_responses", default=None)


async def _track_response(response: Any) -> None:
    resources = _responses.get()
    if resources is not None:
        resources.push_async_callback(response.aclose)


@contextmanager
def _response_scope(resources: AsyncExitStack) -> Iterator[None]:
    token = _responses.set(resources)
    try:
        yield
    finally:
        _responses.reset(token)


class GeminiTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> genai.Client:
        options = parameters["http_options"].model_copy(
            update={"async_client_args": {"event_hooks": {"response": [_track_response]}}}
        )
        return genai.Client(**{**parameters, "http_options": options})

    async def close_client(self, client: genai.Client) -> None:
        try:
            await client.aio.aclose()
        finally:
            client.close()

    @asynccontextmanager
    async def upstream(self, client: genai.Client, request: dict) -> AsyncIterator[Any]:
        async with AsyncExitStack() as resources:
            with _response_scope(resources):
                upstream = await client.aio.models.generate_content_stream(**request)
            resources.push_async_callback(upstream.aclose)

            async def events():
                while True:
                    # Reset before yielding: consumers may resume/close a stream
                    # from a different task. Only SDK reads use this request's scope.
                    with _response_scope(resources):
                        try:
                            event = await anext(upstream)
                        except StopAsyncIteration:
                            return
                    yield event

            iterator = events()
            resources.push_async_callback(iterator.aclose)
            yield iterator
