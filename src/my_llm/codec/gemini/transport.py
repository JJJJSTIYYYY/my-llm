"""Reusable Gemini SDK client with request-local response-body cleanup."""

from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from google import genai

from my_llm.codec.base.transport import BaseTransport

_responses: ContextVar[AsyncExitStack | None] = ContextVar("gemini_responses", default=None)


async def _track_response(response: Any) -> None:
    """Register an SDK HTTP response with the active request cleanup stack."""
    resources = _responses.get()
    if resources is not None:
        resources.push_async_callback(response.aclose)


@contextmanager
def _response_scope(resources: AsyncExitStack) -> Iterator[None]:
    """Bind HTTP-response tracking to this request and restore the previous task context."""
    token = _responses.set(resources)
    try:
        yield
    finally:
        _responses.reset(token)


class GeminiTransport(BaseTransport):
    __slots__ = ()

    def create_client(self, parameters: dict) -> genai.Client:
        """Create a Gemini client whose response hook tracks request-owned HTTP bodies."""
        options = parameters["http_options"].model_copy(
            update={"async_client_args": {"event_hooks": {"response": [_track_response]}}}
        )
        return genai.Client(**{**parameters, "http_options": options})

    async def close_client(self, client: genai.Client) -> None:
        """Release asynchronous client resources and always close the synchronous client."""
        try:
            await client.aio.aclose()
        finally:
            client.close()

    @asynccontextmanager
    async def upstream(self, client: genai.Client, request: dict) -> AsyncIterator[Any]:
        """Open a Gemini event iterator with request-local HTTP-response cleanup.

        Scope tracking around SDK creation and each read so responses are owned by
        this request even when consumers resume or close it from another task.
        Reset that context before yielding to the consumer. Exiting closes both
        iterators and tracked HTTP bodies while leaving the reusable client open.
        """
        async with AsyncExitStack() as resources:
            with _response_scope(resources):
                upstream = await client.aio.models.generate_content_stream(**request)
            resources.push_async_callback(upstream.aclose)

            async def events():
                """Read each SDK event inside the request scope, then yield outside that scope."""
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
