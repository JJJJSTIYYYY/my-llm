"""Request-owned SDK doubles and SSE wire serialization for tests."""

import json
from types import SimpleNamespace


class FakeStream:
    def __init__(self, events, *, failure=None):
        self.events = iter(events)
        self.failure = failure
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.failure is not None:
                raise self.failure
            raise StopAsyncIteration

    async def close(self):
        self.closed = True

    aclose = close


class FakeClient:
    def __init__(self, events, *, failure=None):
        self.events = list(events)
        self.failure = failure
        self.upstream = FakeStream(self.events, failure=failure)
        self.closed = False
        self.close_count = 0
        self.streams = []
        self.requests = []
        self.responses = SimpleNamespace(create=self.request)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))
        self.messages = SimpleNamespace(create=self.request)

    async def request(self, **kwargs):
        assert not self.closed
        self.requests.append(kwargs)
        self.upstream = FakeStream(self.events, failure=self.failure)
        self.streams.append(self.upstream)
        return self.upstream

    async def close(self):
        self.close_count += 1
        self.closed = True


def sse(events, *, anthropic=False, done=True):
    return "".join(
        (f"event: {e['type']}\n" if anthropic or e.get("type", "").startswith("response.") else "")
        + f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        for e in events
    ) + ("" if anthropic or not done else "data: [DONE]\n\n")
