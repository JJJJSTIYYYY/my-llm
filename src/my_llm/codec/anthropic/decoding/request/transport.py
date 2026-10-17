"""Request-owned Anthropic Messages SDK transport."""

from typing import Any

from anthropic import AsyncAnthropic

from my_llm.codec.shared.transport import owned_stream
from my_llm.profile.resolution import ResolvedProfile, client_parameters


def open_stream(request: dict, profile: ResolvedProfile, **parameters: Any):
    client = AsyncAnthropic(**client_parameters(profile))
    return owned_stream(client, lambda: client.messages.create(**request))
