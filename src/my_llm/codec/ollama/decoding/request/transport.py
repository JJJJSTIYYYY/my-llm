"""Ollama host, authentication and client-header semantics."""

from typing import Any

from ollama import AsyncClient as AsyncOllama

from my_llm.codec.shared.transport import owned_stream
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, client_parameters


def open_stream(request: dict, profile: ResolvedProfile, **parameters: Any):
    values = client_parameters(profile)
    options = {key: value for key, value in values.items() if key == "timeout"}
    headers = merge_dicts(profile.get("extra_headers", {}), parameters.get("extra_headers"))
    if "api_key" in values:
        headers = {"Authorization": f"Bearer {values['api_key']}", **headers}
    if headers:
        options["headers"] = headers
    client = AsyncOllama(host=values.get("base_url"), **options)
    return owned_stream(client, lambda: client.chat(**request), close_stream="aclose")
