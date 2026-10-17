"""Ollama host, authentication and client-header semantics."""

from typing import Any

from ollama import AsyncClient as AsyncOllama

from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.shared.transport import owned_stream
from my_llm.exceptions import UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, client_parameters


def client_headers(profile: ResolvedProfile, **parameters: Any) -> dict:
    return merge_dicts(profile.get("extra_headers", {}), parameters.get("extra_headers"))


class OllamaTransport(BaseTransport):
    """Official SDK lifecycle and native request execution for this protocol."""

    __slots__ = ()

    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> AsyncOllama:
        values = client_parameters(profile)
        options = {key: value for key, value in values.items() if key == "timeout"}
        headers = client_headers(profile, **parameters)
        if "api_key" in values:
            headers = {"Authorization": f"Bearer {values['api_key']}", **headers}
        if headers:
            options["headers"] = headers
        return AsyncOllama(host=values.get("base_url"), **options)

    def client_key(self, profile: ResolvedProfile, **parameters: Any) -> tuple:
        return tuple(sorted(client_headers(profile, **parameters).items()))

    def validate_profile(self, profile: ResolvedProfile) -> None:
        retries = profile.get("max_retries")
        if isinstance(retries, int) and not isinstance(retries, bool) and retries > 0:
            raise UnsupportedCapabilityError("Ollama SDK does not support a positive max_retries")

    def upstream(
        self, client: AsyncOllama, request: dict, profile: ResolvedProfile, **parameters: Any
    ):
        return owned_stream(lambda: client.chat(**request), close_stream="aclose")
