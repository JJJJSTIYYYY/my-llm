"""Stateless translation of recognized intermediate values and request options."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.conversion import capture
from my_llm.codec.shared.options import PreparedOptions, request_parameters
from my_llm.profile.resolution import ResolvedProfile


class BaseDecoder(ABC):
    """Translate values and request options; Transport validates the result."""

    __slots__ = ()
    body_container: str | None = None
    headers_field: str | None = "$.extra_headers"
    body_overlay: str | None = None
    omit_empty_body = False

    @property
    @abstractmethod
    def protocol(self) -> str: ...

    @final
    def request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        options: PreparedOptions,
        **parameters: Any,
    ) -> dict:
        """Translate recognized values and assemble a native request."""
        native_messages = self.messages(messages, profile, **parameters)
        native_tools = self.tools(tools, profile, **parameters) if tools is not None else None
        return self.assemble_request(
            native_messages,
            native_tools,
            profile,
            request=self.request_parameters(profile, options),
            **parameters,
        )

    @final
    def request_parameters(self, profile: ResolvedProfile, options: PreparedOptions) -> dict:
        """Apply decode declarations at SDK-root paths for every protocol."""
        request = request_parameters(
            profile,
            options,
            body_container=self.body_container,
            headers_field=self.headers_field,
            overlay=self.body_overlay,
            omit_empty_body=self.omit_empty_body,
        )
        return self._request_parameters(request, profile, options)

    def _request_parameters(
        self, request: dict, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict:
        """Add protocol-specific controls after the common parameter pipeline."""
        return request

    @abstractmethod
    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        request: dict,
        **parameters: Any,
    ) -> dict:
        """Place translated messages in the already prepared request envelope."""
        ...

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return capture(self._messages(values, profile, **parameters))

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return capture(self._tools(values, profile, **parameters))

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return self._chunks(values, profile, **parameters)

    def result(self, messages: list) -> dict | list:
        return messages[0]

    @abstractmethod
    def _messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate recognized message fields without retaining source envelopes."""
        ...

    @abstractmethod
    def _tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate recognized tool definitions and their open parameter schemas."""
        ...

    @abstractmethod
    def _chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Build independent events, capturing only borrowed recognized payloads."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None:
        """Translate token accounting."""
        ...
