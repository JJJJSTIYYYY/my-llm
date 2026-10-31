"""Instance-owned anthropic protocol facade."""

from copy import deepcopy

from my_llm.codec.anthropic.decoding.decoder import AnthropicDecoder
from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
from my_llm.codec.anthropic.transport import AnthropicTransport
from my_llm.codec.base.codec import BaseCodec
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class AnthropicCodec(BaseCodec):
    """Anthropic Messages conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "anthropic" and the
    BaseCodec conversion interface. Requires the anthropic optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = "anthropic"
    _DEFAULTS = ProtocolDefaults(
        feedback_reasoning_content=Config(default_value="always"),
        max_tokens=Parameter(path="$.max_tokens", default_value=omit),
        content=Parameter(path="$.delta.text", default_value=omit),
        reasoning_content=Parameter(path="$.delta.thinking", default_value=omit),
        reasoning_enable=Parameter(
            path="$.thinking",
            default_value={"type": "enabled", "budget_tokens": 1024},
        ),
        reasoning_effort=Parameter(
            supports=False,
            path="$.output_config.effort",
            default_value=omit,
        ),
    )

    def __init__(self) -> None:
        """Create independent Anthropic defaults, encoder, decoder and transport."""
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = AnthropicEncoder()
        self.decoder = AnthropicDecoder()
        self.transport = AnthropicTransport()

    @property
    def defaults(self) -> ProtocolDefaults:
        """Return the protocol defaults owned by this Anthropic codec instance."""
        return self._defaults
