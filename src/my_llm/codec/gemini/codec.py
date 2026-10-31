"""Instance-owned gemini protocol facade."""

from copy import deepcopy

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.gemini.decoding.decoder import GeminiDecoder
from my_llm.codec.gemini.encoding.encoder import GeminiEncoder
from my_llm.codec.gemini.transport import GeminiTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class GeminiCodec(BaseCodec):
    """Gemini Content conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "gemini" and the BaseCodec
    conversion interface. Requires the google-genai optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = "gemini"
    _DEFAULTS = ProtocolDefaults(
        feedback_reasoning_content=Config(default_value="always"),
        max_tokens=Parameter(path="$.config.max_output_tokens", default_value=omit),
        content=Parameter(
            path="$.candidates[0].content.parts[0].text",
            default_value=omit,
        ),
        reasoning_content=Parameter(
            path="$.candidates[0].content.parts[0].text",
            default_value=omit,
        ),
        reasoning_enable=Parameter(
            path="$.config.thinking_config",
            default_value={"include_thoughts": True, "thinking_budget": -1},
        ),
        reasoning_effort=Parameter(
            supports=False,
            path="$.config.thinking_config.thinking_level",
            default_value=omit,
        ),
    )

    def __init__(self) -> None:
        """Create independent Gemini defaults, encoder, decoder and transport."""
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = GeminiEncoder()
        self.decoder = GeminiDecoder()
        self.transport = GeminiTransport()

    @property
    def defaults(self) -> ProtocolDefaults:
        """Return the protocol defaults owned by this Gemini codec instance."""
        return self._defaults
