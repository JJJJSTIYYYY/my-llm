"""Instance-owned gemini protocol facade."""

from copy import deepcopy

import my_llm.codec.gemini.constants as constants
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.gemini.decoding.decoder import GeminiDecoder
from my_llm.codec.gemini.encoding.encoder import GeminiEncoder
from my_llm.codec.gemini.transport import GeminiTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, InputParameter, OutputParameter, omit


class GeminiCodec(BaseCodec):
    """Gemini Content conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "gemini" and the BaseCodec
    conversion interface. Requires the google-genai optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = constants.PROTOCOL
    _DEFAULTS = ProtocolDefaults(
        should_feedback_reasoning_item=Config(default_value="always"),
        max_tokens=InputParameter(path=constants.MAX_TOKENS_PATH, default_value=omit),
        content=OutputParameter(
            path=constants.TEXT_PATH,
            default_value=omit,
        ),
        reasoning_content=OutputParameter(
            path=constants.TEXT_PATH,
            default_value=omit,
        ),
        feedback_reasoning_item=OutputParameter(path=constants.FEEDBACK_PATH, default_value=omit),
        reasoning_enable=InputParameter(
            path=constants.REASONING_ENABLE_PATH,
            default_value=constants.DEFAULT_REASONING,
        ),
        reasoning_effort=InputParameter(
            supports=False,
            path=constants.REASONING_EFFORT_PATH,
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
