"""Instance-owned anthropic protocol facade."""

from copy import deepcopy

import my_llm.codec.anthropic.constants as constants
from my_llm.codec.anthropic.decoding.decoder import AnthropicDecoder
from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder
from my_llm.codec.anthropic.transport import AnthropicTransport
from my_llm.codec.base.codec import BaseCodec
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, InputParameter, OutputParameter, omit


class AnthropicCodec(BaseCodec):
    """Anthropic Messages conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "anthropic" and the
    BaseCodec conversion interface. Requires the anthropic optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = constants.PROTOCOL
    _DEFAULTS = ProtocolDefaults(
        should_feedback_reasoning_item=Config(default_value="always"),
        max_tokens=InputParameter(path=constants.MAX_TOKENS_PATH, default_value=omit),
        content=OutputParameter(path=constants.TEXT_PATH, default_value=omit),
        reasoning_content=OutputParameter(path=constants.REASONING_PATH, default_value=omit),
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
