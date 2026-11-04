"""Instance-owned openai_responses protocol facade."""

from copy import deepcopy

import my_llm.codec.openai.responses.constants as constants
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.codec.openai.responses.transport import ResponsesTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, InputParameter, OutputParameter, omit


class ResponsesCodec(BaseCodec):
    """OpenAI Responses conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "openai_responses" and the
    BaseCodec conversion interface. Requires the openai optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = constants.PROTOCOL
    _DEFAULTS = ProtocolDefaults(
        should_feedback_reasoning_item=Config(default_value="always"),
        max_tokens=InputParameter(path=constants.MAX_TOKENS_PATH, default_value=omit),
        content=OutputParameter(path=constants.TEXT_PATH, default_value=omit),
        reasoning_content=OutputParameter(path=constants.TEXT_PATH, default_value=omit),
        feedback_reasoning_item=OutputParameter(path=constants.FEEDBACK_PATH, default_value=omit),
        reasoning_enable=InputParameter(
            path=constants.REASONING_ENABLE_PATH,
            default_value=constants.DEFAULT_REASONING,
        ),
        reasoning_effort=InputParameter(
            supports=True, path=constants.REASONING_EFFORT_PATH, default_value=omit
        ),
    )

    def __init__(self) -> None:
        """Create independent Responses defaults, encoder, decoder and transport."""
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = ResponsesEncoder()
        self.decoder = ResponsesDecoder()
        self.transport = ResponsesTransport()

    @property
    def defaults(self) -> ProtocolDefaults:
        """Return the protocol defaults owned by this Responses codec instance."""
        return self._defaults
