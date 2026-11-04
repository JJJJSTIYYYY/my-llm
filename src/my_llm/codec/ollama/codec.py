"""Instance-owned ollama protocol facade."""

from copy import deepcopy

import my_llm.codec.ollama.constants as constants
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.ollama.decoding.decoder import OllamaDecoder
from my_llm.codec.ollama.encoding.encoder import OllamaEncoder
from my_llm.codec.ollama.transport import OllamaTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, InputParameter, OutputParameter, omit


class OllamaCodec(BaseCodec):
    """Ollama Chat conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "ollama" and the BaseCodec
    conversion interface. Requires the ollama optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = constants.PROTOCOL
    _DEFAULTS = ProtocolDefaults(
        should_feedback_reasoning_item=Config(default_value="always"),
        api_key=InputParameter(
            path=constants.API_KEY_PATH,
            default_value=omit,
        ),
        max_tokens=InputParameter(path=constants.MAX_TOKENS_PATH, default_value=omit),
        content=OutputParameter(path=constants.TEXT_PATH, default_value=omit),
        reasoning_content=OutputParameter(path=constants.REASONING_PATH, default_value=omit),
        feedback_reasoning_item=OutputParameter(path=constants.REASONING_PATH, default_value=omit),
        reasoning_enable=InputParameter(path=constants.REASONING_ENABLE_PATH, default_value=True),
        reasoning_effort=InputParameter(
            supports=False, path=constants.REASONING_ENABLE_PATH, default_value=omit
        ),
    )

    def __init__(self) -> None:
        """Create independent Ollama defaults, encoder, decoder and transport."""
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = OllamaEncoder()
        self.decoder = OllamaDecoder()
        self.transport = OllamaTransport()

    @property
    def defaults(self) -> ProtocolDefaults:
        """Return the protocol defaults owned by this Ollama codec instance."""
        return self._defaults
