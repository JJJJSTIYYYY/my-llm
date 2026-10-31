"""Instance-owned ollama protocol facade."""

from copy import deepcopy

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.ollama.decoding.decoder import OllamaDecoder
from my_llm.codec.ollama.encoding.encoder import OllamaEncoder
from my_llm.codec.ollama.transport import OllamaTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class OllamaCodec(BaseCodec):
    """Ollama Chat conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "ollama" and the BaseCodec
    conversion interface. Requires the ollama optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = "ollama"
    _DEFAULTS = ProtocolDefaults(
        feedback_reasoning_content=Config(default_value="always"),
        api_key=Parameter(
            path="$.extra_headers.Authorization",
            default_value=omit,
        ),
        max_tokens=Parameter(path="$.options.num_predict", default_value=omit),
        content=Parameter(path="$.message.content", default_value=omit),
        reasoning_content=Parameter(path="$.message.thinking", default_value=omit),
        reasoning_enable=Parameter(path="$.think", default_value=True),
        reasoning_effort=Parameter(supports=False, path="$.think", default_value=omit),
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
