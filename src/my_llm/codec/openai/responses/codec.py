"""Instance-owned openai_responses protocol facade."""

from copy import deepcopy

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.codec.openai.responses.transport import ResponsesTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class ResponsesCodec(BaseCodec):
    """OpenAI Responses conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "openai_responses" and the
    BaseCodec conversion interface. Requires the openai optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = "openai_responses"
    _DEFAULTS = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(default_value=True),
        max_tokens=Parameter(path="$.max_output_tokens", default_value=omit),
        content=Parameter(path="$.delta", default_value=omit),
        reasoning_content=Parameter(path="$.delta", default_value=omit),
        reasoning_enable=Parameter(
            path="$.reasoning",
            default_value={"effort": "medium", "summary": "detailed"},
        ),
        reasoning_effort=Parameter(supports=True, path="$.reasoning.effort", default_value=omit),
    )

    def __init__(self) -> None:
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = ResponsesEncoder()
        self.decoder = ResponsesDecoder()
        self.transport = ResponsesTransport()

    @property
    def defaults(self) -> ProtocolDefaults:
        return self._defaults
