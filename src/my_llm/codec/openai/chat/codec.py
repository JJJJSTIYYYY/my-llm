"""Instance-owned openai_chat protocol facade."""

from copy import deepcopy

from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, Parameter, omit


class ChatCompletionsCodec(BaseCodec):
    """OpenAI Chat Completions conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "openai_chat" and the
    BaseCodec conversion interface. Requires the openai optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = "openai_chat"
    _DEFAULTS = ProtocolDefaults(
        feedback_reasoning_content=Config(default_value="always"),
        max_tokens=Parameter(path="$.max_completion_tokens", default_value=omit),
        content=Parameter(path="$.choices[0].delta.content", default_value=omit),
        reasoning_content=Parameter(
            path="$.choices[0].delta.reasoning_content",
            default_value=omit,
        ),
        reasoning_enable=Parameter(
            path="$.reasoning_effort",
            default_value="medium",
        ),
        reasoning_effort=Parameter(supports=False, path="$.reasoning_effort", default_value=omit),
    )

    def __init__(self) -> None:
        """Create independent Chat Completions defaults, encoder, decoder and transport."""
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = ChatCompletionsEncoder()
        self.decoder = ChatCompletionsDecoder()
        self.transport = ChatCompletionsTransport()

    @property
    def defaults(self) -> ProtocolDefaults:
        """Return the protocol defaults owned by this Chat Completions codec instance."""
        return self._defaults
