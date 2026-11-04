"""Instance-owned openai_chat protocol facade."""

from copy import deepcopy

import my_llm.codec.openai.chat.constants as constants
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, InputParameter, OutputParameter, omit


class ChatCompletionsCodec(BaseCodec):
    """OpenAI Chat Completions conversion with protocol defaults and SDK transport.

    Construct without arguments. Uses protocol name "openai_chat" and the
    BaseCodec conversion interface. Requires the openai optional dependency.
    """

    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = constants.PROTOCOL
    _DEFAULTS = ProtocolDefaults(
        should_feedback_reasoning_item=Config(default_value="always"),
        max_tokens=InputParameter(path=constants.MAX_TOKENS_PATH, default_value=omit),
        content=OutputParameter(path=constants.TEXT_PATH, default_value=omit),
        reasoning_content=OutputParameter(
            path=constants.REASONING_PATH,
            default_value=omit,
        ),
        feedback_reasoning_item=OutputParameter(path=constants.REASONING_PATH, default_value=omit),
        reasoning_enable=InputParameter(
            path=constants.REASONING_EFFORT_PATH,
            default_value=constants.DEFAULT_REASONING_EFFORT,
        ),
        reasoning_effort=InputParameter(
            supports=False, path=constants.REASONING_EFFORT_PATH, default_value=omit
        ),
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
