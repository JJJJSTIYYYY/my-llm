"""Instance-owned openai_responses protocol facade."""

from copy import deepcopy
from typing import Any

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from pydantic import BaseModel

import my_llm.codec.openai.responses.constants as constants
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.responses.decoding.decoder import ResponsesDecoder
from my_llm.codec.openai.responses.encoding.encoder import ResponsesEncoder
from my_llm.codec.openai.responses.transport import ResponsesTransport
from my_llm.profile.configuration import ProtocolDefaults
from my_llm.profile.types import Config, InputParameter, OutputParameter, omit

_OBJECT_TYPES = {
    "message": ResponseOutputMessage,
    "reasoning": ResponseReasoningItem,
    "function_call": ResponseFunctionToolCall,
    "function_call_output": ResponseFunctionToolCallOutputItem,
}


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

    def to_dict(self, message: BaseModel) -> dict[str, Any]:
        """Serialize one explicit Responses item, including nested SDK content."""
        return message.model_dump(mode="python", by_alias=True, exclude_unset=True)

    def to_object(self, message: dict[str, Any]) -> BaseModel:
        """Construct the native SDK variant for one supported Responses output item."""
        return _OBJECT_TYPES[message.get("type", "message")].model_construct(**message)
