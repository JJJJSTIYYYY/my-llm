"""Profile-driven asynchronous standard-protocol LLM conversion and execution."""

from openai import Omit, omit

from my_llm._version import __version__
from my_llm.codec import (
    BaseCodec,
    BaseDecoder,
    BaseEncoder,
    BaseTransport,
    MyCodec,
    PreparedOptions,
)
from my_llm.exceptions import (
    ConfigurationError,
    IncompleteStreamError,
    MessageValidationError,
    MyLLMError,
    ResponseFailedError,
    ResponseIncompleteError,
    StreamError,
    UnsupportedCapabilityError,
    UnsupportedProviderError,
)
from my_llm.invoke import MyLLM, generate
from my_llm.profile import (
    ModelProfile,
    OverridableProfile,
    Protocol,
    ProtocolDefaults,
    ProviderProfile,
    ReasoningEffort,
    ResolvedProfile,
    resolve_profiles,
)
from my_llm.stream import MessageStream

__all__ = [
    "BaseCodec",
    "BaseDecoder",
    "BaseEncoder",
    "BaseTransport",
    "MyCodec",
    "MyLLM",
    "MessageStream",
    "ModelProfile",
    "OverridableProfile",
    "ProviderProfile",
    "ResolvedProfile",
    "resolve_profiles",
    "Protocol",
    "ProtocolDefaults",
    "PreparedOptions",
    "ReasoningEffort",
    "Omit",
    "omit",
    "generate",
    "__version__",
    "ConfigurationError",
    "IncompleteStreamError",
    "MessageValidationError",
    "MyLLMError",
    "ResponseFailedError",
    "ResponseIncompleteError",
    "StreamError",
    "UnsupportedCapabilityError",
    "UnsupportedProviderError",
]
