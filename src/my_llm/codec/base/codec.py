"""Instance-owned protocol components and semantic-loss policy boundary."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, final

from my_llm.basic.exceptions import ConfigurationError, SemanticLossError
from my_llm.basic.semantic_loss import BaseSemanticLossAdaptor, SemanticLossContext, collect_losses
from my_llm.cache.message_cache import MessageCacheBatch
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.base.transport import BaseTransport
from my_llm.middleware.models import FunctionTool, Message
from my_llm.middleware.stream import StreamBatch
from my_llm.profile.configuration import Protocol, ProtocolDefaults
from my_llm.profile.resolution import ResolvedProfile


class BaseCodec(ABC):
    """Abstract protocol extension with encoder, decoder, transport and defaults.

    Register a zero-argument factory with MyCodec.register. Each factory call must
    create an independent codec, encoder, decoder and transport. MyCodec injects
    its instance-owned adaptor; components may retain instance-local state.
    Encoding returns the finite middleware models; decoding returns native values.
    An optional ResolvedProfile supplies standard field routes and compatibility
    settings. It cannot expand the supported middleware fields.
    """

    __slots__ = ("semantic_loss_adaptor",)

    def __init__(self, semantic_loss_adaptor: BaseSemanticLossAdaptor | None = None) -> None:
        self.semantic_loss_adaptor = (
            BaseSemanticLossAdaptor() if semantic_loss_adaptor is None else semantic_loss_adaptor
        )

    def translate(
        self,
        operation: Callable[[], Any],
        context: SemanticLossContext,
        *,
        draft_key: str | None = None,
        enable_message_cache: bool = False,
        message_cache_batch: MessageCacheBatch | None = None,
    ):
        """Publish translated messages before invoking the semantic-loss adaptor."""
        if enable_message_cache and message_cache_batch is None:
            raise ConfigurationError("message caching requires a translated batch")
        draft = None
        completed = False
        try:
            with collect_losses(context) as (losses, owner):
                draft = operation()
                completed = True
                if owner:
                    losses.complete(draft[draft_key] if draft_key is not None else draft)
        except SemanticLossError as error:
            if enable_message_cache:
                message_cache_batch.commit()
            result = self.semantic_loss_adaptor.handle(error)
            if draft_key is None:
                draft = result
            elif not completed:
                draft = {draft_key: result}
        else:
            if enable_message_cache:
                message_cache_batch.commit()
        return draft

    @property
    @abstractmethod
    def protocol_or_api_style(self) -> Protocol:
        """Native identity used only for routing and provenance."""
        ...

    @property
    @abstractmethod
    def defaults(self) -> ProtocolDefaults:
        """Standard profile settings owned and declared by this protocol."""
        ...

    @property
    @abstractmethod
    def encoder(self) -> BaseEncoder:
        """The native-to-intermediate implementation; required for every codec."""
        ...

    @property
    @abstractmethod
    def decoder(self) -> BaseDecoder:
        """The intermediate-to-native implementation; required for every codec."""
        ...

    @property
    @abstractmethod
    def transport(self) -> BaseTransport:
        """Official SDK lifecycle and execution, independent of conversion."""
        ...

    @final
    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate construction-time protocol settings through Decoder."""
        self.decoder.validate_profile(profile)

    @final
    def validate_input(self, messages: list, tools: list | None = None) -> None:
        """Validate the bound input protocol through Encoder."""
        self.encoder.validate_input(messages, tools)

    @final
    def request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        """Return validated SDK call arguments from intermediate messages and tools.

        profile selects upstream settings; parameters supplies generation controls.
        This method performs no network I/O.
        """
        self.validate_profile(profile)
        return self.translate(
            lambda: self.decoder.request(messages, tools, profile, **parameters),
            SemanticLossContext(None, self.protocol_or_api_style, "input", "request"),
        )

    @final
    def encode_messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[Message]:
        """Return supported middleware messages from native values."""
        return self.encoder.messages(messages, profile, **parameters)

    @final
    def decode_messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return native message dictionaries from intermediate messages."""
        return self.translate(
            lambda: self.decoder.messages(messages, profile, **parameters),
            SemanticLossContext(None, self.protocol_or_api_style),
        )

    @final
    def encode_tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[FunctionTool]:
        """Return supported middleware function declarations from native tools."""
        return self.encoder.tools(tools, profile, **parameters)

    @final
    def decode_tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return native tool declarations from intermediate tools."""
        return self.translate(
            lambda: self.decoder.tools(tools, profile, **parameters),
            SemanticLossContext(None, self.protocol_or_api_style),
        )

    @final
    def encode_chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[StreamBatch]:
        """Return finite middleware operation batches from native stream events."""
        return self.encoder.chunks(chunks, profile, **parameters)

    @final
    def decode_chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Return native event dictionaries from the supplied intermediate events."""
        return self.translate(
            lambda: self.decoder.chunks(chunks, profile, **parameters),
            SemanticLossContext(None, self.protocol_or_api_style),
        )

    @final
    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        """Return this protocol when the entire input conforms, otherwise None."""
        return self.protocol_or_api_style if self.encoder.matches(messages_or_tools) else None
