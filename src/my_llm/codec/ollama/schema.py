"""Read the public SDK method contract without constructing a client."""

from typing import Any

from ollama import AsyncClient, Image, Message, Options
from pydantic import ValidationError

from my_llm.basic.data import read
from my_llm.basic.exceptions import MessageValidationError
from my_llm.codec.shared.schema import parameter_types, projected, validate_values

REQUEST_SCHEMA = {**parameter_types(AsyncClient.chat), "options": Options}


def validate_ollama_values(values: list, schema: Any, *, filtered: bool = False) -> None:
    if schema is Message and isinstance(values, list):
        # The public chat API wraps wire-format base64 strings in Image
        # before validating Message. Mirror that SDK boundary without
        # changing the dictionaries used for conversion or serialization.
        try:
            normalized = []
            for value in values:
                message = value if filtered else projected(value, Message)
                images = read(value, "images")
                if images:
                    message = {
                        **message,
                        "images": [
                            image if isinstance(image, Image) else Image(value=image)
                            for image in images
                        ],
                    }
                normalized.append(message)
            values = normalized
            filtered = True
        except (ValidationError, TypeError, ValueError) as exc:
            raise MessageValidationError("images do not conform to the native protocol") from exc
    validate_values(values, schema, filtered=filtered)
