"""Decode native content parts into intermediate blocks."""

from typing import Any

from my_llm.codec.openai.responses.encoding.messages.fields import content_fields
from my_llm.codec.shared.conversion import blocks, extras, residual


def encode_content(content: Any) -> list[dict]:
    result = []
    for block in blocks(content):
        kind = block.get("type")
        value = extras(block)
        if kind in ("input_text", "output_text"):
            value["type"] = "text"
        elif kind in ("input_image", "input_file"):
            file = block
            uri = (
                block.get("image_url")
                if kind == "input_image"
                else file.get("file_data", file.get("file_url"))
            )
            image_options = (
                {"detail": uri["detail"]} if isinstance(uri, dict) and "detail" in uri else {}
            )
            if isinstance(uri, dict):
                uri = uri.get("url")
            # Without a usable URI (for example, a file_id-only block), leave
            # the native block intact instead of consuming fields we cannot map.
            if isinstance(uri, str):
                source = {"type": "url", "url": uri}
                if uri.startswith("data:"):
                    header, _, data = uri.partition(",")
                    source = {
                        "type": "base64",
                        "media_type": header[5:].removesuffix(";base64"),
                        "data": data,
                    }
                value = {
                    **residual(block, content_fields(block)),
                    **image_options,
                    "type": "image" if kind == "input_image" else "document",
                    "source": source,
                }
        result.append(value)
    return result
