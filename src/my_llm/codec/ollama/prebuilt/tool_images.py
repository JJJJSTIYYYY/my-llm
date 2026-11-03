"""Optional Ollama tool-image relocation with asynchronous URL downloads."""

from asyncio import TaskGroup
from base64 import b64encode
from weakref import WeakSet

from httpx import AsyncClient, HTTPError, InvalidURL

from my_llm.basic.exceptions import MessageValidationError
from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.ollama.encoding.content import media_type
from my_llm.codec.shared.prebuilt.tool_images import append_image_message, tool_image_content

_completed: WeakSet[object] = WeakSet()


async def _download(client: AsyncClient, url: str) -> dict:
    """Download a supported image; expected HTTP/media failures become a marker."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = b64encode(response.content).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "data": data, "media_type": media_type(data)},
        }
    except (HTTPError, InvalidURL, MessageValidationError):
        return {"type": "text", "text": "[Unsupported Image]"}


def _message(content: list[dict]) -> dict:
    images = [part["source"]["data"] for part in content if part["type"] == "image"]
    return {
        "role": "user",
        "content": "\n".join(part["text"] for part in content if part["type"] == "text")
        or "Images returned by tools.",
        **({"images": images} if images else {}),
    }


async def move_tool_images_to_user_message(
    draft: object, losses: SemanticLossGroup, context: SemanticLossContext
) -> None:
    """Download URL images, move base64 data, and cache the complete repair once."""
    content = tool_image_content(
        draft,
        losses,
        context,
        protocol="ollama",
        reason="Ollama tool results cannot represent images",
        message_key="messages",
        completed=_completed,
    )
    if not content:
        return
    urls = dict.fromkeys(
        part["source"]["url"]
        for part in content
        if part["type"] == "image" and part["source"]["type"] == "url"
    )
    if urls:
        async with AsyncClient(timeout=10.0, follow_redirects=True) as client:
            async with TaskGroup() as group:
                downloads = {url: group.create_task(_download(client, url)) for url in urls}
            content = [
                downloads[part["source"]["url"]].result()
                if part["type"] == "image" and part["source"]["type"] == "url"
                else part
                for part in content
            ]
    append_image_message(draft, context, "messages", _message(content), _completed)
