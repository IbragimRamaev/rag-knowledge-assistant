import httpx2
from openai import OpenAI

from app.config import settings
from app.core.interfaces import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by the OpenAI embeddings API."""

    def __init__(self, client: OpenAI | None = None, model: str = settings.embedding_model) -> None:
        if client is None:
            # Some environments have a brotli build incompatible with httpx2's default
            # Accept-Encoding negotiation (TypeError: process() takes no keyword arguments).
            # Force gzip/deflate to sidestep brotli entirely.
            http_client = httpx2.Client(headers={"Accept-Encoding": "gzip, deflate"})
            client = OpenAI(api_key=settings.openai_api_key, http_client=http_client)
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
