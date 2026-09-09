from abc import ABC, abstractmethod

from app.core.models import ChunkData, RetrievedChunk


class EmbeddingProvider(ABC):
    """Turns text into vector embeddings. Swap implementations without touching callers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""


class VectorStore(ABC):
    """Persists embedded chunks and retrieves the most similar ones. Swap implementations
    (pgvector, Qdrant, ...) without touching retrieval or API code."""

    @abstractmethod
    async def save_chunks(self, chunks: list[ChunkData], company_id: str | None = None) -> None:
        """Persist chunks (each must already have an embedding), scoped to company_id."""

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 5, company_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to query_embedding, best match first."""
