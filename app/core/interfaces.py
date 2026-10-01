from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.core.models import ChunkData, HistoryTurn, RetrievedChunk


class EmbeddingProvider(ABC):
    """Turns text into vector embeddings. Swap implementations without touching callers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
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


class LLMProvider(ABC):
    """Generates a grounded answer from a question and retrieved context. Swap implementations
    (Claude, GPT, ...) without touching the RAG engine."""

    @abstractmethod
    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> str:
        """Answer using only context_chunks; state ignorance if they're insufficient.
        `history` (recent prior turns, if any) shapes tone only - it is not grounding."""

    @abstractmethod
    def generate_stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> AsyncIterator[str]:
        """Same contract as generate(), yielded incrementally as text deltas instead
        of returned as one string."""
