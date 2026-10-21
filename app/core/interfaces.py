from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.core.models import (
    ChunkData,
    GenerationResult,
    HistoryTurn,
    RetrievedChunk,
    StreamSources,
    StreamTextChunk,
)


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

    @abstractmethod
    async def ping(self) -> None:
        """Cheap liveness check - raise if the store isn't reachable. Used by
        GET /health; must not do anything as expensive as an actual search."""


class LLMProvider(ABC):
    """Generates a grounded answer from a question and retrieved context. Swap implementations
    (Claude, GPT, ...) without touching the RAG engine."""

    @abstractmethod
    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> GenerationResult:
        """Answer using only context_chunks; state ignorance if they're insufficient.
        `history` (recent prior turns, if any) shapes tone only - it is not grounding.
        `cited_documents` on the result is whichever of context_chunks' source
        documents the model actually relied on, if it's able to say - empty if not."""

    @abstractmethod
    def generate_stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> AsyncIterator[StreamTextChunk | StreamSources]:
        """Same contract as generate(), streamed: text chunks as they're generated,
        then exactly one final StreamSources event carrying cited_documents (as
        generate() does, this is empty if the model didn't cite anything)."""
