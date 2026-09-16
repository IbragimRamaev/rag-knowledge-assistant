from app.core.interfaces import EmbeddingProvider, VectorStore
from app.core.models import RetrievedChunk


class Retriever:
    """Finds the chunks most relevant to a question. Plain vector search today; swap in
    hybrid search (BM25 + vector) later without touching the RAG engine."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(
        self, question: str, top_k: int = 5, company_id: str | None = None
    ) -> list[RetrievedChunk]:
        [query_embedding] = await self._embedding_provider.embed([question])
        return await self._vector_store.search(query_embedding, top_k=top_k, company_id=company_id)
