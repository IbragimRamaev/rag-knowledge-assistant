"""
Smoke test for Step 3 (Storage): run the ingestion pipeline against a real
OpenAI embedding provider, persist the chunks in PgVectorStore, then run a
similarity search against them.

Requires OPENAI_API_KEY and DATABASE_URL to be set (via .env), and the
schema from app/db/schema.sql to already be applied.

Usage:
    python scripts/smoke_test_storage.py
"""

import asyncio

from app.config import settings
from app.ingestion.pipeline import ingest_documents
from app.providers.openai_embedding import OpenAIEmbeddingProvider
from app.storage.pg_vector_store import PgVectorStore

DOCUMENTS_DIR = "data/documents"
TEST_QUESTION = "Сколько дней отпуска положено сотруднику?"


async def main() -> None:
    embedding_provider = OpenAIEmbeddingProvider()

    chunks = ingest_documents(DOCUMENTS_DIR, embedding_provider)
    print(f"Ingested {len(chunks)} chunks from {DOCUMENTS_DIR}")

    store = PgVectorStore(settings.database_url)
    try:
        await store.save_chunks(chunks)
        print(f"Saved {len(chunks)} chunks (company_id={settings.default_company_id})\n")

        [query_embedding] = embedding_provider.embed([TEST_QUESTION])
        results = await store.search(query_embedding, top_k=3)

        print(f"Query: {TEST_QUESTION!r}\n")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. score={result.score:.4f} [{result.source_document}#{result.chunk_index}]"
            )
            print(f"   {result.text[:200]}\n")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
