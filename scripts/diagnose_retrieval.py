"""
Diagnostic: for a single question, embed it and run PgVectorStore.search()
directly, then print every retrieved chunk's source_document, score, and full
text. Unlike scripts/smoke_test_rag.py, this doesn't call the LLM or truncate
chunk text - it exists purely to inspect why a particular document is being
retrieved.

Requires OPENAI_API_KEY and DATABASE_URL to be set (via .env), and the
knowledge base already ingested (scripts/smoke_test_rag.py or POST /ingest).

Usage:
    python scripts/diagnose_retrieval.py "Сколько дней отпуска положено сотруднику?"
"""

import asyncio
import sys

from app.config import settings
from app.providers.openai_embedding import OpenAIEmbeddingProvider
from app.storage.pg_vector_store import PgVectorStore

DEFAULT_QUESTION = "Сколько дней отпуска положено сотруднику?"


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    embedding_provider = OpenAIEmbeddingProvider()
    store = PgVectorStore(settings.database_url)
    try:
        [query_embedding] = await embedding_provider.embed([question])
        chunks = await store.search(query_embedding, top_k=5)

        print(f"Question: {question!r}")
        print(f"Retrieved {len(chunks)} chunks:\n")

        for rank, chunk in enumerate(chunks, start=1):
            print(f"--- #{rank}  {chunk.source_document}  score={chunk.score:.4f} ---")
            print(chunk.text)
            print()
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
